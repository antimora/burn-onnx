use super::prelude::*;
use onnx_ir::ir::{ArgType, DType};
use onnx_ir::node::range::RangeInput;
use proc_macro2::Literal;

/// A static bound as `f64` (for the codegen-time element count) and as a literal
/// typed like the range element, so it mixes with runtime scalars of that type.
/// Returns `None` for a runtime bound.
fn static_bound(input: &RangeInput, dtype: &DType) -> Option<(f64, TokenStream)> {
    match (input, dtype) {
        (RangeInput::StaticFloat(v), DType::F32) => {
            Some((*v, super::super::codegen::f32_to_tokens(*v as f32)))
        }
        (RangeInput::StaticFloat(v), _) => Some((*v, super::super::codegen::f64_to_tokens(*v))),
        (RangeInput::Static(v), DType::I32) => {
            let lit = Literal::i32_suffixed(*v as i32);
            Some((*v as f64, quote! { #lit }))
        }
        (RangeInput::Static(v), DType::I16) => {
            let lit = Literal::i16_suffixed(*v as i16);
            Some((*v as f64, quote! { #lit }))
        }
        (RangeInput::Static(v), _) => {
            let lit = Literal::i64_suffixed(*v);
            Some((*v as f64, quote! { #lit }))
        }
        (RangeInput::Runtime(_), _) => None,
    }
}

impl NodeCodegen for onnx_ir::node::range::RangeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Each parameter is used more than once below, so a static literal or a
        // scalar tensor readback is bound to a local first. Native scalars are
        // plain `Copy` idents and are used directly. Returns the binding
        // statement (possibly empty) and the expression to use.
        let elem_dtype = self.outputs.first().unwrap().ty.elem_type();
        let range_param_tokens = |config: &RangeInput,
                                  inputs: &[Argument],
                                  scope: &mut super::super::scope::ScopeAtPosition<'_>,
                                  local: &str|
         -> (TokenStream, TokenStream) {
            let local = Ident::new(local, Span::call_site());
            match (config, static_bound(config, &elem_dtype)) {
                (_, Some((_, literal))) => (quote! { let #local = #literal; }, quote! { #local }),
                (RangeInput::Runtime(runtime_ref), None) => {
                    let arg = &inputs[runtime_ref.input_index];
                    match &arg.ty {
                        ArgType::ScalarNative(_) => {
                            let name = arg_to_ident(arg);
                            (quote! {}, quote! { #name })
                        }
                        ArgType::ScalarTensor(dtype) => {
                            let tensor = scope.arg(arg);
                            let native = on_device_to_native(quote! { #tensor }, dtype);
                            (quote! { let #local = #native; }, quote! { #local })
                        }
                        _ => panic!("Range parameter must be a scalar"),
                    }
                }
                _ => panic!("Range parameter must be a scalar"),
            }
        };

        let output_dtype = elem_dtype.to_tokens();
        // `arange` yields an Int tensor and `cast` keeps the kind, so a float
        // range has to switch kind with `.float()` first.
        let to_output = if elem_dtype.is_float() {
            quote! { .float().cast(#output_dtype) }
        } else {
            quote! { .cast(#output_dtype) }
        };
        let zero = if elem_dtype.is_float() {
            Literal::f64_unsuffixed(0.0)
        } else {
            Literal::i64_unsuffixed(0)
        };

        // Use formula: output[i] = start + i * delta, for i in 0..n
        // where n = max(ceil((limit - start) / delta), 0)
        // This correctly handles both positive and negative delta.
        match (
            static_bound(&self.config.start, &elem_dtype),
            static_bound(&self.config.limit, &elem_dtype),
            static_bound(&self.config.delta, &elem_dtype),
        ) {
            (Some((s, s_lit)), Some((l, _)), Some((d, d_lit))) => {
                // All static: precompute n at codegen time
                let n = ((l - s) / d).ceil().max(0.0) as i64;
                let n_lit = Literal::i64_suffixed(n);
                quote! {
                    let #output = Tensor::arange(0..#n_lit, &self.device)
                        #to_output
                        .mul_scalar(#d_lit)
                        .add_scalar(#s_lit);
                }
            }
            _ => {
                // At least one runtime value: compute n at runtime
                let (bind_start, start) =
                    range_param_tokens(&self.config.start, &self.inputs, scope, "start");
                let (bind_limit, limit) =
                    range_param_tokens(&self.config.limit, &self.inputs, scope, "limit");
                let (bind_delta, delta) =
                    range_param_tokens(&self.config.delta, &self.inputs, scope, "delta");
                quote! {
                    let #output = {
                        #bind_start
                        #bind_limit
                        #bind_delta
                        assert!(#delta != #zero);
                        let n = ((#limit - #start) as f64 / #delta as f64)
                            .ceil().max(0.0) as i64;
                        Tensor::arange(0..n, &self.device)
                            #to_output
                            .mul_scalar(#delta)
                            .add_scalar(#start)
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::node::range::{RangeConfig, RangeInput, RangeNodeBuilder};

    #[test]
    fn test_range_static() {
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Static(10),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..5i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(2i64)
                .add_scalar(0i64);
            output
        }
        ");
    }

    #[test]
    fn test_range_negative_delta() {
        let config = RangeConfig::new(
            RangeInput::Static(10),
            RangeInput::Static(0),
            RangeInput::Static(-2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..5i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(-2i64)
                .add_scalar(10i64);
            output
        }
        ");
    }

    #[test]
    fn test_range_runtime() {
        let config = RangeConfig::new(
            RangeInput::Runtime(RuntimeInputRef {
                name: "start".to_string(),
                input_index: 0,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 1,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "delta".to_string(),
                input_index: 2,
            }),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar("start", DType::I64)
            .input_scalar("limit", DType::I64)
            .input_scalar("delta", DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, start: i64, limit: i64, delta: i64) -> Tensor<1, Int> {
            let output = {
                assert!(delta != 0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_mixed_static_and_tensor() {
        // A static literal and a scalar tensor readback are each bound once;
        // the native scalar is used directly.
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 0,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "delta".to_string(),
                input_index: 1,
            }),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar_tensor("limit", DType::I64)
            .input_scalar("delta", DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, limit: Tensor<1, Int>, delta: i64) -> Tensor<1, Int> {
            let output = {
                let start = 0i64;
                let limit = (limit).into_scalar::<i64>();
                assert!(delta != 0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_empty() {
        // start >= limit with positive delta produces empty range
        let config = RangeConfig::new(
            RangeInput::Static(10),
            RangeInput::Static(0),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..0i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(2i64)
                .add_scalar(10i64);
            output
        }
        ");
    }

    #[test]
    fn test_range_static_float() {
        let config = RangeConfig::new(
            RangeInput::StaticFloat(1.5),
            RangeInput::StaticFloat(5.0),
            RangeInput::StaticFloat(0.5),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1> {
            let output = Tensor::arange(0..7i64, &self.device)
                .float()
                .cast(burn::tensor::DType::F32)
                .mul_scalar(0.5f32)
                .add_scalar(1.5f32);
            output
        }
        ");
    }

    #[test]
    fn test_range_static_f64() {
        let config = RangeConfig::new(
            RangeInput::StaticFloat(0.0),
            RangeInput::StaticFloat(1.0),
            RangeInput::StaticFloat(0.1),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1> {
            let output = Tensor::arange(0..10i64, &self.device)
                .float()
                .cast(burn::tensor::DType::F64)
                .mul_scalar(0.1f64)
                .add_scalar(0f64);
            output
        }
        ");
    }

    #[test]
    fn test_range_float_mixed() {
        // Static bounds are emitted as f32 so they mix with the runtime f32 limit.
        let config = RangeConfig::new(
            RangeInput::StaticFloat(0.0),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 0,
            }),
            RangeInput::StaticFloat(0.25),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar("limit", DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, limit: f32) -> Tensor<1> {
            let output = {
                let start = 0f32;
                let delta = 0.25f32;
                assert!(delta != 0.0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .float()
                    .cast(burn::tensor::DType::F32)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_float_scalar_tensor() {
        let config = RangeConfig::new(
            RangeInput::Runtime(RuntimeInputRef {
                name: "start".to_string(),
                input_index: 0,
            }),
            RangeInput::StaticFloat(4.0),
            RangeInput::StaticFloat(1.0),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar_tensor("start", DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, start: Tensor<1>) -> Tensor<1> {
            let output = {
                let start = (start).into_scalar::<f32>();
                let limit = 4f32;
                let delta = 1f32;
                assert!(delta != 0.0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .float()
                    .cast(burn::tensor::DType::F32)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_int32_mixed() {
        // Static bounds are emitted as i32 so they mix with the runtime i32 limit.
        let config = RangeConfig::new(
            RangeInput::Static(1),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 0,
            }),
            RangeInput::Static(3),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar("limit", DType::I32)
            .output_tensor("output", 1, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, limit: i32) -> Tensor<1, Int> {
            let output = {
                let start = 1i32;
                let delta = 3i32;
                assert!(delta != 0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .cast(burn::tensor::DType::I32)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }
}
