use super::prelude::*;

impl NodeCodegen for onnx_ir::concat::ConcatNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        // Check if any inputs are scalars
        let has_scalar = self.inputs.iter().any(|arg| arg.ty.is_scalar());
        // Shape inputs are host arrays; a tensor output means they have to be
        // moved on device before `Tensor::cat`.
        let has_shape = self
            .inputs
            .iter()
            .any(|arg| matches!(arg.ty, ArgType::Shape(_)));

        // Determine if this is tensor or shape concatenation based on output type
        match &self.outputs.first().unwrap().ty {
            ArgType::Tensor(_) if has_scalar || has_shape => {
                let all_scalars = self.inputs.iter().all(|arg| arg.ty.is_scalar());

                if all_scalars {
                    // All scalars - create a single tensor directly
                    let dtype = self.inputs[0].ty.elem_type();
                    let dtype_tokens = dtype.to_tokens();
                    let kind = match dtype {
                        DType::Bool(_) => quote! { , Bool },
                        _ if dtype.is_float() => quote! {},
                        _ => quote! { , Int },
                    };
                    let scalar_inputs: Vec<_> =
                        self.inputs.iter().map(|arg| scope.arg(arg)).collect();

                    quote! {
                        let #output: Tensor<1 #kind> = Tensor::from_data(
                            burn::tensor::TensorData::from([#(#scalar_inputs),*]),
                            (&self.device, #dtype_tokens)
                        );
                    }
                } else {
                    // Mixed inputs - convert scalars and shapes to rank-1 tensors, then cat
                    let mut inits = Vec::new();
                    let mut input_exprs = Vec::new();

                    for (i, input_arg) in self.inputs.iter().enumerate() {
                        if let ArgType::Shape(rank) = &input_arg.ty {
                            // Shape is a host `[i64; N]` array: move it on device
                            // as an i64 rank-1 tensor so it can join the cat.
                            let shape_name = arg_to_ident(input_arg);
                            let rank_lit = rank.to_tokens();
                            let temp_name =
                                Ident::new(&format!("shape_as_tensor_{}", i), Span::call_site());
                            inits.push(quote! {
                                let #temp_name: Tensor<1, Int> = Tensor::from_data(
                                    burn::tensor::TensorData::new(#shape_name.to_vec(), [#rank_lit]),
                                    (&self.device, burn::tensor::DType::I64)
                                );
                            });
                            input_exprs.push(quote! { #temp_name });
                            continue;
                        }

                        let input = scope.arg(input_arg);

                        if input_arg.ty.is_scalar() {
                            let dtype = input_arg.ty.elem_type();
                            let dtype_tokens = dtype.to_tokens();
                            let kind = match dtype {
                                DType::Bool(_) => quote! { , Bool },
                                _ if dtype.is_float() => quote! {},
                                _ => quote! { , Int },
                            };
                            let temp_name =
                                Ident::new(&format!("scalar_as_tensor_{}", i), Span::call_site());
                            let init = quote! {
                                let #temp_name: Tensor<1 #kind> = Tensor::from_data(
                                    burn::tensor::TensorData::from([#input]),
                                    (&self.device, #dtype_tokens)
                                );
                            };
                            inits.push(init);
                            input_exprs.push(quote! { #temp_name });
                        } else if has_shape && input_arg.ty.elem_type() != DType::I64 {
                            // Shape-derived tensors are i64; align the others so
                            // every element of the cat shares one dtype.
                            input_exprs.push(quote! { #input.cast(burn::tensor::DType::I64) });
                        } else {
                            input_exprs.push(input);
                        }
                    }

                    quote! {
                        let #output = {
                            #(#inits)*
                            burn::tensor::Tensor::cat([#(#input_exprs),*].into(), #dim)
                        };
                    }
                }
            }
            ArgType::Tensor(_) => {
                // Tensor concatenation (no scalars)
                let inputs = self.inputs.iter().map(|arg| scope.arg(arg));

                quote! {
                    let #output = burn::tensor::Tensor::cat([#(#inputs),*].into(), #dim);
                }
            }
            ArgType::Shape(shape) => {
                // Shape concatenation - shapes are 1D so concat is always on axis 0
                if self.config.axis != 0 {
                    panic!(
                        "Shape concatenation only supports dim=0, got dim={}",
                        self.config.axis
                    );
                }
                let output_rank = shape;

                let has_tensor = self
                    .inputs
                    .iter()
                    .any(|arg| matches!(arg.ty, ArgType::Tensor(_)));

                if has_tensor {
                    // A tensor input lives on device, so the fixed-size shape
                    // array has to be assembled with a host readback instead of
                    // by slicing arrays.
                    let mut pushes = Vec::new();
                    for (i, input) in self.inputs.iter().enumerate() {
                        if matches!(input.ty, ArgType::Tensor(_)) {
                            let tensor = scope.arg(input);
                            let data_name =
                                Ident::new(&format!("tensor_data_{}", i), Span::call_site());
                            pushes.push(quote! {
                                let #data_name = #tensor.cast(burn::tensor::DType::I64).to_data();
                                shape_parts.extend(#data_name.iter::<i64>());
                            });
                        } else {
                            let input_name = arg_to_ident(input);
                            if input.ty.is_scalar() {
                                pushes.push(quote! { shape_parts.push(#input_name as i64); });
                            } else {
                                pushes.push(
                                    quote! { shape_parts.extend_from_slice(&#input_name[..]); },
                                );
                            }
                        }
                    }

                    quote! {
                        let #output: [i64; #output_rank] = {
                            let mut shape_parts = alloc::vec::Vec::with_capacity(#output_rank);
                            #(#pushes)*
                            shape_parts.try_into().unwrap()
                        };
                    }
                } else {
                    // Generate code to concatenate shape arrays
                    // Handle scalar inputs by converting them to single-element arrays
                    let mut shape_parts = Vec::new();
                    for input in &self.inputs {
                        let input_name = arg_to_ident(input);
                        if input.ty.is_scalar() {
                            // Scalar: wrap in array and slice
                            shape_parts.push(quote! { &[#input_name][..] });
                        } else {
                            // Shape: already an array, just slice
                            shape_parts.push(quote! { &#input_name[..] });
                        }
                    }

                    quote! {
                        let #output: [i64; #output_rank] = [#(#shape_parts),*].concat().try_into().unwrap();
                    }
                }
            }
            _ => panic!("Concat only supports Tensor or Shape outputs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::concat::{ConcatConfig, ConcatNode, ConcatNodeBuilder};

    fn create_concat_node(name: &str, num_inputs: usize, axis: usize) -> ConcatNode {
        let config = ConcatConfig { axis };
        let mut builder = ConcatNodeBuilder::new(name);

        for i in 0..num_inputs {
            builder = builder.input_tensor(&format!("input{}", i), 2, DType::F32);
        }

        builder
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_concat_two_tensors() {
        let node = create_concat_node("concat1", 2, 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input0: Tensor<2>, input1: Tensor<2>) -> Tensor<2> {
            let output = burn::tensor::Tensor::cat([input0, input1].into(), 0);
            output
        }
        ");
    }

    #[test]
    fn test_concat_three_tensors() {
        let node = create_concat_node("concat1", 3, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input0: Tensor<2>,
            input1: Tensor<2>,
            input2: Tensor<2>,
        ) -> Tensor<2> {
            let output = burn::tensor::Tensor::cat([input0, input1, input2].into(), 1);
            output
        }
        ");
    }

    #[test]
    fn test_concat_scalar_inputs() {
        let config = ConcatConfig { axis: 0 };
        let node = ConcatNodeBuilder::new("concat_scalars")
            .input_scalar("s0", DType::I64)
            .input_scalar("s1", DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, s0: i64, s1: i64) -> Tensor<1, Int> {
            let output: Tensor<1, Int> = Tensor::from_data(
                burn::tensor::TensorData::from([s0, s1]),
                (&self.device, burn::tensor::DType::I64),
            );
            output
        }
        ");
    }

    #[test]
    fn test_concat_mixed_scalar_and_tensor() {
        let config = ConcatConfig { axis: 0 };
        let node = ConcatNodeBuilder::new("concat_mixed")
            .input_scalar("s0", DType::F32)
            .input_tensor("t0", 1, DType::F32)
            .config(config)
            .output_tensor("output", 1, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, s0: f32, t0: Tensor<1>) -> Tensor<1> {
            let output = {
                let scalar_as_tensor_0: Tensor<1> = Tensor::from_data(
                    burn::tensor::TensorData::from([s0]),
                    (&self.device, burn::tensor::DType::F32),
                );
                burn::tensor::Tensor::cat([scalar_as_tensor_0, t0].into(), 0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_concat_shape_and_tensor_to_tensor() {
        // A Shape input mixed with a runtime-length tensor: the shape array has
        // to be moved on device so both sides can be `cat`ed (issue #438).
        let config = ConcatConfig { axis: 0 };
        let node = ConcatNodeBuilder::new("concat_shape_tensor")
            .input_shape("head", 1)
            .input_tensor("tail", 1, DType::I64)
            .config(config)
            .output_tensor("output", 1, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, head: [i64; 1], tail: Tensor<1, Int>) -> Tensor<1, Int> {
            let output = {
                let shape_as_tensor_0: Tensor<1, Int> = Tensor::from_data(
                    burn::tensor::TensorData::new(head.to_vec(), [1]),
                    (&self.device, burn::tensor::DType::I64),
                );
                burn::tensor::Tensor::cat([shape_as_tensor_0, tail].into(), 0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_concat_shape_output_with_tensor_input() {
        // A rank-1 tensor of known length joining a Shape output: the values
        // have to be read back on host, not sliced like an array (issue #438).
        let config = ConcatConfig { axis: 0 };
        let node = ConcatNodeBuilder::new("concat_shape_out")
            .input_shape("dims", 3)
            .input_tensor("extra", 1, DType::I64)
            .config(config)
            .output_shape("output", 5)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, dims: [i64; 3], extra: Tensor<1, Int>) -> [i64; 5] {
            let output: [i64; 5usize] = {
                let mut shape_parts = alloc::vec::Vec::with_capacity(5usize);
                shape_parts.extend_from_slice(&dims[..]);
                let tensor_data_1 = extra.cast(burn::tensor::DType::I64).to_data();
                shape_parts.extend(tensor_data_1.iter::<i64>());
                shape_parts.try_into().unwrap()
            };
            output
        }
        ");
    }

    #[test]
    fn test_concat_shape_and_i32_tensor_casts_to_i64() {
        // Shape-derived tensors are i64, so a non-i64 int tensor is cast to
        // match before the cat.
        let config = ConcatConfig { axis: 0 };
        let node = ConcatNodeBuilder::new("concat_shape_i32")
            .input_shape("head", 2)
            .input_tensor("tail", 1, DType::I32)
            .config(config)
            .output_tensor("output", 1, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, head: [i64; 2], tail: Tensor<1, Int>) -> Tensor<1, Int> {
            let output = {
                let shape_as_tensor_0: Tensor<1, Int> = Tensor::from_data(
                    burn::tensor::TensorData::new(head.to_vec(), [2]),
                    (&self.device, burn::tensor::DType::I64),
                );
                burn::tensor::Tensor::cat(
                    [shape_as_tensor_0, tail.cast(burn::tensor::DType::I64)].into(),
                    0,
                )
            };
            output
        }
        ");
    }
}
