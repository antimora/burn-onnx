use super::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Min,
    Max,
    Sum,
    Prod,
    Mean,
    L1,
    L2,
    LogSum,
    LogSumExp,
    SumSquare,
}

/// The axes a reduction runs over, lowered for codegen.
enum Axes {
    /// Known at build time. An empty list means every dimension.
    Static(Vec<usize>),
    /// Read from a runtime input into the `__axes` local that `binding` declares.
    Runtime { binding: TokenStream },
}

/// Name of the generated local holding runtime axes.
fn runtime_axes_ident() -> TokenStream {
    quote! { __axes }
}

impl Axes {
    /// Statement that has to precede any use of the axes, empty for static axes.
    ///
    /// Keeping it on the enum means an emission site cannot reference the axes without
    /// also declaring them.
    fn binding(&self) -> TokenStream {
        match self {
            Axes::Static(_) => quote! {},
            Axes::Runtime { binding } => binding.clone(),
        }
    }

    /// Static axis list with "empty means every dimension" resolved, or `None` when the
    /// axes are only known at run time.
    ///
    /// Empty static axes have to resolve to `0..input_rank` rather than stay empty,
    /// which `squeeze_dims` would read as "every size-1 dimension" instead.
    fn resolved_dims(&self, input_rank: usize) -> Option<Vec<usize>> {
        match self {
            Axes::Runtime { .. } => None,
            Axes::Static(dims) if dims.is_empty() => Some((0..input_rank).collect()),
            Axes::Static(dims) => Some(dims.clone()),
        }
    }

    /// The axis list as Burn sees it: the runtime slice, or the resolved literal list.
    fn dims_tokens(&self, input_rank: usize) -> TokenStream {
        match self.resolved_dims(input_rank) {
            Some(dims) => dims.to_tokens(),
            None => runtime_axes_ident(),
        }
    }

    /// Whether these axes reduce the whole tensor to a single element.
    fn reduces_everything(&self) -> bool {
        matches!(self, Axes::Static(dims) if dims.is_empty())
    }
}

/// Drop the reduced axes from a result that still carries them as size-1 dimensions.
///
/// `output_rank == 0` is a scalar result, which the caller reshapes; squeezing every
/// dimension away here would leave a rank Burn cannot express.
fn squeeze_reduced_axes(expr: TokenStream, dims: &TokenStream, output_rank: usize) -> TokenStream {
    if output_rank == 0 {
        expr
    } else {
        quote! { #expr.squeeze_dims::<#output_rank>(&#dims) }
    }
}

impl ReductionType {
    /// Generate the code for a reduction operation along all dimensions.
    fn try_forward_reduce(&self, input: TokenStream) -> Option<TokenStream> {
        match self {
            ReductionType::Min => Some(quote! { #input.min() }),
            ReductionType::Max => Some(quote! { #input.max() }),
            ReductionType::Sum => Some(quote! { #input.sum() }),
            ReductionType::Prod => Some(quote! { #input.prod() }),
            ReductionType::Mean => Some(quote! { #input.mean() }),
            _ => None,
        }
    }

    /// Generate the code for a reduction operation along a specific dimension.
    fn forward_reduce_by_dim(&self, input: TokenStream, dim: usize) -> TokenStream {
        match self {
            ReductionType::Min => quote! { #input.min_dim(#dim) },
            ReductionType::Max => quote! { #input.max_dim(#dim) },
            ReductionType::Sum => quote! { #input.sum_dim(#dim) },
            ReductionType::Prod => quote! { #input.prod_dim(#dim) },
            ReductionType::Mean => quote! { #input.mean_dim(#dim) },
            _ => panic!("Unsupported reduction type {:?}", self),
        }
    }

    /// Generate the code for a reduction over several dimensions at once, keeping the
    /// reduced dimensions as size 1.
    fn forward_reduce_by_dims(&self, input: TokenStream, dims: TokenStream) -> TokenStream {
        match self {
            ReductionType::Min => quote! { #input.min_dims(&#dims) },
            ReductionType::Max => quote! { #input.max_dims(&#dims) },
            ReductionType::Sum => quote! { #input.sum_dims(&#dims) },
            ReductionType::Prod => quote! { #input.prod_dims(&#dims) },
            ReductionType::Mean => quote! { #input.mean_dims(&#dims) },
            _ => panic!("Unsupported reduction type {:?}", self),
        }
    }

    fn forward_reduce(
        &self,
        input: TokenStream,
        axes: &Axes,
        keepdims: bool,
        input_rank: usize,
        output_rank: usize,
    ) -> TokenStream {
        // Whole-tensor reductions have a dedicated kernel for most reduction types.
        if axes.reduces_everything()
            && let Some(reduced_input) = self.try_forward_reduce(input.clone())
        {
            return if keepdims {
                quote! { #reduced_input.expand([1; #output_rank]) }
            } else {
                reduced_input
            };
        }

        // Burn's `*_dims` and `squeeze_dims` take the axes as a slice and wrap negative
        // entries themselves, so runtime axes need nothing here that static axes do not.
        let dims = axes.dims_tokens(input_rank);
        let reduced = match axes.resolved_dims(input_rank) {
            // Static axes fold one dimension at a time, which is what Burn's `*_dims`
            // does internally and reads better in the generated model.
            Some(dims) => dims.iter().fold(input, |tokens, dim| {
                self.forward_reduce_by_dim(tokens, *dim)
            }),
            None => self.forward_reduce_by_dims(input, dims.clone()),
        };

        if keepdims {
            reduced
        } else {
            squeeze_reduced_axes(reduced, &dims, output_rank)
        }
    }
}

// Helper macro to implement NodeCodegen for reduce nodes
macro_rules! impl_reduce_node {
    ($node_type:ty, $reduction_type:expr) => {
        impl NodeCodegen for $node_type {
            fn inputs(&self) -> &[Argument] {
                &self.inputs
            }

            fn outputs(&self) -> &[Argument] {
                &self.outputs
            }

            fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
                let input_arg = self.inputs.first().unwrap();
                let output_arg = self.outputs.first().unwrap();

                // Get input rank and check if it's boolean
                let (input_rank, is_bool) = match &input_arg.ty {
                    onnx_ir::ir::ArgType::Tensor(tensor) => {
                        (tensor.rank, tensor.dtype.is_bool())
                    }
                    _ => panic!("Reduce node input must be a tensor"),
                };

                // Get output rank. Scalar outputs reduce to rank 0 here and are given
                // their final shape by `output_expr` below, so the reduction itself must
                // not try to squeeze down to a rank Burn cannot express.
                let output_rank = match &output_arg.ty {
                    onnx_ir::ir::ArgType::Tensor(tensor) => tensor.rank,
                    onnx_ir::ir::ArgType::ScalarTensor(_)
                    | onnx_ir::ir::ArgType::ScalarNative(_) => 0,
                    _ => panic!("Reduce node output must be tensor or scalar"),
                };

                let input = scope.arg(input_arg);
                let output = arg_to_ident(output_arg);

                let keepdims = self.config.keepdims;

                // ONNX: with `noop_with_empty_axes` set, an empty `axes` is an identity
                // rather than a full reduction.
                let empty_static_axes = matches!(
                    &self.config.axes,
                    onnx_ir::reduce::ReduceAxes::Static(dims) if dims.is_empty()
                );
                if self.config.noop_with_empty_axes && empty_static_axes {
                    return quote! { let #output = #input; };
                }

                // Runtime axes are read into a local once, which every use below reaches
                // through `axes.binding()`.
                let axes = match &self.config.axes {
                    onnx_ir::reduce::ReduceAxes::Static(dims) => Axes::Static(dims.clone()),
                    onnx_ir::reduce::ReduceAxes::Runtime(axes_ref) => {
                        let axes_arg = &self.inputs[axes_ref.input_index];
                        let axes_ident = runtime_axes_ident();
                        let axes_expr = match &axes_arg.ty {
                            onnx_ir::ir::ArgType::Shape(_) => {
                                let axes_input = arg_to_ident(axes_arg);
                                quote! { #axes_input.to_vec() }
                            }
                            // onnx-ir rejects anything but a rank-1 tensor or a Shape.
                            _ => {
                                let axes_input = scope.arg(axes_arg);
                                quote! { #axes_input.into_data().iter::<i64>().collect() }
                            }
                        };
                        Axes::Runtime {
                            binding: quote! {
                                let #axes_ident: alloc::vec::Vec<i64> = #axes_expr;
                            },
                        }
                    }
                };
                let axes_binding = axes.binding();

                // For boolean tensors with Min/Max reduction, use all()/any()
                if is_bool && matches!($reduction_type, ReductionType::Min | ReductionType::Max) {
                    let (bool_reduction_all, bool_reduction_dim) = match $reduction_type {
                        ReductionType::Min => (quote! { all }, quote! { all_dim }),
                        ReductionType::Max => (quote! { any }, quote! { any_dim }),
                        _ => unreachable!(),
                    };

                    // Burn has no `all_dims`/`any_dims`, so runtime axes fold in the
                    // generated code instead of at build time.
                    let squeeze_dims = axes.dims_tokens(input_rank);
                    let reduced_input = if axes.reduces_everything() {
                        quote! { #input.#bool_reduction_all() }
                    } else {
                        match axes.resolved_dims(input_rank) {
                            Some(dims) => dims.iter().fold(input.clone(), |tokens, dim| {
                                quote! { #tokens.#bool_reduction_dim(#dim) }
                            }),
                            None => quote! {
                                #squeeze_dims.iter().fold(#input, |tensor, axis| {
                                    tensor.#bool_reduction_dim(*axis)
                                })
                            },
                        }
                    };

                    let final_output = if keepdims {
                        if axes.reduces_everything() {
                            quote! { #reduced_input.expand([1; #output_rank]) }
                        } else {
                            reduced_input
                        }
                    } else if matches!(&output_arg.ty, onnx_ir::ir::ArgType::ScalarTensor(_)) {
                        // Keep as Tensor<1> on device
                        quote! { #reduced_input.reshape([1]) }
                    } else {
                        squeeze_reduced_axes(reduced_input, &squeeze_dims, output_rank)
                    };

                    return match &output_arg.ty {
                        onnx_ir::ir::ArgType::ScalarNative(_) => {
                            quote! {
                                let #output = {
                                    #axes_binding
                                    #final_output.into_scalar::<bool>()
                                };
                            }
                        }
                        _ => {
                            quote! {
                                let #output = { #axes_binding #final_output };
                            }
                        }
                    };
                }

                let raw_output_expr = match $reduction_type {
                    ReductionType::SumSquare => {
                        let input_square = quote! { #input.square() };
                        ReductionType::Sum.forward_reduce(
                            input_square,
                            &axes,
                            keepdims,
                            input_rank,
                            output_rank,
                        )
                    }
                    ReductionType::L1 => {
                        let input_abs = quote! { #input.abs() };
                        ReductionType::Sum.forward_reduce(
                            input_abs,
                            &axes,
                            keepdims,
                            input_rank,
                            output_rank,
                        )
                    }
                    ReductionType::L2 => {
                        let input_square = quote! { #input.square() };
                        let input_square_reduced = ReductionType::Sum.forward_reduce(
                            input_square,
                            &axes,
                            keepdims,
                            input_rank,
                            output_rank,
                        );

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    dtype @ (onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64) => {
                                        let dtype_tokens = dtype.to_tokens();
                                        // Cast to F32 before sqrt to avoid overflow/underflow
                                        quote! { #input_square_reduced.float().cast(burn::tensor::DType::F32).sqrt().int().cast(#dtype_tokens) }
                                    }
                                    _ => {
                                        // Float types - cast to F32 before sqrt, then back
                                        quote! {
                                            let input_dtype = #input.dtype();
                                            #input_square_reduced.cast(burn::tensor::DType::F32).sqrt().cast(input_dtype)
                                        }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    ReductionType::LogSum => {
                        let input_reduced = ReductionType::Sum.forward_reduce(
                            input.clone(),
                            &axes,
                            keepdims,
                            input_rank,
                            output_rank,
                        );

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    dtype @ (onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64) => {
                                        let dtype_tokens = dtype.to_tokens();
                                        quote! { #input_reduced.float().cast(burn::tensor::DType::F32).log().int().cast(#dtype_tokens) }
                                    }
                                    _ => {
                                        quote! {
                                            let input_dtype = #input.dtype();
                                            #input_reduced.cast(burn::tensor::DType::F32).log().cast(input_dtype)
                                        }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    ReductionType::LogSumExp => {
                        let input_double_from_local = match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64 => {
                                        quote! { input.float().cast(burn::tensor::DType::F32) }
                                    }
                                    _ => {
                                        quote! { input.cast(burn::tensor::DType::F32) }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        };

                        // The running max has to keep its rank so it broadcasts back
                        // over the input, so both intermediate reductions run with
                        // keepdims and the reduced axes are dropped once at the end.
                        let input_max_reduced = ReductionType::Max.forward_reduce(
                            quote! { input_double.clone() },
                            &axes,
                            true,
                            input_rank,
                            input_rank,
                        );

                        let exp_reduced = ReductionType::Sum.forward_reduce(
                            quote! { input_exp_reduced },
                            &axes,
                            true,
                            input_rank,
                            input_rank,
                        );

                        let combined = quote! { (input_max_reduced + exp_sum_reduced.log()) };
                        let combined = if keepdims {
                            combined
                        } else {
                            let dims = axes.dims_tokens(input_rank);
                            squeeze_reduced_axes(combined, &dims, output_rank)
                        };

                        // Both reductions above keep their rank, so `input_max_reduced`
                        // has the input's rank with 1s on the reduced axes and the
                        // subtraction broadcasts on its own.
                        let input_reduced = quote! {
                            let input = #input;
                            let input_dtype = input.dtype();
                            let input_double = #input_double_from_local;
                            let input_max_reduced = #input_max_reduced;
                            let input_exp_reduced = (input_double - input_max_reduced.clone()).exp();
                            let exp_sum_reduced = #exp_reduced;
                            #combined
                        };

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    dtype @ (onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64) => {
                                        let dtype_tokens = dtype.to_tokens();
                                        quote! { #input_reduced.int().cast(#dtype_tokens) }
                                    }
                                    _ => {
                                        quote! { #input_reduced.cast(input_dtype) }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    _ => $reduction_type.forward_reduce(
                        input,
                        &axes,
                        keepdims,
                        input_rank,
                        output_rank,
                    ),
                };

                // Handle scalar outputs by extracting the scalar value from the tensor result
                let output_expr = match &output_arg.ty {
                    onnx_ir::ir::ArgType::ScalarTensor(_) => {
                        // Keep as Tensor<1> on device (no GPU stall)
                        quote! { #raw_output_expr.reshape([1]) }
                    }
                    onnx_ir::ir::ArgType::ScalarNative(dtype) => {
                        on_device_to_native(raw_output_expr, dtype)
                    }
                    onnx_ir::ir::ArgType::Tensor(_) => raw_output_expr,
                    _ => panic!("Reduce node output must be tensor or scalar"),
                };

                quote! { let #output = { #axes_binding #output_expr }; }
            }

            fn register_imports(&self, _imports: &mut BurnImports) {
                // No special imports needed for reduce operations
            }
        }
    };
}

// Implement NodeCodegen for all reduce node types
impl_reduce_node!(onnx_ir::node::reduce::ReduceMaxNode, ReductionType::Max);
impl_reduce_node!(onnx_ir::node::reduce::ReduceMinNode, ReductionType::Min);
impl_reduce_node!(onnx_ir::node::reduce::ReduceSumNode, ReductionType::Sum);
impl_reduce_node!(onnx_ir::node::reduce::ReduceProdNode, ReductionType::Prod);
impl_reduce_node!(onnx_ir::node::reduce::ReduceMeanNode, ReductionType::Mean);
impl_reduce_node!(onnx_ir::node::reduce::ReduceL1Node, ReductionType::L1);
impl_reduce_node!(onnx_ir::node::reduce::ReduceL2Node, ReductionType::L2);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceLogSumNode,
    ReductionType::LogSum
);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceLogSumExpNode,
    ReductionType::LogSumExp
);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceSumSquareNode,
    ReductionType::SumSquare
);

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::node::reduce::{
        ReduceAxes, ReduceConfig, ReduceMaxNode, ReduceMaxNodeBuilder, ReduceMeanNodeBuilder,
        ReduceSumNodeBuilder,
    };

    /// Axes supplied by the node's second input rather than known at build time.
    fn runtime_axes() -> ReduceAxes {
        ReduceAxes::Runtime(RuntimeInputRef::new("axes".to_string(), 1))
    }

    fn create_reduce_max_node(name: &str, config: ReduceConfig) -> ReduceMaxNode {
        ReduceMaxNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_reduce_max_keepdims() {
        let config = ReduceConfig::new(ReduceAxes::Static(vec![1]), true, false);
        let node = create_reduce_max_node("reduce_max1", config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = { input.max_dim(1usize) };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_mean_keepdims() {
        let config = ReduceConfig::new(ReduceAxes::Static(vec![1]), true, false);
        let node = ReduceMeanNodeBuilder::new("reduce_mean1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = { input.mean_dim(1usize) };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_sum_keepdims() {
        let config = ReduceConfig::new(ReduceAxes::Static(vec![1]), true, false);
        let node = ReduceSumNodeBuilder::new("reduce_sum1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = { input.sum_dim(1usize) };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_max_multiple_dims() {
        let config = ReduceConfig::new(ReduceAxes::Static(vec![1, 2]), true, false);
        let node = create_reduce_max_node("reduce_max1", config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = { input.max_dim(1usize).max_dim(2usize) };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_sum_multiple_dims_no_keepdims() {
        let config = ReduceConfig::new(ReduceAxes::Static(vec![1, 2]), false, false);
        let node = ReduceSumNodeBuilder::new("reduce_sum1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<1> {
            let output = {
                input.sum_dim(1usize).sum_dim(2usize).squeeze_dims::<1usize>(&[1, 2])
            };
            output
        }
        ");
    }
    #[test]
    fn test_reduce_sum_runtime_axes_keepdims() {
        // Opset 18 moved `axes` to an input; when it is not a constant the axes are only
        // known at run time (#459).
        let config = ReduceConfig::new(runtime_axes(), true, false);
        let node = ReduceSumNodeBuilder::new("reduce_sum1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("axes", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, axes: Tensor<1, Int>) -> Tensor<3> {
            let output = {
                let __axes: alloc::vec::Vec<i64> = axes.into_data().iter::<i64>().collect();
                input.sum_dims(&__axes)
            };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_mean_runtime_axes_no_keepdims() {
        let config = ReduceConfig::new(runtime_axes(), false, false);
        let node = ReduceMeanNodeBuilder::new("reduce_mean1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("axes", 1, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, axes: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let __axes: alloc::vec::Vec<i64> = axes.into_data().iter::<i64>().collect();
                input.mean_dims(&__axes).squeeze_dims::<2usize>(&__axes)
            };
            output
        }
        ");
    }

    #[test]
    fn test_reduce_sum_noop_with_empty_axes() {
        // `noop_with_empty_axes` turns an empty axes list into an identity.
        let config = ReduceConfig::new(ReduceAxes::Static(vec![]), true, true);
        let node = ReduceSumNodeBuilder::new("reduce_sum1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("axes", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, axes: Tensor<1, Int>) -> Tensor<3> {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_reduce_sum_all_dims_named_no_keepdims() {
        // Naming every axis with keepdims=0 yields a scalar, which must not be reached
        // by squeezing the tensor down to rank 0.
        let config = ReduceConfig::new(ReduceAxes::Static(vec![0, 1, 2]), false, false);
        let node = ReduceSumNodeBuilder::new("reduce_sum1")
            .input_tensor("input", 3, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<1> {
            let output = { input.sum_dim(0usize).sum_dim(1usize).sum_dim(2usize).reshape([1]) };
            output
        }
        ");
    }
}
