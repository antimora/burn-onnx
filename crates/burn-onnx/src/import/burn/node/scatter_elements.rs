use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_elements::ScatterElementsReduction;

impl NodeCodegen for onnx_ir::scatter_elements::ScatterElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let dim = self.config.axis.to_tokens();
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data_arg = self.inputs.first().unwrap();
        let data_kind = match &data_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input for data"),
        };

        if matches!(data_kind, TensorKind::Bool) {
            if !matches!(self.config.reduction, ScatterElementsReduction::None) {
                panic!(
                    "ScatterElements with {:?} reduction not supported for bool tensors",
                    self.config.reduction
                );
            }

            // Bool scatter only maps to a logical `or`, which cannot clear a bit, so the
            // assignment round-trips through i64 and uses the same delta identity as below.
            return quote! {
                let #output = {
                    let bool_data = #data.int().cast(burn::tensor::DType::I64);
                    let bool_updates = #updates.int().cast(burn::tensor::DType::I64);
                    let gathered = bool_data.clone().gather(#dim, #indices.clone());
                    bool_data
                        .scatter(#dim, #indices, bool_updates - gathered, burn::tensor::IndexingUpdateOp::Add)
                        .bool()
                };
            };
        }

        // Assign, Min and Max are not implemented by the flex and cubecl backends
        // (tracel-ai/burn#5522), so they gather the current values and scatter-add a delta:
        //   none: data[p] + (updates[p] - data[p])       = updates[p]
        //   max:  data[p] + max(0, updates[p] - data[p]) = max(data[p], updates[p])
        //   min:  data[p] + min(0, updates[p] - data[p]) = min(data[p], updates[p])
        // Elsewhere the tensor is unchanged. This assumes unique indices; ONNX applies the
        // reduction repeatedly when indices are duplicated.
        let delta = match &self.config.reduction {
            ScatterElementsReduction::Add => {
                return quote! {
                    let #output = #data.scatter(#dim, #indices, #updates, burn::tensor::IndexingUpdateOp::Add);
                };
            }
            ScatterElementsReduction::Mul => {
                return quote! {
                    let #output = #data.scatter(#dim, #indices, #updates, burn::tensor::IndexingUpdateOp::Mul);
                };
            }
            ScatterElementsReduction::None => quote! { #updates - gathered },
            ScatterElementsReduction::Max => quote! { (#updates - gathered).clamp_min(0) },
            ScatterElementsReduction::Min => quote! { (#updates - gathered).clamp_max(0) },
        };

        quote! {
            let #output = {
                let gathered = #data.clone().gather(#dim, #indices.clone());
                #data.scatter(#dim, #indices, #delta, burn::tensor::IndexingUpdateOp::Add)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::scatter_elements::{
        ScatterElementsConfig, ScatterElementsNodeBuilder, ScatterElementsReduction,
    };

    #[test]
    fn test_scatter_elements_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(0, indices, updates - gathered, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_add() {
        let config = ScatterElementsConfig::new(1, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = data.scatter(1, indices, updates, burn::tensor::IndexingUpdateOp::Add);
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_mul() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Mul);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = data.scatter(0, indices, updates, burn::tensor::IndexingUpdateOp::Mul);
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_max() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Max);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(
                    0,
                    indices,
                    (updates - gathered).clamp_min(0),
                    burn::tensor::IndexingUpdateOp::Add,
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_min() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Min);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(
                    0,
                    indices,
                    (updates - gathered).clamp_max(0),
                    burn::tensor::IndexingUpdateOp::Add,
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_int() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Int>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = {
                let gathered = data.clone().gather(0, indices.clone());
                data.scatter(0, indices, updates - gathered, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let bool_data = data.int().cast(burn::tensor::DType::I64);
                let bool_updates = updates.int().cast(burn::tensor::DType::I64);
                let gathered = bool_data.clone().gather(0, indices.clone());
                bool_data
                    .scatter(
                        0,
                        indices,
                        bool_updates - gathered,
                        burn::tensor::IndexingUpdateOp::Add,
                    )
                    .bool()
            };
            output
        }
        ");
    }

    #[test]
    #[should_panic(expected = "reduction not supported for bool tensors")]
    fn test_scatter_elements_bool_add_panics() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        codegen_forward_default(&node);
    }
}
