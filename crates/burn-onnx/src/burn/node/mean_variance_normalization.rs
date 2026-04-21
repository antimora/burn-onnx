use super::prelude::*;
use onnx_ir::node::mean_variance_normalization::MeanVarianceNormalizationNode;

impl NodeCodegen for MeanVarianceNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Chain `.mean_dim(ax)` calls for each axis. Burn's `mean_dim` keeps the
        // reduced dimension as size 1, so the result broadcasts cleanly against
        // the original tensor regardless of how many axes we reduce over.
        let mean_chain = self.config.axes.iter().map(|&ax| {
            let ax = ax.to_tokens();
            quote! { .mean_dim(#ax) }
        });
        let mean_chain: TokenStream = mean_chain.collect();

        quote! {
            let #output = {
                let x = #input;
                let mean = x.clone() #mean_chain;
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32) #mean_chain;
                centered / variance.sqrt()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::mean_variance_normalization::{
        MeanVarianceNormalizationConfig, MeanVarianceNormalizationNodeBuilder,
    };

    #[test]
    fn default_axes_per_channel() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![0, 2, 3]))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let x = input;
                let mean = x.clone().mean_dim(0).mean_dim(2).mean_dim(3);
                let centered = x - mean;
                let variance = centered
                    .clone()
                    .powf_scalar(2f32)
                    .mean_dim(0)
                    .mean_dim(2)
                    .mean_dim(3);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn single_axis() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![1]))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let x = input;
                let mean = x.clone().mean_dim(1);
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32).mean_dim(1);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn forward_with_clone() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![0, 2, 3]))
            .build();
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let x = input.clone();
                let mean = x.clone().mean_dim(0).mean_dim(2).mean_dim(3);
                let centered = x - mean;
                let variance = centered
                    .clone()
                    .powf_scalar(2f32)
                    .mean_dim(0)
                    .mean_dim(2)
                    .mean_dim(3);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }
}
