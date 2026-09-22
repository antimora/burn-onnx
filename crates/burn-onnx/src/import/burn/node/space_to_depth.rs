use super::prelude::*;

impl NodeCodegen for onnx_ir::space_to_depth::SpaceToDepthNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let block_size = self.config.block_size;

        // burn's PixelUnshuffle orders output channels as (c, block_h, block_w);
        // ONNX wants (block_h, block_w, c), so the channel groups are swapped after.
        quote! {
            let #output = {
                let unshuffled = burn::nn::PixelUnshuffleConfig::new(#block_size)
                    .init()
                    .forward(#input);
                let [b, c, h, w] = unshuffled.dims();
                unshuffled
                    .reshape([b, c / (#block_size * #block_size), #block_size * #block_size, h, w])
                    .swap_dims(1, 2)
                    .reshape([b, c, h, w])
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::space_to_depth::{SpaceToDepthConfig, SpaceToDepthNodeBuilder};

    #[test]
    fn test_space_to_depth() {
        let config = SpaceToDepthConfig::new(2);
        let node = SpaceToDepthNodeBuilder::new("s2d1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let unshuffled = burn::nn::PixelUnshuffleConfig::new(2usize)
                    .init()
                    .forward(input);
                let [b, c, h, w] = unshuffled.dims();
                unshuffled
                    .reshape([b, c / (2usize * 2usize), 2usize * 2usize, h, w])
                    .swap_dims(1, 2)
                    .reshape([b, c, h, w])
            };
            output
        }
        ");
    }
}
