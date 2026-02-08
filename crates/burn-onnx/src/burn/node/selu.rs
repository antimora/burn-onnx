use super::prelude::*;

impl NodeCodegen for onnx_ir::selu::SeluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = burn::tensor::activation::selu(#input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::selu::{SeluConfig, SeluNode, SeluNodeBuilder};

    fn create_node(name: &str) -> SeluNode {
        let config = SeluConfig::new(
            1.67326319217681884765625,
            1.05070102214813232421875,
        );

        SeluNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_selu_forward() {
        let node = create_node("selu1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::selu(input);
            output
        }
        ");
    }
}
