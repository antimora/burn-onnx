use super::prelude::*;
use onnx_ir::node::stft::StftConfig;

impl NodeCodegen for onnx_ir::node::stft::StftNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let signal_arg = &self.inputs[0];
        let output_arg = self.outputs.first().unwrap();

        let signal_tensor = match &signal_arg.ty {
            ArgType::Tensor(t) => t,
            other => unreachable!("STFT signal type validated in onnx-ir, got {other:?}"),
        };
        let signal_rank = signal_tensor.rank;
        let squeezed_rank = signal_rank - 1;
        let squeeze_dim = squeezed_rank as isize;

        let signal = scope.arg(signal_arg);
        let output = arg_to_ident(output_arg);

        let StftConfig {
            onesided,
            frame_step,
            frame_length,
            has_window,
        } = self.config;

        // window, when present, is always input[2] (input[1] is frame_step, baked to Static)
        let window_expr = if has_window {
            let window_arg = &self.inputs[2];
            let window = scope.arg(window_arg);
            quote! { Some(#window) }
        } else {
            quote! { None }
        };

        quote! {
            let #output = {
                let signal = #signal.squeeze_dims::<#squeezed_rank>(&[#squeeze_dim]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: #frame_length,
                    hop_length: #frame_step,
                    win_length: None,
                    center: false,
                    onesided: #onesided,
                };
                stft(signal, #window_expr, options)
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::signal::stft");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::stft::{StftConfig, StftNodeBuilder};

    #[test]
    fn test_stft_onesided_no_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: 4,
            frame_length: 16,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 16usize,
                    hop_length: 4usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                };
                stft(signal, None, options)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_full_no_window() {
        let config = StftConfig {
            onesided: false,
            frame_step: 8,
            frame_length: 8,
            has_window: false,
        };
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, signal: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 8usize,
                    hop_length: 8usize,
                    win_length: None,
                    center: false,
                    onesided: false,
                };
                stft(signal, None, options)
            };
            output
        }
        ");
    }

    #[test]
    fn test_stft_with_window() {
        let config = StftConfig {
            onesided: true,
            frame_step: 16,
            frame_length: 32,
            has_window: true,
        };
        // This test deliberately keeps frame_step as a dynamic input so it appears
        // in the forward() signature. In real models lift_constants folds it to
        // Static and it drops out; the builder helpers cannot produce Static args
        // directly, so we use a placeholder to exercise the window indexing logic.
        let node = StftNodeBuilder::new("stft1")
            .input_tensor("signal", 3, DType::F32)
            .input_tensor("_frame_step_placeholder", 0, DType::I64)
            .input_tensor("window", 1, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            signal: Tensor<B, 3>,
            _frame_step_placeholder: Tensor<B, 0, Int>,
            window: Tensor<B, 1>,
        ) -> Tensor<B, 4> {
            let output = {
                let signal = signal.squeeze_dims::<2usize>(&[2isize]);
                let options = burn::tensor::signal::StftOptions {
                    n_fft: 32usize,
                    hop_length: 16usize,
                    win_length: None,
                    center: false,
                    onesided: true,
                };
                stft(signal, Some(window), options)
            };
            output
        }
        ");
    }
}
