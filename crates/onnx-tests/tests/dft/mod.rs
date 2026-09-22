use crate::include_models;
include_models!(dft_onesided, dft_full, dft_length);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_onesided() {
        let device = Default::default();
        let model: dft_onesided::Model = dft_onesided::Model::new(&device);

        // Input: [1, 8, 1] real signal [1, 2, 3, 4, 5, 6, 7, 8]
        let input = burn::tensor::Tensor::<3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]],
            &device,
        );

        let output = model.forward(input);

        // Expected onesided DFT output: [1, 5, 2]
        let expected = burn::tensor::Tensor::<3>::from_floats(
            [[
                [36.0f32, 0.0],
                [-4.0, 9.656_855],
                [-4.0, 4.0],
                [-4.0, 1.656_854_3],
                [-4.0, 0.0],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn dft_full_spectrum() {
        let device = Default::default();
        let model: dft_full::Model = dft_full::Model::new(&device);

        // Input: [1, 8, 1] real signal [1, 2, 3, 4, 5, 6, 7, 8]
        let input = burn::tensor::Tensor::<3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]],
            &device,
        );

        let output = model.forward(input);

        // Expected full DFT output: [1, 8, 2]
        // Full spectrum = onesided + conjugate mirror
        let expected = burn::tensor::Tensor::<3>::from_floats(
            [[
                [36.0f32, 0.0],
                [-4.0, 9.656_855],
                [-4.0, 4.0],
                [-4.0, 1.656_854_3],
                [-4.0, 0.0],
                [-4.0, -1.656_854_3],
                [-4.0, -4.0],
                [-4.0, -9.656_855],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn dft_length_pads_and_truncates() {
        let device = Default::default();
        let model: dft_length::Model = dft_length::Model::new(&device);

        let input = burn::tensor::Tensor::<3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0]]],
            &device,
        );

        let (padded, truncated) = model.forward(input);

        let expected_padded = burn::tensor::Tensor::<3>::from_floats(
            [[
                [15.0f32, 0.0],
                [-5.414_213_7, -7.242_640_5],
                [3.0, 2.0],
                [-2.585_786_3, -1.242_640_7],
                [3.0, 0.0],
            ]],
            &device,
        );
        let expected_truncated = burn::tensor::Tensor::<3>::from_floats(
            [[[10.0f32, 0.0], [-2.0, 2.0], [-2.0, 0.0], [-2.0, -2.0]]],
            &device,
        );

        padded.to_data().assert_approx_eq::<f32>(
            &expected_padded.to_data(),
            burn::tensor::Tolerance::default(),
        );
        truncated.to_data().assert_approx_eq::<f32>(
            &expected_truncated.to_data(),
            burn::tensor::Tolerance::default(),
        );
    }
}
