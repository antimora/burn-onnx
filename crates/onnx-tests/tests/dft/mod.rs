use crate::include_models;
include_models!(dft_onesided, dft_full, dft_length, dft_complex);

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

        let input =
            burn::tensor::Tensor::<3>::from_floats([[[1.0], [2.0], [3.0], [4.0], [5.0]]], &device);

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

    #[test]
    fn dft_complex_and_inverse() {
        let device = Default::default();
        let model: dft_complex::Model = dft_complex::Model::new(&device);

        let x = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(0..16, &device)
            .float()
            .mul_scalar(0.5)
            .sub_scalar(2.0)
            .reshape([1, 8, 2]);
        let r = burn::tensor::Tensor::<3>::from_floats(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]],
            &device,
        );

        let (spectrum, roundtrip, real_inverse) = model.forward(x.clone(), r);

        let tolerance = burn::tensor::Tolerance::absolute(1e-4);
        spectrum.to_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([[
                [12.0f32, 16.0],
                [-13.656_86, 5.656_85],
                [-8.0, 0.0],
                [-5.656_85, -2.343_15],
                [-4.0, -4.0],
                [-2.343_15, -5.656_85],
                [0.0, -8.0],
                [5.656_85, -13.656_86],
            ]]),
            tolerance,
        );
        roundtrip
            .to_data()
            .assert_approx_eq::<f32>(&x.to_data(), tolerance);
        real_inverse.to_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([[
                [4.5f32, 0.0],
                [-0.5, -1.207_11],
                [-0.5, -0.5],
                [-0.5, -0.207_11],
                [-0.5, 0.0],
                [-0.5, 0.207_11],
                [-0.5, 0.5],
                [-0.5, 1.207_11],
            ]]),
            tolerance,
        );
    }
}
