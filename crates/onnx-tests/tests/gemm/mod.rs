use crate::include_models;
include_models!(
    gemm,
    gemm_linear_opset6,
    gemm_no_c,
    gemm_non_unit_alpha_beta
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn gemm_test() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm::Model::new(&device);

        // Create input matrices
        let a = Tensor::<2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b = Tensor::<2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);
        let c = 1.0;

        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0 + 1.0, 22.0 + 1.0] = [20.0, 23.0]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0 + 1.0, 50.0 + 1.0] = [44.0, 51.0]
        let expected =
            Tensor::<2>::from_data(TensorData::from([[20.0, 23.0], [44.0, 51.0]]), &device);

        // Run the model
        let output = model.forward(a, b, c);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn gemm_test_non_unit_alpha_beta() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm_non_unit_alpha_beta::Model::new(&device);

        // Create input matrices
        let a = Tensor::<2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b = Tensor::<2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);
        let c = 1.0;

        // Alpha = Beta = 0.5
        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0 * .5 + 1.0 * .5, 22.0 * .5 + 1.0 * .5] = [10.0, 11.5]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0 * .5 + 1.0 * .5, 50.0 * .5 + 1.0 * .5] = [22.0, 25.5]
        let expected =
            Tensor::<2>::from_data(TensorData::from([[10.0, 11.5], [22.0, 25.5]]), &device);

        // Run the model
        let output = model.forward(a, b, c);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn gemm_test_no_c() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm_no_c::Model::new(&device);

        // Create input matrices
        let a = Tensor::<2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b = Tensor::<2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);

        // Alpha = Beta = 0.5
        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0, 22.0]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0, 50.0]
        let expected =
            Tensor::<2>::from_data(TensorData::from([[19.0, 22.0], [43.0, 50.0]]), &device);

        // Run the model
        let output = model.forward(a, b);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn gemm_linear_opset6() {
        // Opset 6 Gemm in the Linear pattern, with the pre-opset-7 `broadcast` attribute
        let device = Default::default();
        let model = gemm_linear_opset6::Model::default();

        let input = Tensor::<2>::from_data(
            TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let output = model.forward(input);

        // Expected values from onnx.reference.ReferenceEvaluator
        let expected = TensorData::from([
            [2.405_213_4f32, -1.560_968_2, -0.019_258_738, -2.343_752_1],
            [5.423_629, 1.603_250_5, 5.612_26, -3.503_514_3],
        ]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
