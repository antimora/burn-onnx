use crate::include_models;
include_models!(conv_transpose_auto_pad);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    fn seq<const D: usize>(shape: [usize; D], scale: f32) -> Tensor<D> {
        let device = Default::default();
        let n = shape.iter().product::<usize>() as i64;
        Tensor::<1, Int>::arange(0..n, &device)
            .float()
            .reshape(shape)
            .mul_scalar(scale)
            .sub_scalar(1.0)
    }

    #[test]
    fn conv_transpose_auto_pad() {
        let device = Default::default();
        let model = conv_transpose_auto_pad::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv_transpose_auto_pad.bpk"),
            &device,
        );

        let (y1, y2, y3, y4, y5, y6, y7) = model.forward(
            seq([1, 1, 3], 0.5),
            seq([1, 1, 2, 3], 0.4),
            seq([1, 1, 2, 2, 2], 0.3),
            seq([1, 1, 2, 2], 0.6),
            seq([1, 1, 2, 2], 0.7),
            seq([1, 1, 3], 0.5),
            seq([1, 1, 2, 2], 0.5),
        );

        // Ground truth from conv_transpose_auto_pad.py (ONNX Runtime, checked against the
        // reference evaluator where it runs).
        let tolerance = Tolerance::absolute(1e-4);
        y1.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [1.5f32, 1.2, 1.4, 0.85, 0.7, 0.5],
                [-0.4, -0.7, -0.95, -0.6, -0.75, -0.5],
            ]]),
            tolerance,
        );
        y2.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [0.04f32, -0.48, -0.36, -1.28, -0.76, -0.6],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.44, -0.96, -0.36, -0.8, -0.28, -0.12],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]]]),
            tolerance,
        );
        y3.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [
                    [1.25f32, 1.15, 1.05, 0.95, 0.88],
                    [1.35, 1.21, 1.07, 0.84, 0.76],
                    [0.53, 0.49, 0.45, 0.32, 0.31],
                ],
                [
                    [0.65, 0.55, 0.45, 0.53, 0.46],
                    [0.51, 0.37, 0.23, 0.36, 0.28],
                    [0.29, 0.25, 0.21, 0.26, 0.25],
                ],
                [
                    [0.05, 0.07, 0.09, -0.25, -0.2],
                    [-0.69, -0.59, -0.49, -1.2, -1.04],
                    [-0.31, -0.23, -0.15, -0.52, -0.41],
                ],
                [
                    [0.17, 0.19, 0.21, 0.05, 0.1],
                    [-0.09, 0.01, 0.11, -0.24, -0.08],
                    [0.17, 0.25, 0.33, 0.14, 0.25],
                ],
            ]]]),
            tolerance,
        );
        // The last row lies past the full result and holds only the bias.
        y4.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [3.0f32, 2.5, 2.4, 2.2],
                [2.0, 1.5, 2.0, 1.8],
                [1.8, 1.9, 1.2, 1.6],
                [2.0, 2.1, 2.0, 2.4],
                [2.0, 2.0, 2.0, 2.0],
            ]]]),
            tolerance,
        );
        // The second layer crops to twice the first layer's output, not its input.
        assert_eq!(y5.dims(), [1, 1, 8, 8]);
        let y5_sum = y5.sum().into_scalar::<f32>();
        assert!((y5_sum - -7.341).abs() < 1e-3, "y5 sum {y5_sum}");
        // Odd total pad with no auto_pad: the extra unit is trimmed from the start.
        y6.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[0.5f32, 0.5, 0.25, 0.0, 0.0, 0.0]]]),
            tolerance,
        );
        y7.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [1.0f32, 0.75, 1.0, 0.375, 0.25],
                [0.25, 0.0, -0.125, 0.0, -0.125],
                [-0.5, -0.75, -1.75, -0.75, -0.75],
                [0.0, 0.0, -0.125, 0.0, 0.125],
                [0.0, 0.0, 0.25, 0.375, 0.5],
            ]]]),
            tolerance,
        );
    }
}
