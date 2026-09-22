use crate::include_models;
include_models!(conv_runtime_weight);

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
    fn conv_runtime_weight() {
        let device = Default::default();
        let model = conv_runtime_weight::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv_runtime_weight.bpk"),
            &device,
        );

        let (y1, y2, yt) = model.forward(
            seq([1, 2, 5], 0.1),
            seq([3, 2, 3], 0.2),
            seq([1, 2, 4, 4], 0.1),
            seq([3, 2, 3, 3], 0.05),
            seq([1, 2, 2, 2], 0.5),
            seq([2, 3, 2, 2], 0.1),
        );

        let tolerance = Tolerance::absolute(1e-4);
        y1.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [2.48f32, 2.18, 1.88],
                [-2.2, -1.78, -1.36],
                [-6.88, -5.74, -4.6],
            ]]),
            tolerance,
        );
        y2.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [[1.505f32, 1.415, 0.38, -0.085], [-2.18, -4.14, -4.92, -3.6]],
                [[2.705, 4.775, 5.36, 3.275], [2.44, 4.08, 4.38, 2.46]],
                [[8.405, 12.635, 14.84, 11.135], [11.56, 16.8, 18.18, 13.02]],
            ]]),
            tolerance,
        );
        yt.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [1.2f32, 1.2, 0.8, 0.9],
                    [1.2, 1.2, 1.0, 1.1],
                    [0.4, 0.6, 0.0, 0.3],
                    [0.8, 1.0, 0.6, 0.9],
                ],
                [
                    [1.2, 1.2, 1.2, 1.3],
                    [1.2, 1.2, 1.4, 1.5],
                    [1.2, 1.4, 1.2, 1.5],
                    [1.6, 1.8, 1.8, 2.1],
                ],
                [
                    [1.2, 1.2, 1.6, 1.7],
                    [1.2, 1.2, 1.8, 1.9],
                    [2.0, 2.2, 2.4, 2.7],
                    [2.4, 2.6, 3.0, 3.3],
                ],
            ]]),
            tolerance,
        );
    }
}
