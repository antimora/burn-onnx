use crate::include_models;
include_models!(split_to_sequence);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn split_to_sequence() {
        // Expected values from split_to_sequence.py (onnx ReferenceEvaluator, except e0).
        let device = Default::default();
        let model = split_to_sequence::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/split_to_sequence.bpk"),
            &device,
        );
        let x = Tensor::<2>::from_data(
            TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), [4, 6]),
            &device,
        );

        let (a0, a1, b0, b1, c0, d0, d1, e0) = model.forward(x);

        a0.to_data()
            .assert_eq(&TensorData::from([0f32, 1., 2., 3., 4., 5.]), true);
        a1.to_data()
            .assert_eq(&TensorData::from([18f32, 19., 20., 21., 22., 23.]), true);
        b0.to_data().assert_eq(
            &TensorData::from([[1f32, 2.], [7., 8.], [13., 14.], [19., 20.]]),
            true,
        );
        b1.to_data().assert_eq(
            &TensorData::from([
                [3f32, 4., 5.],
                [9., 10., 11.],
                [15., 16., 17.],
                [21., 22., 23.],
            ]),
            true,
        );
        c0.to_data().assert_eq(
            &TensorData::from([[4f32, 5.], [10., 11.], [16., 17.], [22., 23.]]),
            true,
        );
        d0.to_data()
            .assert_eq(&TensorData::from([[2f32], [8.], [14.], [20.]]), true);
        d1.to_data()
            .assert_eq(&TensorData::from([[5f32], [11.], [17.], [23.]]), true);
        // onnxruntime (and torch's `unbind`) drop the axis here; the spec would keep it.
        e0.to_data()
            .assert_eq(&TensorData::from([6f32, 7., 8., 9., 10., 11.]), true);
    }
}
