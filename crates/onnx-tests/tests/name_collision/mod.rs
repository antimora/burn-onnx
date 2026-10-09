// Include the models for this node type
use crate::include_models;
include_models!(name_collision);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn colliding_names_keep_their_values() {
        let device = Default::default();
        let model: name_collision::Model = name_collision::Model::default();

        let input = Tensor::<2>::from_data(
            TensorData::from([[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            &device,
        );
        let (y, z, w) = model.forward(input);

        // "/c/INT64/[-1]" and "/c/INT64/[1]" are different axes
        assert_eq!(y.dims(), [2, 3, 1]);
        assert_eq!(z.dims(), [2, 1, 3]);

        // "t:0" (Neg) and "t/0" (Abs) are different values: w = -x - |x|
        let expected = TensorData::from([[-2.0f32, 0.0, -6.0], [0.0, -10.0, 0.0]]);
        w.to_data().assert_eq(&expected, true);
    }
}
