use crate::include_models;
include_models!(thresholded_relu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn thresholded_relu() {
        let device = Default::default();
        let model: thresholded_relu::Model<TestBackend> = thresholded_relu::Model::new(&device);

        // Run the model (alpha=2.0, input scaled by 3.0)
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.4901425, -0.4147929, 1.9430656],
                [4.5690894, -0.7024601, -0.7024109],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[0.0f32, 0.0, 0.0], [4.569_089_4, 0.0, 0.0]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
