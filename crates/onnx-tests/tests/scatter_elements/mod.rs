use crate::include_models;
include_models!(
    scatter_elements,
    scatter_elements_axis1,
    scatter_elements_add,
    scatter_elements_mul,
    scatter_elements_max,
    scatter_elements_min,
    scatter_elements_bool
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

    #[test]
    fn scatter_elements_default() {
        let device = Default::default();
        let model: scatter_elements::Model = scatter_elements::Model::new(&device);

        let data = Tensor::<2>::zeros([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[2.0f32, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_axis1() {
        let device = Default::default();
        let model: scatter_elements_axis1::Model = scatter_elements_axis1::Model::new(&device);

        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let indices = Tensor::<2, Int>::from_ints([[2, 0], [1, 2], [0, 1]], &device);
        let updates = Tensor::<2>::from_floats([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], &device);

        let output = model.forward(data, indices, updates);

        let expected =
            TensorData::from([[20.0f32, 2.0, 10.0], [4.0, 30.0, 40.0], [50.0, 60.0, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    fn make_test_inputs(device: &Device) -> (Tensor<2>, Tensor<2, Int>, Tensor<2>) {
        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], device);
        (data, indices, updates)
    }

    // Updates that beat the existing value at some targets and fall below it at others, so
    // both branches of the max/min reduction are exercised.
    fn make_mixed_test_inputs(device: &Device) -> (Tensor<2>, Tensor<2, Int>, Tensor<2>) {
        let data =
            Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], device);
        let updates = Tensor::<2>::from_floats([[9.5, 1.1, 2.5], [0.5, 8.5, 7.5]], device);
        (data, indices, updates)
    }

    #[test]
    fn scatter_elements_with_add_reduction() {
        let device = Default::default();
        let model: scatter_elements_add::Model = scatter_elements_add::Model::new(&device);

        let data = Tensor::<2>::ones([3, 3], &device);
        let indices = Tensor::<2, Int>::from_ints([[1, 0, 2], [0, 2, 1]], &device);
        let updates = Tensor::<2>::from_floats([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], &device);

        let output = model.forward(data, indices, updates);

        let expected = TensorData::from([[3.0f32, 2.1, 1.0], [2.0, 1.0, 3.2], [1.0, 3.1, 2.2]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_mul_reduction() {
        let device = Default::default();
        let model: scatter_elements_mul::Model = scatter_elements_mul::Model::new(&device);

        let (data, indices, updates) = make_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]*=1.0=4, data[0,1]*=1.1=2.2, data[2,2]*=1.2=10.8
        // data[0,0]*=2.0=2, data[2,1]*=2.1=16.8, data[1,2]*=2.2=13.2
        let expected = TensorData::from([[2.0f32, 2.2, 3.0], [4.0, 5.0, 13.2], [7.0, 16.8, 10.8]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_max_reduction() {
        let device = Default::default();
        let model: scatter_elements_max::Model = scatter_elements_max::Model::new(&device);

        let (data, indices, updates) = make_mixed_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]=max(4,9.5)=9.5, data[0,1]=max(2,1.1)=2, data[2,2]=max(9,2.5)=9
        // data[0,0]=max(1,0.5)=1, data[2,1]=max(8,8.5)=8.5, data[1,2]=max(6,7.5)=7.5
        let expected = TensorData::from([[1.0f32, 2.0, 3.0], [9.5, 5.0, 7.5], [7.0, 8.5, 9.0]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_with_min_reduction() {
        let device = Default::default();
        let model: scatter_elements_min::Model = scatter_elements_min::Model::new(&device);

        let (data, indices, updates) = make_mixed_test_inputs(&device);
        let output = model.forward(data, indices, updates);

        // data[1,0]=min(4,9.5)=4, data[0,1]=min(2,1.1)=1.1, data[2,2]=min(9,2.5)=2.5
        // data[0,0]=min(1,0.5)=0.5, data[2,1]=min(8,8.5)=8, data[1,2]=min(6,7.5)=6
        let expected = TensorData::from([[0.5f32, 1.1, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 2.5]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn scatter_elements_bool_none() {
        let device = Default::default();
        let model: scatter_elements_bool::Model = scatter_elements_bool::Model::new(&device);

        let data = Tensor::<2, Bool>::from_bool(
            TensorData::from([[true, true, true], [false, false, false]]),
            &device,
        );
        let indices = Tensor::<2, Int>::from_ints([[2, 0, 1], [1, 2, 0]], &device);
        let updates = Tensor::<2, Bool>::from_bool(
            TensorData::from([[false, false, true], [true, true, false]]),
            &device,
        );

        let output = model.forward(data, indices, updates);

        // Targets are both set and cleared, which a logical-or scatter could not express.
        let expected = TensorData::from([[false, true, false], [false, true, true]]);
        assert_eq!(output.to_data(), expected);
    }
}
