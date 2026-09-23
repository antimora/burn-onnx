// Import the shared macro
use crate::include_models;
include_models!(
    attention_4d,
    attention_3d,
    attention_attn_mask_bool,
    attention_attn_mask_int,
    attention_attn_mask_float,
    attention_softcap,
    attention_cache,
    attention_custom_scale,
    attention_is_causal,
    attention_qk_output_0,
    attention_qk_output_1,
    attention_qk_output_2,
    attention_qk_output_3,
    attention_gqa_causal
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn simple_4d() {
        let device = Default::default();
        let model: attention_4d::Model = attention_4d::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn simple_3d() {
        let device = Default::default();
        let model: attention_3d::Model = attention_3d::Model::new(&device);

        let q = Tensor::<3>::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let k = Tensor::<3>::from_floats([[[0.0, 1.0], [1.0, 0.0]]], &device);
        let v = Tensor::<3>::from_floats([[[0.25, 0.5], [0.3, 0.6]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[0.283488f32, 0.566976], [0.266511, 0.533023]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_bool() {
        let device = Default::default();
        let model: attention_attn_mask_bool::Model = attention_attn_mask_bool::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask =
            Tensor::<2, Bool>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_int() {
        let device = Default::default();
        let model: attention_attn_mask_int::Model = attention_attn_mask_int::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<2, Int>::from_ints([[2, 0], [0, 3]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_float() {
        let device = Default::default();
        let model: attention_attn_mask_float::Model =
            attention_attn_mask_float::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<2>::from_floats([[2.0, 0.0], [0.0, 3.0]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn softcap() {
        let device = Default::default();
        let model: attention_softcap::Model = attention_softcap::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283176f32, 0.566352], [0.266823, 0.533647]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[allow(clippy::type_complexity)]
    fn cached_attn_inputs() -> (
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<2, Bool>,
        Tensor<4>,
        Tensor<4>,
    ) {
        let device = &Default::default();
        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], device);
        let k = Tensor::<4>::from_floats([[[[1.0, 0.0]]]], device);
        let v = Tensor::<4>::from_floats([[[[0.3, 0.6]]]], device);
        let attn_mask =
            Tensor::<2, Bool>::from_bool(TensorData::from([[true, true], [true, true]]), device);
        let past_k = Tensor::<4>::from_floats([[[[0.0, 1.0]]]], device);
        let past_v = Tensor::<4>::from_floats([[[[0.25, 0.5]]]], device);

        (q, k, v, attn_mask, past_k, past_v)
    }

    #[test]
    fn cache() {
        let device = Default::default();
        let model: attention_cache::Model = attention_cache::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (output, present_k, present_v) = model.forward(q, k, v, attn_mask, past_k, past_v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);
        let expected_k = TensorData::from([[[[0.0, 1.0], [1.0, 0.0]]]]);
        let expected_v = TensorData::from([[[[0.25, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
        present_k
            .to_data()
            .assert_approx_eq::<f32>(&expected_k, Tolerance::default());
        present_v
            .to_data()
            .assert_approx_eq::<f32>(&expected_v, Tolerance::default());
    }

    #[test]
    fn custom_scale() {
        let device = Default::default();
        let model: attention_custom_scale::Model = attention_custom_scale::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.294039f32, 0.588079], [0.255960, 0.511920]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn is_causal() {
        let device = Default::default();
        let model: attention_is_causal::Model = attention_is_causal::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_0() {
        let device = Default::default();
        let model: attention_qk_output_0::Model = attention_qk_output_0::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.707106], [0.707106, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_1() {
        let device = Default::default();
        let model: attention_qk_output_1::Model = attention_qk_output_1::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.707106], [0.707106, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_2() {
        let device = Default::default();
        let model: attention_qk_output_2::Model = attention_qk_output_2::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.67904], [0.67904, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_3() {
        let device = Default::default();
        let model: attention_qk_output_3::Model = attention_qk_output_3::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.336474f32, 0.663525], [0.663525, 0.336474]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn attention_gqa_causal() {
        let device = Default::default();
        let model = attention_gqa_causal::Model::new(&device);
        let seq = |shape: [usize; 4], scale: f32| {
            let n = shape.iter().product::<usize>() as i64;
            Tensor::<1, burn::tensor::Int>::arange(0..n, &device)
                .float()
                .reshape(shape)
                .mul_scalar(scale)
                .remainder_scalar(1.7)
                .sub_scalar(0.8)
        };
        let mask = Tensor::<2>::from_floats([[0.0, -0.5, 0.3], [0.2, 0.0, -1.0]], &device);

        let y = model.forward(
            seq([1, 4, 2, 4], 0.37),
            seq([1, 2, 3, 4], 0.23),
            seq([1, 2, 3, 4], 0.41),
            mask,
        );

        y.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [-0.8f32, -0.39, 0.02, 0.43],
                    [-0.091_481, -0.415_921, -0.005_921, 0.404_078],
                ],
                [
                    [0.72, -0.57, -0.16, 0.25],
                    [0.696_09, -0.593_911, -0.183_911, 0.226_09],
                ],
                [
                    [-0.8, -0.39, 0.02, 0.43],
                    [-0.113_607, -0.415_112, -0.005_112, 0.404_888],
                ],
                [
                    [0.72, -0.57, -0.16, 0.25],
                    [0.685_064, -0.604_937, -0.194_936, 0.215_064],
                ],
            ]]),
            burn::tensor::Tolerance::absolute(1e-4),
        );
    }
}
