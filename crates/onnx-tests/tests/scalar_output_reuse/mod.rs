// Import the shared macro
use crate::include_models;
include_models!(scalar_output_reuse);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    /// A scalar graph output that also feeds a later node must stay readable for
    /// the boundary conversion to a native scalar (issue #567).
    #[test]
    fn scalar_output_reuse() {
        let device = Default::default();
        let model = scalar_output_reuse::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/scalar_output_reuse.bpk"),
            &device,
        );

        let input =
            Tensor::<1>::from_floats([0.49671414, -0.1382643, 0.64768857, 1.5230298], &device);

        let (max, neg) = model.forward(input);

        assert_eq!(max, 1.5230298);
        assert_eq!(neg, -1.5230298);
    }
}
