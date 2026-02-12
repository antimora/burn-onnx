extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::PytorchStore;
use std::time::Instant;

model_checks_common::backend_type!();

// Include the generated model
include!(concat!(env!("OUT_DIR"), "/model/face_detector.rs"));

/// Test data structure matching the PyTorch saved format.
/// BlazeFace outputs two 3D tensors: regressors and classificators.
#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input: Param<Tensor<B, 4>>,
    output_0: Param<Tensor<B, 3>>,
    output_1: Param<Tensor<B, 3>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        // BlazeFace Short Range: 128x128 NHWC input, 896 anchors
        // output_0: [1, 896, 16] regressors (bbox + keypoints)
        // output_1: [1, 896, 1] classificators (face confidence)
        Self {
            input: Initializer::Zeros.init([1, 128, 128, 3], device),
            output_0: Initializer::Zeros.init([1, 896, 16], device),
            output_1: Initializer::Zeros.init([1, 896, 1], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("MediaPipe Face Detector (BlazeFace) Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("mediapipe-face-detector");
    println!("Artifacts directory: {}", artifacts_dir.display());

    // Check if artifacts exist
    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model.");
        eprintln!("Example: uv run get_model.py");
        std::process::exit(1);
    }

    // Check if model files exist
    let model_file = artifacts_dir.join("face_detector.onnx");
    let test_data_file = artifacts_dir.join("face_detector_test_data.pt");

    if !model_file.exists() {
        eprintln!("Error: Model file not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    if !test_data_file.exists() {
        eprintln!("Error: Test data file not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    // Initialize the model with weights
    println!("Initializing model...");
    let start = Instant::now();
    let device = Default::default();
    let weights_path = concat!(env!("OUT_DIR"), "/model/face_detector.bpk");
    let model: Model<MyBackend> = Model::from_file(weights_path, &device);
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Load test data from PyTorch file
    println!("\nLoading test data from {}...", test_data_file.display());
    let start = Instant::now();
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file(&test_data_file);
    test_data
        .load_from(&mut store)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensor from test data
    let input = test_data.input.val();
    let input_shape = input.shape();
    println!("  Input shape: {:?}", input_shape.dims);

    // Get reference outputs
    let ref_0 = test_data.output_0.val();
    let ref_1 = test_data.output_1.val();
    println!("  Reference output_0 shape: {:?}", ref_0.shape().dims);
    println!("  Reference output_1 shape: {:?}", ref_1.shape().dims);

    // Run inference
    println!("\nRunning model inference...");
    let start = Instant::now();
    let (out_0, out_1) = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    println!("\nModel outputs:");
    println!("  output_0 shape: {:?}", out_0.shape().dims);
    println!("  output_1 shape: {:?}", out_1.shape().dims);

    // Compare outputs
    println!("\nComparing outputs with reference data...");

    let mut all_passed = true;

    for (name, output, reference) in [
        ("output_0 (regressors)", out_0, ref_0),
        ("output_1 (classificators)", out_1, ref_1),
    ] {
        print!("\n  Checking {name}:");
        if output
            .clone()
            .all_close(reference.clone(), Some(1e-4), Some(1e-4))
        {
            println!(" PASS (within 1e-4)");
        } else {
            println!(" MISMATCH");
            all_passed = false;

            let diff = output.clone() - reference.clone();
            let abs_diff = diff.abs();
            let max_diff = abs_diff.clone().max().into_scalar();
            let mean_diff = abs_diff.mean().into_scalar();
            println!("    Max abs diff:  {:.6}", max_diff);
            println!("    Mean abs diff: {:.6}", mean_diff);

            println!("\n    Sample values (first 5):");
            let out_flat = output.flatten::<1>(0, 2);
            let ref_flat = reference.flatten::<1>(0, 2);
            for i in 0..5.min(out_flat.dims()[0]) {
                let m: f32 = out_flat.clone().slice(s![i..i + 1]).into_scalar();
                let r: f32 = ref_flat.clone().slice(s![i..i + 1]).into_scalar();
                println!("      [{i}] model={m:.6}, ref={r:.6}, diff={:.6}", (m - r).abs());
            }
        }
    }

    println!("\n========================================");
    println!("Summary:");
    println!("  Model init:  {init_time:.2?}");
    println!("  Data load:   {load_time:.2?}");
    println!("  Inference:   {inference_time:.2?}");
    if all_passed {
        println!("  Validation:  PASS");
        println!("========================================");
        println!("Model test completed successfully!");
    } else {
        println!("  Validation:  FAIL");
        println!("========================================");
        println!("Model test completed with differences.");
        std::process::exit(1);
    }
}
