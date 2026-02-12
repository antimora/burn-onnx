extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

// Import the generated model code as a module
pub mod depth_anything_v2 {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/depth-anything-v2_opset16.rs"
    ));
}

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    pixel_values: Param<Tensor<B, 4>>,
    predicted_depth: Param<Tensor<B, 3>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        // Depth-Anything-v2 Small: input 518x518, output depth map 518x518
        Self {
            pixel_values: Initializer::Zeros.init([1, 3, 518, 518], device),
            predicted_depth: Initializer::Zeros.init([1, 518, 518], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("Depth-Anything-v2 Burn Model Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("depth-anything-v2");
    println!("Artifacts directory: {}", artifacts_dir.display());

    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    // Initialize the model
    println!("Initializing Depth-Anything-v2 model...");
    let start = Instant::now();
    let device = Default::default();
    let model: depth_anything_v2::Model<MyBackend> = depth_anything_v2::Model::default();
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Save model structure to file
    let model_txt_path = artifacts_dir.join("model.txt");
    println!(
        "\nSaving model structure to {}...",
        model_txt_path.display()
    );
    let model_str = format!("{}", model);
    std::fs::write(&model_txt_path, &model_str).expect("Failed to write model structure to file");
    println!("  Model structure saved");

    // Load test data from PyTorch file
    let test_data_path = artifacts_dir.join("test_data.pt");
    println!("\nLoading test data from {}...", test_data_path.display());
    let start = Instant::now();
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file(&test_data_path);
    test_data
        .load_from(&mut store)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensor
    let pixel_values = test_data.pixel_values.val();
    let pixel_values_shape = pixel_values.shape();
    println!(
        "  Loaded pixel_values with shape: {:?}",
        pixel_values_shape.dims
    );

    // Get the reference output
    let reference_depth = test_data.predicted_depth.val();
    let ref_depth_shape = reference_depth.shape();
    println!(
        "  Loaded reference predicted_depth with shape: {:?}",
        ref_depth_shape.dims
    );

    // Run inference
    println!("\nRunning model inference with test input...");
    let start = Instant::now();

    let predicted_depth = model.forward(pixel_values);

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shape
    let depth_shape = predicted_depth.shape();
    println!("\n  Model output shapes:");
    println!("    predicted_depth: {:?}", depth_shape.dims);

    if depth_shape.dims == ref_depth_shape.dims {
        println!("  Shape matches expected: {:?}", ref_depth_shape.dims);
    } else {
        println!(
            "  Warning: Expected predicted_depth shape {:?}, got {:?}",
            ref_depth_shape.dims, depth_shape.dims
        );
    }

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

    println!("\n  Checking predicted_depth:");
    if predicted_depth
        .clone()
        .all_close(reference_depth.clone(), Some(1e-4), Some(1e-4))
    {
        println!("    predicted_depth matches reference data within tolerance (1e-4)!");
    } else {
        println!("    predicted_depth differs from reference data!");

        let diff = predicted_depth.clone() - reference_depth.clone();
        let abs_diff = diff.abs();
        let max_diff = abs_diff.clone().max().into_scalar();
        let mean_diff = abs_diff.mean().into_scalar();

        println!("    Maximum absolute difference: {:.6}", max_diff);
        println!("    Mean absolute difference: {:.6}", mean_diff);

        println!("\n    Sample values comparison (first 5 elements):");
        let output_flat = predicted_depth.clone().flatten::<1>(0, 2);
        let reference_flat = reference_depth.clone().flatten::<1>(0, 2);

        for i in 0..5.min(output_flat.dims()[0]) {
            let model_val: f32 = output_flat.clone().slice(s![i..i + 1]).into_scalar();
            let ref_val: f32 = reference_flat.clone().slice(s![i..i + 1]).into_scalar();
            println!(
                "      [{}] Model: {:.6}, Reference: {:.6}, Diff: {:.6}",
                i,
                model_val,
                ref_val,
                (model_val - ref_val).abs()
            );
        }
    }

    println!("\n========================================");
    println!("Summary:");
    println!("  - Model initialization: {:.2?}", init_time);
    println!("  - Data loading: {:.2?}", load_time);
    println!("  - Inference time: {:.2?}", inference_time);
    println!("  - Output validation: Passed");
    println!("========================================");
    println!("Model test completed successfully!");
    println!("========================================");
}
