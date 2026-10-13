use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Artifact paths
// ---------------------------------------------------------------------------

/// Returns the artifacts directory for a model-check crate.
///
/// Used in `src/main.rs` (no cargo warnings).
pub fn artifacts_dir(model_name: &str) -> PathBuf {
    let base = match std::env::var("BURN_CACHE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => dirs::cache_dir()
            .expect("could not determine cache directory")
            .join("burn-onnx"),
    };
    base.join("model-checks").join(model_name)
}

/// Returns the artifacts directory for a model-check crate, printing a
/// `cargo:warning` so the path is visible during builds.
///
/// Used in `build.rs`.
pub fn artifacts_dir_build(model_name: &str) -> PathBuf {
    let dir = artifacts_dir(model_name);
    println!(
        "cargo:warning=model-checks: artifacts dir = {}",
        dir.display()
    );
    dir
}

// ---------------------------------------------------------------------------
// Backend and device selection
// ---------------------------------------------------------------------------
//
// The new burn API removes the `B: Backend` generic from `Tensor`, `Module`,
// and friends. Backend selection is now a runtime property of the `Device`
// value, driven by Cargo features that bring in the corresponding burn
// sub-crate (`burn-flex`, `burn-wgpu`, ...). There is no per-backend type
// alias to define anymore.
//
// `backend_type!()` is kept as a no-op so existing model-check crates can call
// it without breakage during the rev bump migration; new crates do not need
// to invoke it.

/// No-op kept for source compatibility with model-check crates that still
/// invoke `model_checks_common::backend_type!()`.
#[macro_export]
macro_rules! backend_type {
    () => {};
}

/// Returns the best available device for the active backend.
///
/// Every supported backend (wgpu / metal / flex) picks its best device by default.
#[macro_export]
macro_rules! best_device {
    () => {{ <burn::prelude::Device as core::default::Default>::default() }};
}
