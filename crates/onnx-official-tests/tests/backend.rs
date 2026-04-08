//! Test backend selection. Mirrors `crates/onnx-tests/tests/backend.rs`
//! so the two suites stay configurable through the same feature flags.

#[cfg(feature = "test-wgpu")]
pub type TestBackend = burn::backend::Wgpu;

#[cfg(all(
    feature = "test-ndarray",
    not(feature = "test-wgpu"),
    not(feature = "test-tch"),
    not(feature = "test-metal")
))]
pub type TestBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "test-metal")]
pub type TestBackend = burn::backend::Metal;

#[cfg(feature = "test-tch")]
pub type TestBackend = burn::backend::LibTorch<f32>;

#[cfg(feature = "test-candle")]
pub type TestBackend = burn::backend::Candle<f32>;
