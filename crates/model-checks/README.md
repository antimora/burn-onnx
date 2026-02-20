# Model Checks

This directory contains model verification and validation tests for burn-onnx. Each subdirectory
represents a different model that we test to ensure burn-onnx can correctly:

1. Import ONNX models
2. Generate Rust code from the models
3. Build and run the generated code

## Purpose

The model-checks serve as integration tests to verify that burn-onnx works correctly with
real-world models. These tests help catch regressions and ensure compatibility with various ONNX
operators and model architectures.

## Structure

Each model directory typically contains:

- Model download/preparation script (e.g., `get_model.py`)
- `build.rs` - Build script that uses burn-onnx to generate Rust code
- `src/main.rs` - Test code that runs the generated model
- `Cargo.toml` - Package configuration

Model artifacts (ONNX files, test data) are stored in the platform cache directory:

- macOS: `~/Library/Caches/burn-onnx/model-checks/<model-name>/`
- Linux: `~/.cache/burn-onnx/model-checks/<model-name>/`

Set `BURN_CACHE_DIR` to override the base cache path (useful for CI).

Generated files (not tracked in git):

- `target/` - Build artifacts and generated model code

## Two-Step Process

### Step 1: Download and Prepare the Model

First, download the model and convert it to the required ONNX format:

```bash
cd model-checks/<model-name>
python get_model.py
# or using uv:
uv run get_model.py
```

The model preparation script typically:

- Downloads the model (if not already present)
- Converts it to ONNX format with the appropriate opset version
- Validates the model structure
- Saves the prepared model to the cache directory

Scripts are designed to skip downloading if the ONNX model already exists, saving time and
bandwidth.

### Step 2: Build and Run the Model

Once the ONNX model is ready, build and run the Rust code:

```bash
cargo build
cargo run
```

The build process will:

- Check that the ONNX model exists (with helpful error messages if not)
- Generate Rust code from the ONNX model using burn-onnx
- Compile the generated code

## Models

| Directory | Model | Source | Related Issue |
|-----------|-------|--------|---------------|
| `albert/` | ALBERT | HuggingFace | |
| `all-minilm-l6-v2/` | all-MiniLM-L6-v2 | HuggingFace | |
| `clip-vit-b-32-text/` | CLIP ViT-B-32 (text) | HuggingFace | |
| `clip-vit-b-32-vision/` | CLIP ViT-B-32 (vision) | HuggingFace | |
| `mediapipe-face-detector/` | MediaPipe Face Detector (BlazeFace) | Google MediaPipe | [#1370](https://github.com/tracel-ai/burn/issues/1370) |
| `modernbert-base/` | ModernBERT-base | HuggingFace | |
| `rf-detr/` | RF-DETR Small | Roboflow | [#4052](https://github.com/tracel-ai/burn/issues/4052) |
| `silero-vad/` | Silero VAD | Silero | |
| `smollm/` | SmolLM / SmolLM2 (135M) | HuggingFace | |
| `yolo/` | YOLO (v5/v8/v10/v11/v12) | Ultralytics | |
