# MediaPipe Face Detector (BlazeFace Short Range)

Model check for the MediaPipe BlazeFace face detection model, originally reported
in [tracel-ai/burn#1370](https://github.com/tracel-ai/burn/issues/1370).

## Model

- **Source**: Google MediaPipe (TFLite, converted to ONNX via tf2onnx)
- **Input**: `[1, 128, 128, 3]` (NHWC image)
- **Outputs**:
  - `regressors`: `[1, 896, 16]` (bounding box + keypoint regressions)
  - `classificators`: `[1, 896, 1]` (face confidence scores)

## Current Status

**Download**: works
**Build**: fails - Pad operator only supports spatial (last 2 dims) padding,
but BlazeFace uses channel-dimension padding for residual connections.

## Usage

```bash
# Download and convert model
uv run get_model.py

# Build (currently fails due to Pad limitation)
cargo build
```
