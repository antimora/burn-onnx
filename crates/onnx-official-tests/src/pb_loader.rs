//! Load `TensorProto` `.pb` files shipped with upstream ONNX node tests.
//!
//! ONNX backend tests serialize each input/output tensor as a single
//! `TensorProto` message. This module decodes those files into plain
//! `(shape, Vec<f32>)` tuples for FLOAT tensors, which is all the M1
//! scaffold needs. Other dtypes will be added as we widen coverage.

use onnx_ir::protos::TensorProto;
use protobuf::Message;
use std::path::Path;

/// ONNX `TensorProto.DataType` enum value for FLOAT (32-bit IEEE 754).
const DATA_TYPE_FLOAT: i32 = 1;

/// A decoded FLOAT tensor: row-major dimensions and a flat value buffer.
#[derive(Debug, Clone)]
pub struct FloatTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl FloatTensor {
    /// Total number of scalar elements implied by `shape`.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Errors that can occur while loading a `.pb` reference tensor.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Proto(protobuf::Error),
    UnsupportedDataType(i32),
    LengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Proto(e) => write!(f, "protobuf decode error: {e}"),
            Self::UnsupportedDataType(t) => {
                write!(
                    f,
                    "unsupported TensorProto data_type {t} (M1 only handles FLOAT=1)"
                )
            }
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor element count mismatch: shape implies {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<protobuf::Error> for LoadError {
    fn from(e: protobuf::Error) -> Self {
        Self::Proto(e)
    }
}

/// Decode a single `TensorProto` `.pb` file as a FLOAT tensor.
///
/// ONNX backend tests typically pack values into the `raw_data` field as
/// little-endian f32 bytes, but the spec also allows `float_data`. We
/// support both and prefer `raw_data` when present.
pub fn load_float_tensor(path: &Path) -> Result<FloatTensor, LoadError> {
    let bytes = std::fs::read(path)?;
    let proto = TensorProto::parse_from_bytes(&bytes)?;

    if proto.data_type != DATA_TYPE_FLOAT {
        return Err(LoadError::UnsupportedDataType(proto.data_type));
    }

    let shape: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
    let expected = shape.iter().product::<usize>();

    let values: Vec<f32> = if !proto.raw_data.is_empty() {
        proto
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        proto.float_data.clone()
    };

    if values.len() != expected {
        return Err(LoadError::LengthMismatch {
            expected,
            actual: values.len(),
        });
    }

    Ok(FloatTensor { shape, values })
}
