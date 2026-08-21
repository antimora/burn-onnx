//! # Reduce Operations (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceSumSquare)
//!
//! Reduction operations that compute aggregates along specified axes of a tensor. These operations
//! reduce the input tensor by applying an aggregation function (sum, mean, max, min, product, or
//! sum of squares) along the specified axes.
//!
//! **ONNX Specs**:
//! - ReduceSum: <https://onnx.ai/onnx/operators/onnx__ReduceSum.html>
//! - ReduceMean: <https://onnx.ai/onnx/operators/onnx__ReduceMean.html>
//! - ReduceMax: <https://onnx.ai/onnx/operators/onnx__ReduceMax.html>
//! - ReduceMin: <https://onnx.ai/onnx/operators/onnx__ReduceMin.html>
//! - ReduceProd: <https://onnx.ai/onnx/operators/onnx__ReduceProd.html>
//! - ReduceSumSquare: <https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html>
//!
//! ## Opset Versions
//! - **Opset 1-10**: Earlier versions with different attribute handling
//! - **Opset 11-12**: Standardized behavior with axes attribute, added noop_with_empty_axes
//! - **Opset 13-17**: Extended type support (bfloat16, uint/int types)
//! - **Opset 18+**: Axes moved from attribute to optional input tensor for dynamic shapes
//!

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, NodeType, RawNode, RuntimeInputRef, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Axes a reduction runs over.
#[derive(Debug, Clone, PartialEq)]
pub enum ReduceAxes {
    /// Axes known at build time, normalized to non-negative indices. An empty list is
    /// ONNX's "empty axes", which reduces every dimension unless `noop_with_empty_axes`
    /// is set.
    Static(Vec<usize>),
    /// Axes supplied by a runtime input. Values are unknown at build time and are *not*
    /// normalized: negative entries reach the backend as written, and Burn's dimension
    /// APIs wrap them.
    Runtime(RuntimeInputRef),
}

impl Default for ReduceAxes {
    fn default() -> Self {
        ReduceAxes::Static(Vec::new())
    }
}

#[derive(Debug, Clone, new)]
pub struct ReduceConfig {
    pub axes: ReduceAxes,
    pub keepdims: bool,
    /// ONNX `noop_with_empty_axes`: when set, an empty `axes` makes the op an identity
    /// instead of a full reduction.
    pub noop_with_empty_axes: bool,
}

/// Node representation for ReduceMax operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceMaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceMin operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceMinNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceMean operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceMeanNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceSum operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceSumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceProd operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceProdNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceSumSquare operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceSumSquareNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceL1 operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceL1Node {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceL2 operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceL2Node {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceLogSum operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceLogSumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Node representation for ReduceLogSumExp operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReduceLogSumExpNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReduceConfig,
}

/// Whether codegen can read this argument's value at run time.
///
/// Reading the axes back on the host needs a rank-1 integer tensor or a Shape; anything
/// else has no meaningful element list.
fn is_readable_axes(ty: &ArgType) -> bool {
    matches!(ty, ArgType::Tensor(tensor) if tensor.rank == 1) || matches!(ty, ArgType::Shape(_))
}

/// Length of a runtime `axes` input, read from its static shape.
///
/// The values are unknown at build time but the count usually is not, and the count is
/// what fixes the output rank when `keepdims` is off.
fn runtime_axes_len(node: &RawNode) -> Option<usize> {
    let ty = &node.get_input(1)?.ty;
    is_readable_axes(ty).then(|| ty.first_dim_static_len())?
}

/// Normalize an ONNX axis against the input rank, rejecting anything outside
/// the spec's `[-r, r-1]`.
fn normalize_axis(axis: i64, rank: usize) -> Result<usize, ProcessError> {
    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized < 0 || normalized >= rank as i64 {
        return Err(ProcessError::InvalidAttribute {
            name: "axes".to_string(),
            reason: format!("axis {axis} is out of range for a rank-{rank} input"),
        });
    }
    Ok(normalized as usize)
}

pub(crate) struct ReduceProcessor;

impl NodeProcessor for ReduceProcessor {
    type Config = ReduceConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(1, 2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for maximum input count
        // Opset 18+ allows optional axes input (2 inputs total). Opset 11-17 only allows 1 input.
        // Should validate: for opset < 18, max 1 input; for opset >= 18, max 2 inputs.
        // Location: After validate_min_inputs

        // TODO: Validate output count
        // Spec requires exactly 1 output. Should add: validate_output_count(node, 1)
        // Location: After input count validation

        // TODO: Missing test coverage for ReduceSumSquare
        // Tests cover ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, but not ReduceSumSquare
        // which is mentioned in spec. Verify if ReduceSumSquare is implemented.
        // Add test: reduce_sum_square (if supported)

        // TODO: Missing test coverage for duplicate axes
        // Spec doesn't explicitly forbid duplicate axes (e.g., axes=[1,1]). Behavior unclear.
        // Add test: reduce_duplicate_axes

        // Validate input type and extract tensor info
        let (tensor_rank, tensor_elem_type, tensor_static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => (tensor.rank, tensor.dtype, tensor.static_shape.clone()),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        let config = self.extract_config(node, opset)?;
        let keepdims = config.keepdims;

        // Axes known at build time, if any. `None` means they arrive at run time.
        let dims = match &config.axes {
            ReduceAxes::Static(dims) => Some(dims),
            ReduceAxes::Runtime(axes) => {
                let axes_ty = &node.inputs[axes.input_index].ty;
                if !is_readable_axes(axes_ty) {
                    return Err(ProcessError::TypeMismatch {
                        expected: "rank-1 Tensor or Shape for the runtime 'axes' input".to_string(),
                        actual: format!("{axes_ty:?}"),
                    });
                }
                None
            }
        };

        // How many dimensions the reduction removes. `None` means the axes arrive at
        // runtime and their count is not recoverable from the axes input's shape.
        let axis_count = match dims {
            Some(dims) => Some(dims.len()),
            None => runtime_axes_len(node),
        };

        // Only usable when it describes every dimension of the input.
        let static_shape = tensor_static_shape.filter(|shape| shape.len() == tensor_rank);

        // ONNX: an empty `axes` is an identity when `noop_with_empty_axes` is set,
        // and reduces every dimension otherwise.
        if axis_count == Some(0) {
            if config.noop_with_empty_axes {
                node.outputs[0].ty = node.inputs[0].ty.clone();
                return Ok(());
            }

            node.outputs[0].ty = if keepdims {
                ArgType::Tensor(TensorType {
                    dtype: tensor_elem_type,
                    rank: tensor_rank,
                    static_shape: Some(vec![Some(1); tensor_rank]),
                })
            } else {
                ArgType::ScalarTensor(tensor_elem_type)
            };
            return Ok(());
        }

        if keepdims {
            // Every named axis collapses to 1, so the rank is unchanged whether or not
            // the axes themselves are known.
            let static_shape = dims.zip(static_shape).map(|(dims, mut shape)| {
                for dim in dims {
                    shape[*dim] = Some(1);
                }
                shape
            });

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: tensor_elem_type,
                rank: tensor_rank,
                static_shape,
            });
            return Ok(());
        }

        // keepdims=0: the output rank depends on how many axes are reduced, so an
        // unknown axis count is not recoverable.
        let Some(axis_count) = axis_count else {
            return Err(ProcessError::Custom(format!(
                "'{}' takes 'axes' as a runtime input of unknown length with keepdims=0, \
                 so the output rank cannot be determined at build time",
                node.name
            )));
        };

        if axis_count == tensor_rank {
            node.outputs[0].ty = ArgType::ScalarTensor(tensor_elem_type);
            return Ok(());
        }

        let static_shape = dims.zip(static_shape).map(|(dims, mut shape)| {
            for dim in dims.iter().rev() {
                shape.remove(*dim);
            }
            shape
        });

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor_elem_type,
            rank: tensor_rank - axis_count,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Validate input type and extract tensor info
        let tensor_rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract attributes. `axes` moved to an input in opset 18 (13 for ReduceSum),
        // so a model may carry it either way.
        let mut axes_attr = None;
        let mut keepdims = 1;
        let mut noop_with_empty_axes = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axes" => axes_attr = Some(value.clone().into_i64s()),
                "keepdims" => keepdims = value.clone().into_i64(),
                "noop_with_empty_axes" => noop_with_empty_axes = value.clone().into_i64(),
                _ => {}
            }
        }

        // The axes input wins over the attribute when both are present. A value that is
        // not statically known stays `Runtime` so codegen reads it at run time, except
        // when the input's shape already proves the list is empty.
        let static_axes: Vec<i64> = match node.get_input(1) {
            Some(axes_arg) => match axes_arg.value() {
                Some(value) => value.to_vec::<i64>().map_err(|e| {
                    ProcessError::Custom(format!("Failed to read 'axes' of '{}': {e}", node.name))
                })?,
                None if runtime_axes_len(node) == Some(0) => Vec::new(),
                None => {
                    return Ok(ReduceConfig::new(
                        ReduceAxes::Runtime(RuntimeInputRef::new(axes_arg.name.clone(), 1)),
                        keepdims == 1,
                        noop_with_empty_axes == 1,
                    ));
                }
            },
            None => axes_attr.unwrap_or_default(),
        };

        let mut dims = static_axes
            .into_iter()
            .map(|axis| normalize_axis(axis, tensor_rank))
            .collect::<Result<Vec<usize>, _>>()?;

        // Sort the dimensions to ensure consistent order
        dims.sort();

        Ok(ReduceConfig::new(
            ReduceAxes::Static(dims),
            keepdims == 1,
            noop_with_empty_axes == 1,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        match builder.node_type {
            NodeType::ReduceMax => Node::ReduceMax(ReduceMaxNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceMin => Node::ReduceMin(ReduceMinNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceMean => Node::ReduceMean(ReduceMeanNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceSum => Node::ReduceSum(ReduceSumNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceProd => Node::ReduceProd(ReduceProdNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceSumSquare => Node::ReduceSumSquare(ReduceSumSquareNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceL1 => Node::ReduceL1(ReduceL1Node {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceL2 => Node::ReduceL2(ReduceL2Node {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceLogSum => Node::ReduceLogSum(ReduceLogSumNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            NodeType::ReduceLogSumExp => Node::ReduceLogSumExp(ReduceLogSumExpNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
                config,
            }),
            _ => panic!("ReduceProcessor called with unsupported node type"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::node::test_utils::TestNodeBuilder;
    use NodeType;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::ReduceMax, "test_reduce_max")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("reduced", 3, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }
        if let Some(kd) = keepdims {
            builder = builder.attr_int("keepdims", kd);
        }

        builder.build()
    }

    #[test]
    fn test_reduce_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(vec![1]));
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(vec![1])); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(Vec::new()));
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(vec![0, 1]));
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(vec![1]));
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_update_outputs_scalar_no_axes_no_keepdims() {
        // Test that reduce with no axes and keepdims=false produces a scalar output
        let mut node = create_test_node(None, Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::ScalarTensor(_) => {
                // This is the expected case - scalar tensor output (stays on device)
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_scalar_all_dims_no_keepdims() {
        // Test that reduce with all dimensions and keepdims=false produces a scalar output
        let mut node = create_test_node(Some(vec![0, 1, 2]), Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::ScalarTensor(_) => {
                // This is the expected case - scalar tensor output (stays on device)
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_partial_dims_no_keepdims() {
        // Test that reduce with partial dimensions and keepdims=false produces a tensor output
        let mut node = create_test_node(Some(vec![1]), Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2)
                assert_eq!(tensor.rank, 2);
            }
            ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => {
                panic!("Expected tensor output but got scalar");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_with_keepdims() {
        // Test that reduce with keepdims=true always produces a tensor output
        let mut node = create_test_node(None, Some(1));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain original rank when keepdims=true
                assert_eq!(tensor.rank, 3);
            }
            ArgType::ScalarTensor(_) | ArgType::ScalarNative(_) => {
                panic!("Expected tensor output but got scalar when keepdims=true");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_keepdims() {
        // Regression test for partial static_shape with keepdims=true
        // This was causing "index out of bounds" panic before the fix
        let mut node = TestNodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_no_keepdims() {
        // Regression test for partial static_shape without keepdims
        let mut node = TestNodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![1]) // Reduce on dimension 1
            .attr_int("keepdims", 0)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2) without keepdims
                assert_eq!(tensor.rank, 2);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_complete_static_shape_keepdims() {
        // Test that complete static_shape is properly updated with keepdims=true
        let mut node = TestNodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![2, 4, 768])) // Complete shape
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be updated: [2, 4, 768] -> [2, 4, 1]
                assert_eq!(tensor.static_shape, Some(vec![Some(2), Some(4), Some(1)]));
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }
    /// Build a Reduce node whose `axes` arrives as a runtime input (opset 18 shape),
    /// with `axes_len` naming the statically known length of that input.
    fn create_runtime_axes_node(
        axes_len: Option<usize>,
        keepdims: i64,
        noop_with_empty_axes: Option<i64>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::ReduceSum, "test_reduce_sum")
            .input_tensor_f32("data", 3, Some(vec![2, 3, 4]))
            .input_tensor_i64("axes", 1, axes_len.map(|len| vec![len]))
            .output_tensor_f32("reduced", 3, None)
            .attr_int("keepdims", keepdims);

        if let Some(noop) = noop_with_empty_axes {
            builder = builder.attr_int("noop_with_empty_axes", noop);
        }

        builder.build()
    }

    #[test]
    fn test_reduce_runtime_axes_leaves_dims_unknown() {
        let node = create_runtime_axes_node(Some(1), 1, None);

        let config = ReduceProcessor.extract_config(&node, 18).unwrap();

        // The whole point of #459: this must not collapse to "no axes given".
        assert!(
            matches!(&config.axes, ReduceAxes::Runtime(axes) if axes.name == "axes"),
            "expected runtime axes, got {:?}",
            config.axes
        );
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_runtime_axes_keepdims_preserves_rank() {
        let mut node = create_runtime_axes_node(Some(1), 1, None);
        let prefs = OutputPreferences::new();

        ReduceProcessor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 3);
                // Which dimension collapses is only known at run time.
                assert_eq!(tensor.static_shape, None);
            }
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_reduce_runtime_axes_no_keepdims_uses_axes_length() {
        let mut node = create_runtime_axes_node(Some(2), 0, None);
        let prefs = OutputPreferences::new();

        ReduceProcessor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => assert_eq!(tensor.rank, 1), // 3 - 2
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_reduce_runtime_axes_no_keepdims_unknown_length_is_rejected() {
        let mut node = create_runtime_axes_node(None, 0, None);
        let prefs = OutputPreferences::new();

        let err = ReduceProcessor
            .infer_types(&mut node, 18, &prefs)
            .expect_err("output rank is not knowable without the axes length");

        assert!(
            err.to_string().contains("output rank"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_reduce_runtime_axes_empty_input_is_static() {
        // A runtime axes input whose shape proves it is empty is the ONNX "empty axes"
        // case, not an unknown one.
        let node = create_runtime_axes_node(Some(0), 1, None);

        let config = ReduceProcessor.extract_config(&node, 18).unwrap();

        assert_eq!(config.axes, ReduceAxes::Static(Vec::new()));
    }

    #[test]
    fn test_reduce_noop_with_empty_axes_is_identity() {
        let mut node = create_runtime_axes_node(Some(0), 1, Some(1));
        let prefs = OutputPreferences::new();

        let config = ReduceProcessor.extract_config(&node, 18).unwrap();
        assert_eq!(config.noop_with_empty_axes, true);

        ReduceProcessor.infer_types(&mut node, 18, &prefs).unwrap();

        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_reduce_empty_axes_without_noop_reduces_everything() {
        let mut node = create_runtime_axes_node(Some(0), 1, Some(0));
        let prefs = OutputPreferences::new();

        ReduceProcessor.infer_types(&mut node, 18, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 3);
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(1), Some(1), Some(1)]),
                    "reducing every dimension with keepdims collapses each to 1"
                );
            }
            other => panic!("Expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_reduce_axis_out_of_range_is_rejected() {
        let node = create_test_node(Some(vec![3]), Some(1));

        let err = ReduceProcessor
            .extract_config(&node, 16)
            .expect_err("axis 3 is out of range for a rank-3 input");

        assert!(err.to_string().contains("out of range"), "got: {err}");
    }
}
