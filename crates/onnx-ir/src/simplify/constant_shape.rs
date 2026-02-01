use std::collections::HashMap;

use crate::TensorDataExt;
use crate::ir::{Argument, NodeType, RawNode};

/// Replace `Shape -> Gather(constant_index)` chains with constant values when the
/// Shape node's input tensor has a statically known shape.
///
/// The orphaned Shape nodes are cleaned up by dead node elimination.
pub(crate) fn simplify_constant_shape(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    // Build output_name -> node index map
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.name.clone(), i);
        }
    }

    // Collect replacements: (gather_node_index, constant_dim_value)
    let mut replacements: Vec<(usize, i64)> = Vec::new();

    for (gi, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Gather || node.inputs.len() < 2 {
            continue;
        }
        if let Some(dim_value) = extract_constant_shape_dim(node, &nodes, &producer) {
            replacements.push((gi, dim_value));
        }
    }

    // Apply replacements: turn Gather into Identity with a constant input
    for (gi, dim_value) in &replacements {
        let gather = &nodes[*gi];
        log::info!(
            "Simplification: replacing Shape->Gather '{}' with constant {}",
            gather.name,
            dim_value,
        );

        let output_name = &gather.outputs[0].name;
        nodes[*gi] = RawNode {
            node_type: NodeType::Identity,
            name: nodes[*gi].name.clone(),
            inputs: vec![Argument::from_const_i64(output_name.clone(), *dim_value)],
            outputs: nodes[*gi].outputs.clone(),
            attrs: HashMap::new(),
        };
    }

    nodes
}

/// Check if a Gather node reads from a Shape node with a statically known input,
/// and if so, return the constant dimension value.
fn extract_constant_shape_dim(
    gather: &RawNode,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
) -> Option<i64> {
    // Gather's axis must be 0 (indexing into the 1D shape output)
    let axis = gather
        .attrs
        .get("axis")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if axis != 0 {
        return None;
    }

    // Gather's data input (input[0]) must come from a Shape node
    let shape_idx = *producer.get(&gather.inputs[0].name)?;
    let shape_node = &nodes[shape_idx];
    if shape_node.node_type != NodeType::Shape {
        return None;
    }

    // The Shape node's input must have a known static shape
    let static_shape = shape_node.inputs[0].ty.static_shape()?;

    // Account for Shape's start/end attributes (opset 15+)
    let rank = static_shape.len();
    let mut start = shape_node
        .attrs
        .get("start")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);
    if start < 0 {
        start += rank as i64;
    }
    let start = start as usize;

    // Gather's index (input[1]) must be a constant scalar
    let index_val = gather.inputs[1].value()?.scalar_i64().ok()?;

    // Compute the actual dimension index into the original shape
    let dim_idx = start + index_val as usize;
    if dim_idx >= static_shape.len() {
        return None;
    }

    Some(static_shape[dim_idx] as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, AttributeValue, DType, TensorType, ValueSource};
    use crate::simplify::tests::arg;
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn tensor_arg_with_shape(name: &str, shape: Vec<usize>) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: shape.len(),
                static_shape: Some(shape),
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn shape_arg(name: &str, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Shape(rank),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn const_scalar_arg(name: &str, value: i64) -> Argument {
        let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
        let data_ref = TensorDataRef::new(bytes, vec![1], DType::I64);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        let value_store = ValueStore::new(std::rc::Rc::new(store), std::rc::Rc::new(constant_map));
        Argument {
            name: name.to_string(),
            ty: ArgType::Scalar(DType::I64),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    fn raw_node(
        name: &str,
        node_type: NodeType,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        attrs: HashMap<String, AttributeValue>,
    ) -> RawNode {
        RawNode {
            node_type,
            name: name.to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    #[test]
    fn test_shape_gather_replaced_with_constant() {
        // tensor(shape=[2,3,4]) -> Shape -> Gather(idx=1) -> should become const 3
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4])],
                vec![shape_arg("shape_out", 3)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 1)],
                vec![arg("dim_val")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let result = simplify_constant_shape(nodes);
        let gather = result.iter().find(|n| n.name == "gather").unwrap();
        assert_eq!(gather.node_type, NodeType::Identity);

        // The input should be a constant with value 3
        let val = gather.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 3);
    }

    #[test]
    fn test_shape_with_start_attr() {
        // tensor(shape=[2,3,4,5]) -> Shape(start=1) -> Gather(idx=1) -> should be 4
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![tensor_arg_with_shape("input", vec![2, 3, 4, 5])],
                vec![shape_arg("shape_out", 3)],
                [("start".to_string(), AttributeValue::Int64(1))]
                    .into_iter()
                    .collect(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 3), const_scalar_arg("idx", 1)],
                vec![arg("dim_val")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let result = simplify_constant_shape(nodes);
        let gather = result.iter().find(|n| n.name == "gather").unwrap();
        assert_eq!(gather.node_type, NodeType::Identity);
        let val = gather.inputs[0].value().unwrap().scalar_i64().unwrap();
        assert_eq!(val, 4); // static_shape[1 + 1] = static_shape[2] = 4
    }

    #[test]
    fn test_gather_from_non_shape_not_replaced() {
        // Gather from a Relu output, not a Shape output
        let nodes = vec![
            raw_node(
                "relu",
                NodeType::Relu,
                vec![arg("input")],
                vec![arg("relu_out")],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![arg("relu_out"), const_scalar_arg("idx", 0)],
                vec![arg("out")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let result = simplify_constant_shape(nodes);
        assert_eq!(result[1].node_type, NodeType::Gather);
    }

    #[test]
    fn test_shape_without_static_shape_not_replaced() {
        // Shape input has no static_shape (dynamic)
        let nodes = vec![
            raw_node(
                "shape",
                NodeType::Shape,
                vec![arg("input")], // arg() creates tensor with static_shape: None
                vec![shape_arg("shape_out", 2)],
                HashMap::new(),
            ),
            raw_node(
                "gather",
                NodeType::Gather,
                vec![shape_arg("shape_out", 2), const_scalar_arg("idx", 0)],
                vec![arg("dim_val")],
                [("axis".to_string(), AttributeValue::Int64(0))]
                    .into_iter()
                    .collect(),
            ),
        ];

        let result = simplify_constant_shape(nodes);
        assert_eq!(result[1].node_type, NodeType::Gather);
    }
}
