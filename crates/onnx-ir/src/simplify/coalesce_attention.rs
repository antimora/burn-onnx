use std::collections::HashMap;

use crate::ir::{AttributeValue, NodeType, RawNode, TensorDataExt};

/// Detect decomposed scaled dot-product attention (SDPA) patterns and replace them
/// with a single Attention node.
///
/// PyTorch's ONNX exporter (especially legacy, opset <=20) decomposes
/// `F.scaled_dot_product_attention(Q, K, V)` into 4-6 primitive ops:
///
/// ```text
/// Q -----> MatMul(Q, K^T) -> [Div/Mul(scale)] -> [Add(mask)] -> Softmax(-1) -> MatMul(scores, V)
/// K -> Transpose --^                                                            V ----------^
/// ```
///
/// This pass recognizes the pattern and replaces it with a single Attention RawNode,
/// enabling Burn's optimized attention primitives. Orphaned nodes are cleaned up by
/// dead node elimination.
///
/// Seeds on Softmax nodes (invariant anchor in every SDPA variant), then traces
/// backward/forward to match the full pattern.
pub(crate) fn coalesce_attention(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    // Build producer map: output_name -> node_index
    let producer = build_producer_map(&nodes);
    // Build consumer map: output_name -> list of node indices that consume it
    let consumer = build_consumer_map(&nodes);

    // Collect replacements: (final_matmul_index, replacement_node)
    let mut replacements: Vec<(usize, RawNode)> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Softmax {
            continue;
        }

        if let Some((final_matmul_idx, attention_node)) =
            try_match_sdpa(i, &nodes, &producer, &consumer)
        {
            replacements.push((final_matmul_idx, attention_node));
        }
    }

    // Apply replacements
    for (idx, replacement) in replacements {
        log::info!(
            "Simplification: coalescing SDPA pattern into Attention node '{}'",
            replacement.name,
        );
        nodes[idx] = replacement;
    }

    nodes
}

/// Try to match a full SDPA pattern starting from a Softmax node.
///
/// Returns `Some((final_matmul_index, attention_node))` if the pattern matches.
fn try_match_sdpa(
    softmax_idx: usize,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<(usize, RawNode)> {
    let softmax = &nodes[softmax_idx];

    // 1. Validate Softmax axis is last dimension
    let softmax_input = softmax.inputs.first()?;
    let rank = softmax_input.ty.rank();
    if rank < 2 {
        return None;
    }
    let axis = get_softmax_axis(softmax, rank)?;
    if axis != rank - 1 {
        return None;
    }

    // 2. Trace backward from Softmax input through optional Add(mask) and Div/Mul(scale)
    let mut pre_softmax_name: &str = &softmax.inputs[0].name;
    let mut mask_input: Option<&str> = None;
    let mut scale_value: Option<f64> = None;

    // Check for optional Add(mask) before Softmax
    if let Some(&add_idx) = producer.get(pre_softmax_name) {
        let add_node = &nodes[add_idx];
        if add_node.node_type == NodeType::Add && is_single_use(&add_node.outputs[0].name, consumer)
        {
            // One input should come from a Div/Mul/MatMul, the other is the mask
            // Try both orderings
            if let Some(result) = try_extract_mask_and_upstream(add_node, nodes, producer, consumer)
            {
                mask_input = Some(result.mask_name);
                pre_softmax_name = result.upstream_output;
                scale_value = result.scale;
            }
        }
    }

    // Check for Div/Mul(scale) if not already found through Add path
    if scale_value.is_none()
        && let Some(&scale_idx) = producer.get(pre_softmax_name)
    {
        let scale_node = &nodes[scale_idx];
        if let Some((upstream_name, sv)) = try_extract_scale(scale_node, nodes, producer, consumer)
        {
            pre_softmax_name = upstream_name;
            scale_value = Some(sv);
        }
    }

    // 3. The pre_softmax_name should now point to QK MatMul output
    let qk_matmul_idx = *producer.get(pre_softmax_name)?;
    let qk_matmul = &nodes[qk_matmul_idx];
    if qk_matmul.node_type != NodeType::MatMul {
        return None;
    }
    if !is_single_use(&qk_matmul.outputs[0].name, consumer) {
        return None;
    }

    // 4. Validate K comes through a Transpose that swaps last two dims
    let k_transpose_idx = *producer.get(&qk_matmul.inputs[1].name)?;
    let k_transpose = &nodes[k_transpose_idx];
    if k_transpose.node_type != NodeType::Transpose {
        return None;
    }
    if !is_last_two_dims_swap(k_transpose)? {
        return None;
    }
    if !is_single_use(&k_transpose.outputs[0].name, consumer) {
        return None;
    }

    // Q and K tensors
    let q_arg = &qk_matmul.inputs[0];
    let k_arg = &k_transpose.inputs[0];

    // Validate Q, K are rank 4
    if q_arg.ty.rank() != 4 || k_arg.ty.rank() != 4 {
        return None;
    }

    // 5. Trace forward: Softmax output feeds into exactly one MatMul (scores * V)
    let softmax_output = &softmax.outputs[0].name;
    let final_matmul_consumers = consumer.get(softmax_output)?;
    if final_matmul_consumers.len() != 1 {
        return None;
    }
    let final_matmul_idx = final_matmul_consumers[0];
    let final_matmul = &nodes[final_matmul_idx];
    if final_matmul.node_type != NodeType::MatMul {
        return None;
    }
    // Softmax output must be input[0] of the final MatMul
    if final_matmul.inputs[0].name != *softmax_output {
        return None;
    }

    // V tensor
    let v_arg = &final_matmul.inputs[1];
    if v_arg.ty.rank() != 4 {
        return None;
    }

    // 6. Build the Attention RawNode
    let attention_node =
        build_attention_node(final_matmul, q_arg, k_arg, v_arg, mask_input, scale_value);

    Some((final_matmul_idx, attention_node))
}

struct MaskAndUpstream<'a> {
    mask_name: &'a str,
    upstream_output: &'a str,
    scale: Option<f64>,
}

/// Given an Add node (potential mask addition), determine which input is the mask
/// and which is the upstream (scale or QK MatMul output).
///
/// Traces through optional Div/Mul scale on the non-mask input.
fn try_extract_mask_and_upstream<'a>(
    add_node: &'a RawNode,
    nodes: &'a [RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<MaskAndUpstream<'a>> {
    // Try both orderings: input[0] is upstream + input[1] is mask, or vice versa
    for (upstream_idx, mask_idx) in [(0, 1), (1, 0)] {
        let upstream_name = &add_node.inputs[upstream_idx].name;
        let mask_name = &add_node.inputs[mask_idx].name;

        // The upstream path should lead to a Div/Mul(scale) or directly to MatMul
        if let Some(&node_idx) = producer.get(upstream_name.as_str()) {
            let upstream_node = &nodes[node_idx];

            // Check if it's a scale node (Div/Mul) with single-use output
            if let Some((matmul_output, sv)) =
                try_extract_scale(upstream_node, nodes, producer, consumer)
            {
                // Verify the scale's upstream is a MatMul
                if let Some(&mm_idx) = producer.get(matmul_output)
                    && nodes[mm_idx].node_type == NodeType::MatMul
                {
                    return Some(MaskAndUpstream {
                        mask_name,
                        upstream_output: matmul_output,
                        scale: Some(sv),
                    });
                }
            }

            // Check if upstream is directly a MatMul (no scale node)
            if upstream_node.node_type == NodeType::MatMul && is_single_use(upstream_name, consumer)
            {
                return Some(MaskAndUpstream {
                    mask_name,
                    upstream_output: upstream_name,
                    scale: None,
                });
            }
        }
    }
    None
}

/// Try to extract scale from a Div or Mul node.
/// Returns (upstream_input_name, scale_value) if successful.
fn try_extract_scale<'a>(
    node: &'a RawNode,
    _nodes: &'a [RawNode],
    _producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<(&'a str, f64)> {
    if !is_single_use(&node.outputs[0].name, consumer) {
        return None;
    }

    match node.node_type {
        NodeType::Div => {
            // Div(x, constant) -> scale = 1/constant
            let divisor = node.inputs[1].value()?.scalar_f64().ok()?;
            if divisor == 0.0 {
                return None;
            }
            Some((&node.inputs[0].name, 1.0 / divisor))
        }
        NodeType::Mul => {
            // Mul(x, constant) or Mul(constant, x) -> scale = constant
            // Try input[1] as constant first (more common)
            if let Some(data) = node.inputs[1].value()
                && let Ok(val) = data.scalar_f64()
            {
                return Some((&node.inputs[0].name, val));
            }
            // Try input[0] as constant
            if let Some(data) = node.inputs[0].value()
                && let Ok(val) = data.scalar_f64()
            {
                return Some((&node.inputs[1].name, val));
            }
            None
        }
        _ => None,
    }
}

/// Build an Attention RawNode that replaces the final MatMul.
fn build_attention_node(
    final_matmul: &RawNode,
    q: &crate::ir::Argument,
    k: &crate::ir::Argument,
    v: &crate::ir::Argument,
    mask: Option<&str>,
    scale: Option<f64>,
) -> RawNode {
    let mut inputs = vec![q.clone(), k.clone(), v.clone()];

    // Attention input[3] is attention_mask (optional)
    if let Some(mask_name) = mask {
        // Find the mask argument from context - we need its type info
        // For now, create a basic argument referencing the mask
        let mut mask_arg = q.clone();
        mask_arg.name = mask_name.to_string();
        inputs.push(mask_arg);
    }

    let mut attrs = HashMap::new();
    if let Some(s) = scale {
        attrs.insert("scale".to_string(), AttributeValue::Float32(s as f32));
    }

    RawNode {
        node_type: NodeType::Attention,
        name: format!("{}_attention", final_matmul.name),
        inputs,
        outputs: final_matmul.outputs.clone(),
        attrs,
    }
}

/// Get the normalized Softmax axis (positive index).
fn get_softmax_axis(softmax: &RawNode, rank: usize) -> Option<usize> {
    let mut axis: i64 = -1; // Default: last axis
    if let Some(attr) = softmax.attrs.get("axis") {
        axis = attr.clone().into_i64();
    }
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis as usize >= rank {
        return None;
    }
    Some(axis as usize)
}

/// Check if a Transpose node swaps the last two dimensions.
fn is_last_two_dims_swap(transpose: &RawNode) -> Option<bool> {
    let rank = transpose.inputs[0].ty.rank();
    if rank < 2 {
        return Some(false);
    }

    let perm: Vec<i64> = if let Some(attr) = transpose.attrs.get("perm") {
        attr.clone().into_i64s()
    } else {
        // Default Transpose reverses all dims - only a swap if rank == 2
        (0..rank as i64).rev().collect()
    };

    if perm.len() != rank {
        return Some(false);
    }

    // Check: all dims except last two are identity, last two are swapped
    for (i, &p) in perm.iter().enumerate() {
        if i < rank - 2 {
            if p != i as i64 {
                return Some(false);
            }
        } else if i == rank - 2 {
            if p != (rank - 1) as i64 {
                return Some(false);
            }
        } else if p != (rank - 2) as i64 {
            return Some(false);
        }
    }

    Some(true)
}

/// Check if an output name is consumed by exactly one node.
fn is_single_use(output_name: &str, consumer: &HashMap<String, Vec<usize>>) -> bool {
    consumer
        .get(output_name)
        .is_some_and(|consumers| consumers.len() == 1)
}

fn build_producer_map(nodes: &[RawNode]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            map.insert(out.name.clone(), i);
        }
    }
    map
}

fn build_consumer_map(nodes: &[RawNode]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for inp in &node.inputs {
            map.entry(inp.name.clone()).or_default().push(i);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, TensorType, ValueSource};
    use crate::simplify::tests::node;
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    /// Create a rank-4 float32 tensor argument.
    fn tensor4(name: &str) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 4,
                static_shape: None,
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    /// Create a constant scalar f32 argument.
    fn const_f32(name: &str, value: f32) -> Argument {
        let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
        let data_ref = TensorDataRef::new(bytes, vec![1], DType::F32);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        let value_store = ValueStore::new(std::rc::Rc::new(store), std::rc::Rc::new(constant_map));
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 0,
                static_shape: Some(vec![]),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
        }
    }

    fn transpose_node(name: &str, input: &str, output: &str, perm: Vec<i64>) -> RawNode {
        RawNode {
            node_type: NodeType::Transpose,
            name: name.to_string(),
            inputs: vec![tensor4(input)],
            outputs: vec![tensor4(output)],
            attrs: [("perm".to_string(), AttributeValue::Int64s(perm))]
                .into_iter()
                .collect(),
        }
    }

    fn matmul_node(name: &str, a: &str, b: &str, output: &str) -> RawNode {
        RawNode {
            node_type: NodeType::MatMul,
            name: name.to_string(),
            inputs: vec![tensor4(a), tensor4(b)],
            outputs: vec![tensor4(output)],
            attrs: Default::default(),
        }
    }

    fn softmax_node(name: &str, input: &str, output: &str, axis: i64) -> RawNode {
        RawNode {
            node_type: NodeType::Softmax,
            name: name.to_string(),
            inputs: vec![tensor4(input)],
            outputs: vec![tensor4(output)],
            attrs: [("axis".to_string(), AttributeValue::Int64(axis))]
                .into_iter()
                .collect(),
        }
    }

    fn binary_node(name: &str, op: NodeType, a: Argument, b: Argument, output: &str) -> RawNode {
        RawNode {
            node_type: op,
            name: name.to_string(),
            inputs: vec![a, b],
            outputs: vec![tensor4(output)],
            attrs: Default::default(),
        }
    }

    /// Build: Transpose(K) -> MatMul(Q,K^T) -> Div(scale) -> Softmax(-1) -> MatMul(scores,V)
    fn build_sdpa_with_div(scale: f32) -> Vec<RawNode> {
        vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            binary_node(
                "div_scale",
                NodeType::Div,
                tensor4("qk"),
                const_f32("scale", scale),
                "qk_scaled",
            ),
            softmax_node("softmax", "qk_scaled", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ]
    }

    /// Build: Transpose(K) -> MatMul(Q,K^T) -> Mul(scale) -> Softmax(-1) -> MatMul(scores,V)
    fn build_sdpa_with_mul(scale: f32) -> Vec<RawNode> {
        vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            binary_node(
                "mul_scale",
                NodeType::Mul,
                tensor4("qk"),
                const_f32("scale", scale),
                "qk_scaled",
            ),
            softmax_node("softmax", "qk_scaled", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ]
    }

    /// Build: Transpose(K) -> MatMul(Q,K^T) -> Div(scale) -> Add(mask) -> Softmax(-1) -> MatMul(scores,V)
    fn build_sdpa_with_mask(scale: f32) -> Vec<RawNode> {
        vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            binary_node(
                "div_scale",
                NodeType::Div,
                tensor4("qk"),
                const_f32("scale", scale),
                "qk_scaled",
            ),
            binary_node(
                "add_mask",
                NodeType::Add,
                tensor4("qk_scaled"),
                tensor4("mask"),
                "qk_masked",
            ),
            softmax_node("softmax", "qk_masked", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ]
    }

    /// Build: Transpose(K) -> MatMul(Q,K^T) -> Softmax(-1) -> MatMul(scores,V) (no scaling)
    fn build_sdpa_no_scale() -> Vec<RawNode> {
        vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            softmax_node("softmax", "qk", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ]
    }

    #[test]
    fn test_basic_sdpa_with_div() {
        let nodes = build_sdpa_with_div(8.0);
        let result = coalesce_attention(nodes);

        let attention = result
            .iter()
            .find(|n| n.node_type == NodeType::Attention)
            .expect("should produce an Attention node");

        assert_eq!(attention.inputs[0].name, "q");
        assert_eq!(attention.inputs[1].name, "k");
        assert_eq!(attention.inputs[2].name, "v");
        assert_eq!(attention.outputs[0].name, "output");

        // Div by 8.0 -> scale = 1/8 = 0.125
        let scale = attention.attrs.get("scale").unwrap().clone().into_f32();
        assert!((scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_basic_sdpa_with_mul() {
        let nodes = build_sdpa_with_mul(0.125);
        let result = coalesce_attention(nodes);

        let attention = result
            .iter()
            .find(|n| n.node_type == NodeType::Attention)
            .expect("should produce an Attention node");

        assert_eq!(attention.inputs[0].name, "q");
        assert_eq!(attention.inputs[1].name, "k");
        assert_eq!(attention.inputs[2].name, "v");

        let scale = attention.attrs.get("scale").unwrap().clone().into_f32();
        assert!((scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_sdpa_with_mask() {
        let nodes = build_sdpa_with_mask(8.0);
        let result = coalesce_attention(nodes);

        let attention = result
            .iter()
            .find(|n| n.node_type == NodeType::Attention)
            .expect("should produce an Attention node");

        assert_eq!(attention.inputs.len(), 4);
        assert_eq!(attention.inputs[0].name, "q");
        assert_eq!(attention.inputs[1].name, "k");
        assert_eq!(attention.inputs[2].name, "v");
        assert_eq!(attention.inputs[3].name, "mask");

        let scale = attention.attrs.get("scale").unwrap().clone().into_f32();
        assert!((scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_sdpa_no_scaling() {
        let nodes = build_sdpa_no_scale();
        let result = coalesce_attention(nodes);

        let attention = result
            .iter()
            .find(|n| n.node_type == NodeType::Attention)
            .expect("should produce an Attention node");

        assert_eq!(attention.inputs[0].name, "q");
        assert_eq!(attention.inputs[1].name, "k");
        assert_eq!(attention.inputs[2].name, "v");

        // No explicit scale -> defaults handled by Attention processor
        assert!(attention.attrs.get("scale").is_none());
    }

    #[test]
    fn test_sdpa_multi_use_not_matched() {
        // QK MatMul output is used by both Div and another consumer
        let mut nodes = build_sdpa_with_div(8.0);
        // Add a node that also consumes "qk"
        nodes.push(node("other", NodeType::Relu, &["qk"], &["other_out"]));

        let result = coalesce_attention(nodes);
        assert!(
            !result.iter().any(|n| n.node_type == NodeType::Attention),
            "should not match when intermediate output has multiple consumers"
        );
    }

    #[test]
    fn test_sdpa_wrong_softmax_axis() {
        // Softmax on axis 0 instead of last dim
        let nodes = vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            softmax_node("softmax", "qk", "attn_weights", 0),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ];

        let result = coalesce_attention(nodes);
        assert!(
            !result.iter().any(|n| n.node_type == NodeType::Attention),
            "should not match when Softmax axis is not the last dimension"
        );
    }

    #[test]
    fn test_sdpa_wrong_transpose_perm() {
        // Transpose that doesn't swap last two dims
        let nodes = vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 2, 1, 3]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            softmax_node("softmax", "qk", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ];

        let result = coalesce_attention(nodes);
        assert!(
            !result.iter().any(|n| n.node_type == NodeType::Attention),
            "should not match when Transpose doesn't swap last two dims"
        );
    }

    #[test]
    fn test_non_sdpa_graph_unchanged() {
        let nodes = vec![
            node("relu1", NodeType::Relu, &["input"], &["r1"]),
            node("relu2", NodeType::Relu, &["r1"], &["output"]),
        ];

        let result = coalesce_attention(nodes);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].node_type, NodeType::Relu);
        assert_eq!(result[1].node_type, NodeType::Relu);
    }

    #[test]
    fn test_two_sdpa_patterns() {
        // Two independent SDPA patterns in the same graph
        let nodes = vec![
            // First SDPA
            transpose_node("transpose_k1", "k1", "k1_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul1", "q1", "k1_t", "qk1"),
            softmax_node("softmax1", "qk1", "attn1", -1),
            matmul_node("sv_matmul1", "attn1", "v1", "out1"),
            // Second SDPA
            transpose_node("transpose_k2", "k2", "k2_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul2", "q2", "k2_t", "qk2"),
            softmax_node("softmax2", "qk2", "attn2", -1),
            matmul_node("sv_matmul2", "attn2", "v2", "out2"),
        ];

        let result = coalesce_attention(nodes);
        let attention_count = result
            .iter()
            .filter(|n| n.node_type == NodeType::Attention)
            .count();
        assert_eq!(attention_count, 2, "should coalesce both SDPA patterns");
    }

    #[test]
    fn test_sdpa_with_mask_no_scale() {
        // Add(mask) directly after MatMul, no Div/Mul scale
        let nodes = vec![
            transpose_node("transpose_k", "k", "k_t", vec![0, 1, 3, 2]),
            matmul_node("qk_matmul", "q", "k_t", "qk"),
            binary_node(
                "add_mask",
                NodeType::Add,
                tensor4("qk"),
                tensor4("mask"),
                "qk_masked",
            ),
            softmax_node("softmax", "qk_masked", "attn_weights", -1),
            matmul_node("sv_matmul", "attn_weights", "v", "output"),
        ];

        let result = coalesce_attention(nodes);
        let attention = result
            .iter()
            .find(|n| n.node_type == NodeType::Attention)
            .expect("should produce an Attention node");

        assert_eq!(attention.inputs.len(), 4);
        assert_eq!(attention.inputs[3].name, "mask");
        assert!(attention.attrs.get("scale").is_none());
    }
}
