/// Tests for simplification passes.
///
/// Verifies that graph simplification correctly transforms patterns
/// into more efficient node representations.
mod test_utils;

use test_utils::*;

/// Pre-scaled SDPA pattern (as seen in RF-DETR) should be coalesced into a single
/// Attention node when simplification is enabled:
///
/// ```text
/// Q -> Transpose([0,2,1,3]) -> Mul(sqrt_scale) -\
///                                                 MatMul -> Softmax -> MatMul(scores, V)
/// K -> Transpose([0,2,3,1]) -> Mul(sqrt_scale) -/
/// ```
#[test]
fn test_prescaled_sdpa_coalesced() {
    let graph = load_onnx_simplified("prescaled_sdpa.onnx");

    // Find the Attention node
    let attention = graph
        .nodes
        .iter()
        .find(|n| matches!(n, onnx_ir::ir::Node::Attention { .. }));
    assert!(
        attention.is_some(),
        "Pre-scaled SDPA pattern should be coalesced into an Attention node"
    );

    // Verify scale is None (dynamic sqrt_scale computes to the default 1/sqrt(head_dim))
    if let Some(onnx_ir::ir::Node::Attention(node)) = attention {
        assert!(
            node.config.scale.is_none(),
            "Dynamic pre-scale should result in default scale (None)"
        );
    }

    // The pattern's MatMul, Softmax, Mul(scale) nodes should be eliminated
    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Softmax { .. })),
        "Softmax should be absorbed into Attention node"
    );
}

/// Without simplification, the pre-scaled SDPA pattern should remain decomposed.
#[test]
fn test_prescaled_sdpa_not_coalesced_without_simplify() {
    let graph = load_onnx("prescaled_sdpa.onnx");

    assert!(
        !has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Attention { .. })),
        "Attention node should not appear without simplification"
    );

    assert!(
        has_node_type(&graph, |n| matches!(n, onnx_ir::ir::Node::Softmax { .. })),
        "Softmax should remain without simplification"
    );
}
