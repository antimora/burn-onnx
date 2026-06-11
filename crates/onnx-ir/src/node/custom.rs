//! # Custom (non-built-in) operators
//!
//! Nodes whose `(domain, op_type)` does not map to a built-in `NodeType` are
//! preserved as `NodeType::Custom` / `Node::Custom(CustomNode)` instead of
//! failing the parse. Type inference for them is supplied by user hooks
//! registered in `burn-onnx` (see `DESIGN-CUSTOM-OPS.md`); without a hook, a
//! best-effort same-as-input fallback keeps the graph buildable for inspection.

use crate::ir::{Argument, Node, PublicAttributesOwned, RawNode};
use crate::processor::{NodeProcessor, NodeSpec, OutputPreferences, ProcessError, same_as_input};

/// Public view of a custom (non-built-in) ONNX node.
///
/// Inputs are full [`Argument`] values with their value stores attached, so
/// constant input data is readable via `Argument::value()`.
#[derive(Debug, Clone)]
pub struct CustomNode {
    /// Node name (sanitized, unique within the graph)
    pub name: String,
    /// Raw ONNX op_type, e.g. "FftReal"
    pub op_type: String,
    /// Raw ONNX domain, e.g. "custom_domain" ("" = default domain)
    pub domain: String,
    /// The inputs of the node.
    pub inputs: Vec<Argument>,
    /// The outputs of the node.
    pub outputs: Vec<Argument>,
    /// ONNX attributes, exposed read-only.
    pub attrs: PublicAttributesOwned,
    /// Opset version of `domain` from the model's opset_import.
    pub opset: usize,
}

impl core::fmt::Display for CustomNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.domain.is_empty() {
            write!(f, "Custom({})", self.op_type)
        } else {
            write!(f, "Custom({}::{})", self.domain, self.op_type)
        }
    }
}

/// Build the public CustomNode view from a RawNode at any pipeline stage.
pub(crate) fn custom_node_view(node: &RawNode) -> CustomNode {
    let identity = node
        .custom_identity
        .as_ref()
        .expect("RawNode with NodeType::Custom must carry a CustomIdentity (parser invariant)");
    CustomNode {
        name: node.name.clone(),
        op_type: identity.op_type.clone(),
        domain: identity.domain.clone(),
        inputs: node.inputs.clone(),
        outputs: node.outputs.clone(),
        attrs: PublicAttributesOwned::from_internal(&node.attrs),
        opset: identity.domain_opset,
    }
}

/// Hook-free processor for `NodeType::Custom`, registered in the global registry.
///
/// Everything a processor must do for a custom node is hook-independent except
/// type inference: the spec is permissive, constants are never lifted, the node
/// is never a no-op, and `build_node` just snapshots the raw node into the
/// public `CustomNode` view. Type inference here is a best-effort fallback for
/// runs without registered hooks; the hook-aware path overrides it during the
/// type-inference phase.
pub(crate) struct CustomProcessor;

impl NodeProcessor for CustomProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        // Permissive: unknown schema, so any opset and any I/O count.
        NodeSpec::default()
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Best-effort fallback when no inference hook is in play. Guarded:
        // same_as_input() indexes inputs[0]/outputs[0].
        if !node.inputs.is_empty() && !node.outputs.is_empty() {
            log::warn!(
                "No inference hook for custom op '{}'; assuming output type equals input type",
                node.name
            );
            same_as_input(node);
        }
        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Custom(custom_node_view(&builder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, CustomIdentity, DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn make_custom_node() -> RawNode {
        let mut node = TestNodeBuilder::new(NodeType::Custom, "test_custom")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 0, None)
            .attr_int("n_fft", 1024)
            .build();
        node.custom_identity = Some(CustomIdentity {
            op_type: "FftReal".to_string(),
            domain: "custom_domain".to_string(),
            domain_opset: 2,
        });
        node
    }

    #[test]
    fn infer_types_falls_back_to_same_as_input() {
        let mut node = make_custom_node();
        let processor = CustomProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn infer_types_tolerates_no_inputs() {
        let mut node = make_custom_node();
        node.inputs.clear();
        let processor = CustomProcessor;
        // Must not panic on the empty-input case.
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
    }

    #[test]
    fn build_node_preserves_identity_and_attrs() {
        let node = make_custom_node();
        let built = CustomProcessor.build_node(node, 16);
        let Node::Custom(custom) = built else {
            panic!("expected Node::Custom");
        };
        assert_eq!(custom.op_type, "FftReal");
        assert_eq!(custom.domain, "custom_domain");
        assert_eq!(custom.opset, 2);
        assert_eq!(custom.attrs.get_i64("n_fft"), Some(1024));
        assert_eq!(custom.name, "test_custom");
        assert_eq!(custom.inputs.len(), 1);
        assert!(matches!(
            custom.inputs[0].ty,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 3,
                ..
            })
        ));
    }

    #[test]
    fn node_type_accessor_and_display() {
        let built = CustomProcessor.build_node(make_custom_node(), 16);
        assert_eq!(built.node_type(), NodeType::Custom);
        assert_eq!(built.to_string(), "Custom(custom_domain::FftReal)");
    }

    #[test]
    fn from_str_never_resolves_to_custom() {
        use core::str::FromStr;
        assert!(NodeType::from_str("Custom").is_err());
        assert!(NodeType::from_str("custom").is_err());
    }
}
