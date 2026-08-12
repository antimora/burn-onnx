//! Custom operator codegen hooks.
//!
//! Users implement [`CustomOp`] for each non-built-in ONNX operator in their
//! model and register the implementations via `ModelGen::register_custom_op`.
//! The [`HookRegistry`] stores them and doubles as the `CustomOpInference`
//! implementation handed to the `onnx-ir` parse pipeline.

use std::collections::HashMap;

use onnx_ir::{ArgType, CustomNode, CustomOpInference, HookCoverage, ProcessError};
use proc_macro2::TokenStream;

use crate::burn::node_traits::Field;
use crate::ext::{CodegenContext, Imports};
use burn_store::TensorSnapshot;

/// Codegen hook for one custom (non-built-in) ONNX operator.
///
/// A hook is matched by ONNX operator identity `(op_type, domain)` and is
/// responsible for both type inference (during parsing) and code generation.
/// Constant inputs are readable via `node.inputs[i].value()` in every method.
pub trait CustomOp: Send + Sync + 'static {
    /// ONNX op_type this hook handles (e.g. "FftReal").
    fn op_type(&self) -> &str;

    /// ONNX domain. Empty string = default ONNX domain.
    fn domain(&self) -> &str {
        ""
    }

    /// Min/max opset gate, checked against the node's domain opset.
    /// `None` max = unbounded.
    fn opset_range(&self) -> (usize, Option<usize>) {
        (1, None)
    }

    /// Infer output ArgTypes. Called during onnx-ir type inference.
    ///
    /// Must return exactly `node.outputs.len()` types; the pipeline rejects a
    /// mismatch. Consumers' output preferences are not consulted for custom
    /// ops: this hook is the sole authority on its output types.
    fn infer_output_types(&self, node: &CustomNode) -> Result<Vec<ArgType>, ProcessError>;

    /// Generate the forward-pass code for this node.
    fn forward(&self, node: &CustomNode, ctx: &mut CodegenContext<'_, '_>) -> TokenStream;

    /// Optional: extra imports emitted as `use` statements in the model file.
    fn register_imports(&self, _imports: &mut Imports<'_>) {}

    /// Optional: declare a module field (e.g. learnable params or state).
    fn field(&self, _node: &CustomNode) -> Option<Field> {
        None
    }

    /// Optional: weights/snapshot collection (parallels the built-in nodes).
    fn collect_snapshots(&self, _node: &CustomNode, _field_name: &str) -> Vec<TensorSnapshot> {
        vec![]
    }
}

/// Registry of user codegen hooks, keyed by ONNX operator identity.
///
/// Owned by `ModelGen` (behind `Arc`), shared with the onnx-ir parse pipeline
/// as its `CustomOpInference` implementation and with `BurnGraph` for codegen
/// dispatch.
#[derive(Default)]
pub(crate) struct HookRegistry {
    customs: HashMap<(String, String), Box<dyn CustomOp>>,
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("customs", &self.customs.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl HookRegistry {
    /// Register a custom op hook. Panics on duplicate `(op_type, domain)`:
    /// registration happens in build scripts, where an immediate, attributable
    /// panic beats a silently shadowed hook.
    pub(crate) fn add_custom_op(&mut self, op: Box<dyn CustomOp>) {
        let key = (op.op_type().to_string(), op.domain().to_string());
        if self.customs.contains_key(&key) {
            panic!(
                "Duplicate custom op registration for '{}::{}'",
                key.1, key.0
            );
        }
        self.customs.insert(key, op);
    }

    /// Look up the hook for an ONNX operator identity.
    pub(crate) fn custom_for(&self, op_type: &str, domain: &str) -> Option<&dyn CustomOp> {
        self.customs
            .get(&(op_type.to_string(), domain.to_string()))
            .map(|b| b.as_ref())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.customs.is_empty()
    }
}

impl CustomOpInference for HookRegistry {
    fn coverage(&self, op_type: &str, domain: &str, opset: usize) -> HookCoverage {
        match self.custom_for(op_type, domain) {
            None => HookCoverage::NoHook,
            Some(op) => {
                let (min, max) = op.opset_range();
                if opset >= min && max.is_none_or(|m| opset <= m) {
                    HookCoverage::Covered
                } else {
                    HookCoverage::OpsetMismatch {
                        supported_min: min,
                        supported_max: max,
                    }
                }
            }
        }
    }

    fn infer(&self, node: &CustomNode) -> Result<Option<Vec<ArgType>>, ProcessError> {
        match self.custom_for(&node.op_type, &node.domain) {
            Some(op) => op.infer_output_types(node).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::ir::{DType, TensorType};

    struct TestOp;

    impl CustomOp for TestOp {
        fn op_type(&self) -> &str {
            "FftReal"
        }

        fn domain(&self) -> &str {
            "custom_domain"
        }

        fn opset_range(&self) -> (usize, Option<usize>) {
            (2, Some(4))
        }

        fn infer_output_types(&self, _node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
            Ok(vec![ArgType::Tensor(TensorType::new(DType::F32, 2, None))])
        }

        fn forward(&self, _node: &CustomNode, _ctx: &mut CodegenContext<'_, '_>) -> TokenStream {
            TokenStream::new()
        }
    }

    fn registry_with_test_op() -> HookRegistry {
        let mut registry = HookRegistry::default();
        registry.add_custom_op(Box::new(TestOp));
        registry
    }

    #[test]
    fn coverage_checks_identity_and_opset_range() {
        let registry = registry_with_test_op();
        assert_eq!(
            registry.coverage("FftReal", "custom_domain", 3),
            HookCoverage::Covered
        );
        assert_eq!(
            registry.coverage("FftReal", "custom_domain", 1),
            HookCoverage::OpsetMismatch {
                supported_min: 2,
                supported_max: Some(4),
            }
        );
        assert_eq!(
            registry.coverage("FftReal", "custom_domain", 5),
            HookCoverage::OpsetMismatch {
                supported_min: 2,
                supported_max: Some(4),
            }
        );
        // Same op_type, different domain: distinct ONNX identity
        assert_eq!(
            registry.coverage("FftReal", "other_domain", 3),
            HookCoverage::NoHook
        );
    }

    #[test]
    #[should_panic(expected = "Duplicate custom op registration")]
    fn duplicate_registration_panics() {
        let mut registry = registry_with_test_op();
        registry.add_custom_op(Box::new(TestOp));
    }
}
