use std::collections::{BTreeSet, HashMap, HashSet};

use crate::ir::{ArgType, Argument, RawNode, ValueSource};
use crate::pipeline::PipelineHooks;
use crate::processor::{
    ArgPreference, OutputPreferences, get_processor_registry, validate_no_rank_zero_tensors,
    validate_node_spec,
};

/// Names of node inputs that are still computed at runtime.
pub(super) fn dynamic_input_names(nodes: &[RawNode]) -> HashSet<String> {
    nodes
        .iter()
        .flat_map(|n| &n.inputs)
        .filter(|arg| arg.value_source == ValueSource::Dynamic)
        .map(|arg| arg.name.clone())
        .collect()
}

/// Re-infer the types of nodes that read a value which became constant during simplification.
///
/// Types are inferred before simplification, so a node whose input only folds to a constant
/// later keeps the type it was given for a runtime input. `ConstantOfShape(Shape(v))` with a
/// Shape-typed `v` is one: the length folds to a constant, but the ConstantOfShape stays a
/// device tensor and takes the Mul/Equal/Where after it onto the device too, only for Expand
/// to read the result back.
///
/// Each node reading a value that became constant in this iteration (`was_dynamic` holds the
/// input names that were runtime values before this iteration's passes) starts a retype that
/// is carried to every consumer whose input type changes. Nodes are kept in topological order,
/// so visiting them by index sees each producer before its consumers. The retype is applied
/// only if all of these hold, and is dropped as a whole otherwise:
/// - every touched node re-infers and passes the checks type inference runs
/// - no graph output changes type, so the generated `forward` signature stays the same
/// - an output only changes kind when it becomes a Shape, and every Shape it produces either
///   feeds a node that outputs a Shape itself or an input that asks for a Shape. Inference
///   accepts a Shape in places codegen cannot (a CumSum of position ids), so a value only
///   moves to the host when it ends up where a host shape is wanted.
///
/// Returns whether any retype was applied.
pub(super) fn retype_new_constant_consumers(
    nodes: &mut [RawNode],
    graph_outputs: &[Argument],
    was_dynamic: &HashSet<String>,
    opset: usize,
    hooks: &PipelineHooks,
) -> bool {
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            consumers.entry(input.name.as_str()).or_default().push(i);
        }
    }
    let consumers: HashMap<String, Vec<usize>> = consumers
        .into_iter()
        .map(|(name, users)| (name.to_string(), users))
        .collect();
    let graph_outputs: HashSet<&str> = graph_outputs.iter().map(|a| a.name.as_str()).collect();

    let roots: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            n.inputs.iter().any(|arg| {
                arg.value_source == ValueSource::Constant && was_dynamic.contains(&arg.name)
            })
        })
        .map(|(i, _)| i)
        .collect();

    let mut applied = false;
    for root in roots {
        let ctx = Retype {
            nodes,
            consumers: &consumers,
            graph_outputs: &graph_outputs,
            opset,
            hooks,
        };
        if let Some(updated) = ctx.run(root) {
            for (i, node) in updated {
                nodes[i] = node;
            }
            applied = true;
        }
    }
    applied
}

struct Retype<'a> {
    nodes: &'a [RawNode],
    consumers: &'a HashMap<String, Vec<usize>>,
    graph_outputs: &'a HashSet<&'a str>,
    opset: usize,
    hooks: &'a PipelineHooks,
}

impl Retype<'_> {
    /// Retype from `root` on copies of the touched nodes. Returns the copies whose types
    /// changed, or `None` when the retype must be dropped.
    fn run(&self, root: usize) -> Option<HashMap<usize, RawNode>> {
        let registry = get_processor_registry();
        let mut touched: HashMap<usize, RawNode> = HashMap::new();
        // Values that became a Shape in this retype; whoever reads them must want a Shape
        let mut new_shapes: HashSet<String> = HashSet::new();
        let mut retyped_outputs: Vec<(String, String, ArgType)> = Vec::new();
        let mut pending = BTreeSet::from([root]);

        while let Some(i) = pending.pop_first() {
            let mut node = touched
                .get(&i)
                .cloned()
                .unwrap_or_else(|| self.nodes[i].clone());
            let prefs = self.output_preferences(&node, &touched);
            let processor = self.hooks.resolve(&node.node_type, registry);
            let inferred = validate_node_spec(&node, self.opset, &processor.spec())
                .and_then(|_| processor.infer_types(&mut node, self.opset, &prefs))
                .and_then(|_| validate_no_rank_zero_tensors(&node));
            if let Err(e) = inferred {
                log::debug!(
                    "Simplification: not retyping from '{}', '{}' does not accept it: {}",
                    self.nodes[root].name,
                    node.name,
                    e
                );
                return None;
            }

            let old = touched.get(&i).unwrap_or(&self.nodes[i]);
            let mut retyped = Vec::new();
            for (new, old) in node.outputs.iter().zip(&old.outputs) {
                if new.ty == old.ty {
                    continue;
                }
                if self.graph_outputs.contains(new.name.as_str()) {
                    return None;
                }
                if new.ty.is_shape() && !old.ty.is_shape() {
                    new_shapes.insert(new.name.clone());
                } else if !same_kind(&new.ty, &old.ty) {
                    return None;
                }
                retyped.push((new.name.clone(), new.ty.clone()));
            }

            // A Shape input must lead to a Shape output here, or be asked for as one
            let wants_shape = |name: &str| {
                processor
                    .input_preferences(&node, self.opset)
                    .ok()
                    .flatten()
                    .is_some_and(|p| {
                        p.get(name)
                            .iter()
                            .any(|pref| matches!(pref, ArgPreference::Shape))
                    })
            };
            let outputs_shape = node.outputs.iter().any(|o| o.ty.is_shape());
            for input in &node.inputs {
                if new_shapes.contains(&input.name) && !outputs_shape && !wants_shape(&input.name) {
                    return None;
                }
            }

            for (name, ty) in &retyped {
                retyped_outputs.push((node.name.clone(), name.clone(), ty.clone()));
            }
            touched.insert(i, node);
            for (name, ty) in retyped {
                for &c in self.consumers.get(&name).into_iter().flatten() {
                    let consumer = touched.entry(c).or_insert_with(|| self.nodes[c].clone());
                    for input in consumer.inputs.iter_mut().filter(|a| a.name == name) {
                        let mut merged = ty.clone();
                        merged.merge_static_shape(&input.ty);
                        input.ty = merged;
                    }
                    pending.insert(c);
                }
            }
        }

        if retyped_outputs.is_empty() {
            return None;
        }
        for (node, output, ty) in retyped_outputs {
            log::info!("Simplification: '{node}' output '{output}' retyped to {ty:?}");
        }
        Some(touched)
    }

    /// Rebuild the output preferences type inference collected from the node's consumers.
    fn output_preferences(
        &self,
        node: &RawNode,
        touched: &HashMap<usize, RawNode>,
    ) -> OutputPreferences {
        let registry = get_processor_registry();
        let mut prefs = OutputPreferences::new();
        for output in &node.outputs {
            for &c in self.consumers.get(&output.name).into_iter().flatten() {
                let consumer = touched.get(&c).unwrap_or(&self.nodes[c]);
                let processor = self.hooks.resolve(&consumer.node_type, registry);
                if let Ok(Some(input_prefs)) = processor.input_preferences(consumer, self.opset) {
                    for pref in input_prefs.get(&output.name) {
                        prefs.add(output.name.clone(), consumer.name.clone(), pref.clone());
                    }
                }
            }
        }
        prefs
    }
}

/// Whether two types differ at most in their known static dimensions.
fn same_kind(a: &ArgType, b: &ArgType) -> bool {
    match (a, b) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => a.dtype == b.dtype && a.rank == b.rank,
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::graph_state::GraphState;
    use crate::ir::{AttributeValue, Attributes, DType, NodeType, TensorData, TensorType};
    use crate::tensor_store::TensorDataRef;

    fn tensor(name: &str, dtype: DType, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype,
                rank,
                static_shape: None,
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn node(name: &str, node_type: NodeType, inputs: Vec<Argument>, output: Argument) -> RawNode {
        RawNode {
            custom_identity: None,
            node_type,
            name: name.to_string(),
            inputs,
            outputs: vec![output],
            attrs: Attributes::new(),
        }
    }

    /// `ConstantOfShape(len)` filled with int64 ones, typed as it was while `len` was a
    /// runtime value, with `len` now folded to the constant [3].
    fn ones(state: &Rc<RefCell<GraphState>>) -> RawNode {
        let bytes: Vec<u8> = 3i64.to_ne_bytes().to_vec();
        let data = TensorDataRef::new(bytes::Bytes::from(bytes), vec![1], DType::I64);
        let mut gs = state.borrow_mut();
        gs.register_constant("len".to_string(), data);
        let len = Argument {
            name: "len".to_string(),
            ty: ArgType::Shape(1),
            value_source: ValueSource::Constant,
            value_store: Some(gs.build_value_store()),
        };
        let mut ones = node(
            "ones",
            NodeType::ConstantOfShape,
            vec![len],
            tensor("ones_out", DType::I64, 1),
        );
        ones.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData::new(vec![1i64], vec![1])),
        );
        ones
    }

    fn retype(nodes: &mut [RawNode], graph_outputs: &[Argument], was_dynamic: &[&str]) -> bool {
        let was_dynamic = was_dynamic.iter().map(|s| s.to_string()).collect();
        retype_new_constant_consumers(
            nodes,
            graph_outputs,
            &was_dynamic,
            16,
            &PipelineHooks::new(None),
        )
    }

    fn types(nodes: &[RawNode]) -> Vec<ArgType> {
        nodes
            .iter()
            .flat_map(|n| n.inputs.iter().chain(&n.outputs))
            .map(|a| a.ty.clone())
            .collect()
    }

    fn state() -> Rc<RefCell<GraphState>> {
        Rc::new(RefCell::new(GraphState::new(&[], &[], Vec::new(), &[])))
    }

    fn expand(state: &Rc<RefCell<GraphState>>) -> Vec<RawNode> {
        vec![
            ones(state),
            node(
                "expand",
                NodeType::Expand,
                vec![
                    tensor("x", DType::F32, 3),
                    tensor("ones_out", DType::I64, 1),
                ],
                tensor("y", DType::F32, 3),
            ),
        ]
    }

    #[test]
    fn shape_wanted_by_expand_is_retyped() {
        let state = state();
        let mut nodes = expand(&state);
        assert!(retype(&mut nodes, &[], &["len"]));
        assert_eq!(nodes[0].outputs[0].ty, ArgType::Shape(3));
        assert_eq!(nodes[1].inputs[1].ty, ArgType::Shape(3));
    }

    #[test]
    fn nothing_newly_constant_is_left_alone() {
        let state = state();
        let mut nodes = expand(&state);
        assert!(!retype(&mut nodes, &[], &[]));
        assert_eq!(nodes[0].outputs[0].ty, tensor("", DType::I64, 1).ty);
    }

    #[test]
    fn graph_output_keeps_its_type() {
        // Retyping it would change the generated forward signature
        let state = state();
        let mut nodes = vec![ones(&state)];
        let outputs = [tensor("ones_out", DType::I64, 1)];
        assert!(!retype(&mut nodes, &outputs, &["len"]));
        assert_eq!(nodes[0].outputs[0].ty, outputs[0].ty);
    }

    #[test]
    fn shape_not_wanted_downstream_is_dropped() {
        // CumSum accepts a Shape in inference but has no host codegen; the Cast after it
        // takes a tensor, so the position-id chain stays on the device as before
        let state = state();
        let mut nodes = vec![
            ones(&state),
            node(
                "cumsum",
                NodeType::CumSum,
                vec![
                    tensor("ones_out", DType::I64, 1),
                    Argument::from_const_i64("axis", 0),
                ],
                tensor("pos", DType::I64, 1),
            ),
            node(
                "cast",
                NodeType::Cast,
                vec![tensor("pos", DType::I64, 1)],
                tensor("pos_f", DType::F32, 1),
            ),
        ];
        nodes[2]
            .attrs
            .insert("to".to_string(), AttributeValue::Int64(1));
        let before = types(&nodes);
        assert!(!retype(&mut nodes, &[], &["len"]));
        assert_eq!(types(&nodes), before);
    }

    #[test]
    fn consumer_rejecting_shape_drops_the_retype() {
        // MatMul fails to infer with a Shape input; no node is left half-retyped
        let state = state();
        let mut nodes = vec![
            ones(&state),
            node(
                "matmul",
                NodeType::MatMul,
                vec![
                    tensor("ones_out", DType::I64, 1),
                    tensor("w", DType::I64, 2),
                ],
                tensor("mm", DType::I64, 1),
            ),
        ];
        let before = types(&nodes);
        assert!(!retype(&mut nodes, &[], &["len"]));
        assert_eq!(types(&nodes), before);
    }
}
