//! ONNX to IR conversion pipeline orchestrator
//!
//! This module provides the high-level orchestration of the ONNX conversion process.
//! It clearly shows the entire conversion flow from start to finish.
//!
//! # Zero-Copy Loading
//!
//! When the `mmap` feature is enabled (default), files are memory-mapped for zero-copy
//! tensor loading. This significantly reduces memory usage for large models.
//!
//! # Usage
//!
//! ```ignore
//! use onnx_ir::OnnxGraphBuilder;
//!
//! // Build from file
//! let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
//!
//! // Build from bytes
//! let graph = OnnxGraphBuilder::new().parse_bytes(&bytes)?;
//!
//! // Build from reader
//! let graph = OnnxGraphBuilder::new().parse_reader(file)?;
//! ```

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::{fmt, fs::File, path::Path};

use protobuf::Message;

use crate::{ir::OnnxGraph, processor::ProcessError, protos::ModelProto};

use super::phases::{
    finalization, initialization, node_conversion, post_processing, type_inference,
};

/// Errors that can occur when parsing ONNX models
#[derive(Debug)]
pub enum Error {
    /// Failed to open or read the ONNX file
    Io { path: String, error: std::io::Error },

    /// Failed to parse ONNX protobuf format
    InvalidFormat { path: Option<String>, error: String },

    /// Model graph nodes are not topologically sorted (ONNX spec violation)
    InvalidGraphStructure { reason: String },

    /// Missing required opset version for default domain
    MissingOpsetVersion,

    /// Type inference failed during IR conversion
    TypeInference(ProcessError),

    /// Generic processing error
    Processing(ProcessError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, error } => {
                write!(f, "Failed to open ONNX file '{}': {}", path, error)
            }
            Error::InvalidFormat { path, error } => {
                if let Some(p) = path {
                    write!(f, "Invalid ONNX format in '{}': {}", p, error)
                } else {
                    write!(f, "Invalid ONNX format: {}", error)
                }
            }
            Error::InvalidGraphStructure { reason } => {
                write!(f, "Invalid ONNX graph structure: {}", reason)
            }
            Error::MissingOpsetVersion => {
                write!(
                    f,
                    "ONNX model must specify opset version for default domain"
                )
            }
            Error::TypeInference(e) => {
                write!(f, "Type inference failed: {e}")
            }
            Error::Processing(e) => {
                write!(f, "Processing error: {e}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<ProcessError> for Error {
    fn from(error: ProcessError) -> Self {
        Error::Processing(error)
    }
}

/// ONNX IR builder with fluent API
///
/// Builds ONNX IR graphs from various sources (files, bytes, readers).
/// Future configuration options can be added without breaking changes.
///
/// # Examples
///
/// ```ignore
/// use onnx_ir::OnnxGraphBuilder;
///
/// // Build from file (uses mmap when feature is enabled)
/// let graph = OnnxGraphBuilder::new().parse_file("model.onnx")?;
///
/// // Build from bytes
/// let graph = OnnxGraphBuilder::new().parse_bytes(&model_bytes)?;
///
/// // Build from reader
/// let graph = OnnxGraphBuilder::new().parse_reader(std::io::Cursor::new(data))?;
/// ```
#[derive(Debug, Clone)]
pub struct OnnxGraphBuilder {
    /// Whether to run graph simplification passes (default: true)
    simplify: bool,
}

impl Default for OnnxGraphBuilder {
    fn default() -> Self {
        Self { simplify: true }
    }
}

impl OnnxGraphBuilder {
    /// Create a new ONNX graph builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable graph simplification passes (default: true)
    ///
    /// When enabled, the builder runs optimization passes on the IR graph
    /// such as dead node elimination, common subexpression elimination, and
    /// pattern-based simplifications.
    pub fn simplify(mut self, simplify: bool) -> Self {
        self.simplify = simplify;
        self
    }

    /// Parse an ONNX model from a file path
    ///
    /// When the `mmap` feature is enabled (default), the file is memory-mapped
    /// for zero-copy tensor loading, significantly reducing memory usage.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened or read
    /// - File is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_file(self, path: impl AsRef<Path>) -> Result<OnnxGraph, Error> {
        let path = path.as_ref();
        log::info!("Parsing ONNX file: {}", path.display());

        // Load file contents - mmap when feature is enabled
        #[cfg(feature = "mmap")]
        let buffer = {
            let file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            // SAFETY: We're mapping a read-only file. The bytes::Bytes keeps
            // the mmap alive for as long as tensor data references it.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Memory-mapped ONNX file ({} bytes)", mmap.len());
            bytes::Bytes::from_owner(mmap)
        };

        #[cfg(not(feature = "mmap"))]
        let buffer = {
            let mut file = File::open(path).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).map_err(|error| Error::Io {
                path: path.display().to_string(),
                error,
            })?;
            log::debug!("Read ONNX file into memory ({} bytes)", buf.len());
            bytes::Bytes::from(buf)
        };

        self.parse_buffer(buffer, Some(path))
    }

    /// Parse an ONNX model from a byte slice
    ///
    /// Note: This copies the data internally. For large models already in memory
    /// as `bytes::Bytes`, consider using the internal buffer directly.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_bytes(self, data: &[u8]) -> Result<OnnxGraph, Error> {
        let buffer = bytes::Bytes::copy_from_slice(data);
        self.parse_buffer(buffer, None)
    }

    /// Parse an ONNX model from a reader
    ///
    /// Reads all data into memory before parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Reading from the reader fails
    /// - Data is not valid ONNX protobuf format
    /// - Graph nodes are not topologically sorted
    /// - Type inference fails
    pub fn parse_reader<R: Read>(self, mut reader: R) -> Result<OnnxGraph, Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(|error| Error::Io {
            path: "<reader>".to_string(),
            error,
        })?;
        log::debug!("Read ONNX from reader ({} bytes)", buf.len());
        let buffer = bytes::Bytes::from(buf);
        self.parse_buffer(buffer, None)
    }

    /// Internal: Parse from a bytes::Bytes buffer
    fn parse_buffer(
        self,
        buffer: bytes::Bytes,
        source_path: Option<&Path>,
    ) -> Result<OnnxGraph, Error> {
        let path_str = source_path.map(|p| p.display().to_string());

        // Get the base directory for external data resolution
        let base_path = source_path.and_then(|p| p.parent());

        let model: ModelProto =
            Message::parse_from_tokio_bytes(&buffer).map_err(|e| Error::InvalidFormat {
                path: path_str.clone(),
                error: e.to_string(),
            })?;

        // ONNX nodes must be topologically sorted per spec:
        // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
        if !model.graph.node.is_top_sorted() {
            return Err(Error::InvalidGraphStructure {
                reason: "Nodes are not topologically sorted (ONNX spec violation)".to_string(),
            });
        }

        let graph = build_graph_with_options(&model, base_path, self.simplify)?;

        if let Some(path) = path_str {
            log::info!("Finished parsing ONNX file: {}", path);
        } else {
            log::info!("Finished parsing ONNX from bytes");
        }
        Ok(graph)
    }
}

/// Build IR graph from ONNX model with base path and simplification option
fn build_graph_with_options(
    model: &ModelProto,
    base_path: Option<&Path>,
    simplify: bool,
) -> Result<OnnxGraph, Error> {
    let (opset_version, domain_opsets) = extract_opset_versions(model)?;
    let graph_builder = build_graph_builder_from_proto(
        &model.graph,
        opset_version,
        &domain_opsets,
        None,
        base_path,
        simplify,
    )?;

    log::debug!(" PHASE 6: Node Conversion (RawNode -> Node) ");
    Ok(graph_builder.convert_to_graph(opset_version))
}

/// Build IR graph as OnnxGraphBuilder from ONNX GraphProto
///
/// This returns OnnxGraphBuilder which still contains RawNode instances.
/// Call convert_to_graph() to get the final OnnxGraph with Node enum instances.
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub(crate) fn build_graph_builder_from_proto(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    domain_opsets: &DomainOpsets,
    name_registry: Option<crate::graph_state::NameRegistry>,
    base_path: Option<&Path>,
    simplify: bool,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    build_graph_builder_from_proto_with_outer_scope(
        graph,
        opset_version,
        domain_opsets,
        name_registry,
        crate::ir::OuterScopeTypes::new(),
        base_path,
        simplify,
    )
}

/// Build IR graph as OnnxGraphBuilder with access to outer scope types
///
/// This is used for building subgraphs that reference values from parent graphs.
/// The `outer_scope` map provides types for values that the subgraph references
/// but doesn't define internally.
///
/// The `base_path` is the directory containing the ONNX file, used for resolving
/// external tensor data paths (for models >2GB).
///
/// # Errors
///
/// Returns an error if node conversion or type inference fails
pub(crate) fn build_graph_builder_from_proto_with_outer_scope(
    graph: &crate::protos::GraphProto,
    opset_version: usize,
    domain_opsets: &DomainOpsets,
    name_registry: Option<crate::graph_state::NameRegistry>,
    outer_scope: crate::ir::OuterScopeTypes,
    base_path: Option<&Path>,
    simplify: bool,
) -> Result<crate::ir::OnnxGraphBuilder, Error> {
    log::debug!(" PHASE 1: Initialization ");
    let state_rc = initialization::initialize_from_graph_with_registry_and_outer_scope(
        graph,
        name_registry,
        outer_scope,
        base_path,
    );

    log::debug!(" PHASE 2: Node Conversion (Proto -> RawNode) ");
    node_conversion::convert_nodes_from_graph(graph, &state_rc, opset_version, domain_opsets)?;

    // Fold constant expressions (Slice, Concat, Unsqueeze, etc.) before type inference.
    // Models exported from PyTorch often split initializer weights via Slice+Concat+Unsqueeze
    // chains before feeding them into RNN nodes. Without folding, these remain Dynamic and
    // block type inference for downstream nodes that need constant weight shapes.
    log::debug!(" PHASE 2b: Early Constant Folding ");
    {
        let mut state = state_rc.borrow_mut();
        let mut nodes = std::mem::take(&mut state.processed_nodes);
        drop(state);

        // Run in a fixed-point loop so that Slice -> Concat -> Unsqueeze chains cascade
        let max_iterations = 10;
        let mut converged = false;
        for _ in 0..max_iterations {
            let const_count_before = nodes
                .iter()
                .filter(|n| n.node_type == crate::ir::NodeType::Constant)
                .count();
            nodes = crate::simplify::constant_fold::fold_weight_rearrangements(nodes, &state_rc);
            let const_count_after = nodes
                .iter()
                .filter(|n| n.node_type == crate::ir::NodeType::Constant)
                .count();
            if const_count_after == const_count_before {
                converged = true;
                break;
            }
        }
        if !converged {
            log::debug!(
                "Early constant folding: reached max iterations ({max_iterations}) without converging"
            );
        }

        let mut state = state_rc.borrow_mut();
        state.processed_nodes = nodes;
    }

    log::debug!(" PHASE 3: Type Inference ");
    type_inference::infer_types(&state_rc, opset_version).map_err(Error::TypeInference)?;

    log::debug!(" PHASE 4: Post-processing ");
    let (nodes, inputs, outputs) = post_processing::post_process(&state_rc, simplify);

    let (mut nodes, inputs, mut outputs) = if simplify {
        log::debug!(" PHASE 4b: Simplification ");
        crate::simplify::simplify_graph(nodes, inputs, outputs, &state_rc)
    } else {
        (nodes, inputs, outputs)
    };

    log::debug!(" PHASE 5: Finalization ");
    Ok(finalization::finalize(
        &mut nodes,
        inputs,
        &mut outputs,
        state_rc,
    ))
}

/// Opset version for every domain listed in the model's `opset_import`.
///
/// ONNX operator identity is `(domain, op_type, opset-for-that-domain)`, so
/// custom-domain nodes must be tagged with their own domain's opset, not the
/// default ONNX opset. Wrapped in `Arc` for cheap cloning into `DeferredGraph`
/// (subgraphs inherit the model-level imports). The default domain ("") is
/// always present; `extract_opset_versions` errors otherwise.
#[derive(Debug, Clone)]
pub(crate) struct DomainOpsets {
    versions: Arc<HashMap<String, usize>>,
    default_opset: usize,
}

impl DomainOpsets {
    pub(crate) fn new(versions: HashMap<String, usize>, default_opset: usize) -> Self {
        Self {
            versions: Arc::new(versions),
            default_opset,
        }
    }

    /// Opset version for `domain`, falling back to the default-domain opset.
    ///
    /// Per the ONNX spec every domain a node uses must appear in
    /// `opset_import`; the fallback is robustness against malformed exporters.
    pub(crate) fn opset_for(&self, domain: &str) -> usize {
        if let Some(version) = self.versions.get(domain) {
            return *version;
        }
        log::warn!(
            "Domain '{domain}' has no opset_import entry (malformed model); \
             falling back to default-domain opset {}",
            self.default_opset
        );
        self.default_opset
    }
}

/// Extract opset versions from the model: the default ONNX domain's version
/// plus the per-domain map for custom-domain nodes.
fn extract_opset_versions(model: &ModelProto) -> Result<(usize, DomainOpsets), Error> {
    let default_opset = model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version as usize)
        .ok_or(Error::MissingOpsetVersion)?;

    let versions: HashMap<String, usize> = model
        .opset_import
        .iter()
        .map(|opset| (opset.domain.clone(), opset.version as usize))
        .collect();

    Ok((default_opset, DomainOpsets::new(versions, default_opset)))
}

/// Trait for checking if a list of nodes is topologically sorted
pub(crate) trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

use crate::protos::NodeProto;

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Iterate over each node in the vector
        for (node_position, node) in self.iter().enumerate() {
            // Iterate over each output of the node
            for output in &node.output {
                // If the output is empty, we don't want to check the rest of the graph, inputs and outputs that are optional
                // can end up as empty strings, so we can't use that as a reason to count the graph as not sorted
                if output.is_empty() {
                    continue;
                }
                // Iterate over each other node in the vector
                for (other_node_position, other_node) in self.iter().enumerate() {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if node_position > other_node_position {
                            // The vector is not topologically sorted
                            return false;
                        }
                    }
                }
            }
        }

        // The vector is topologically sorted
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, Node};
    use crate::protos::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TypeProto, ValueInfoProto,
    };
    use crate::protos::{TensorShapeProto, tensor_shape_proto, type_proto};

    fn tensor_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        let mut shape = TensorShapeProto::new();
        for d in dims {
            let mut dim = tensor_shape_proto::Dimension::new();
            dim.set_dim_value(*d);
            shape.dim.push(dim);
        }
        let mut tensor = type_proto::Tensor::new();
        tensor.elem_type = 1; // FLOAT
        tensor.shape = ::protobuf::MessageField::some(shape);
        let mut ty = TypeProto::new();
        ty.set_tensor_type(tensor);
        let mut vi = ValueInfoProto::new();
        vi.name = name.to_string();
        vi.type_ = ::protobuf::MessageField::some(ty);
        vi
    }

    /// A single-node model: input -> <domain>::<op_type> -> output
    fn single_node_model(domain: &str, op_type: &str, opset_imports: &[(&str, i64)]) -> Vec<u8> {
        let mut node = NodeProto::new();
        node.name = "the_node".to_string();
        node.op_type = op_type.to_string();
        node.domain = domain.to_string();
        node.input.push("input".to_string());
        node.output.push("output".to_string());

        let mut graph = GraphProto::new();
        graph.name = "test_graph".to_string();
        graph.input.push(tensor_value_info("input", &[2, 3]));
        graph.output.push(tensor_value_info("output", &[2, 3]));
        graph.node.push(node);

        let mut model = ModelProto::new();
        model.graph = ::protobuf::MessageField::some(graph);
        for (domain, version) in opset_imports {
            let mut op = OperatorSetIdProto::new();
            op.domain = domain.to_string();
            op.version = *version;
            model.opset_import.push(op);
        }
        model.write_to_bytes().unwrap()
    }

    fn parse_single_custom(bytes: &[u8]) -> crate::node::custom::CustomNode {
        let graph = OnnxGraphBuilder::new().parse_bytes(bytes).unwrap();
        let custom = graph
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Custom(c) => Some(c.clone()),
                _ => None,
            })
            .expect("graph should contain a Custom node");
        custom
    }

    #[test]
    fn custom_domain_node_parses_as_custom() {
        let bytes = single_node_model(
            "custom.domain",
            "FftLike",
            &[("", 16), ("custom.domain", 2)],
        );
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "FftLike");
        assert_eq!(custom.domain, "custom.domain");
        // Domain-specific opset, not the default ONNX opset
        assert_eq!(custom.opset, 2);
        // Fallback inference: output type mirrors input
        assert!(matches!(
            &custom.outputs[0].ty,
            ArgType::Tensor(t) if t.dtype == DType::F32 && t.rank == 2
        ));
    }

    #[test]
    fn unknown_default_domain_op_parses_as_custom() {
        let bytes = single_node_model("", "TotallyUnknownOp", &[("", 16)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "TotallyUnknownOp");
        assert_eq!(custom.domain, "");
        assert_eq!(custom.opset, 16);
    }

    #[test]
    fn builtin_name_in_custom_domain_stays_custom() {
        // ONNX identity is (domain, op_type): my.domain::MatMul is NOT the
        // built-in MatMul and must not silently get default-domain semantics.
        let bytes = single_node_model("my.domain", "MatMul", &[("", 16), ("my.domain", 5)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.op_type, "MatMul");
        assert_eq!(custom.domain, "my.domain");
        assert_eq!(custom.opset, 5);
    }

    #[test]
    fn missing_domain_import_falls_back_to_default_opset() {
        // Malformed model: the node's domain has no opset_import entry.
        let bytes = single_node_model("no.import.domain", "SomeOp", &[("", 16)]);
        let custom = parse_single_custom(&bytes);

        assert_eq!(custom.opset, 16);
    }

    #[test]
    fn builtin_op_still_resolves_normally() {
        let bytes = single_node_model("", "Relu", &[("", 16)]);
        let graph = OnnxGraphBuilder::new().parse_bytes(&bytes).unwrap();
        assert!(graph.nodes.iter().any(|n| matches!(n, Node::Relu(_))));
        assert!(!graph.nodes.iter().any(|n| matches!(n, Node::Custom(_))));
    }
}
