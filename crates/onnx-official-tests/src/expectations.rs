//! Parser for `expectations.toml` — the declarative source of truth for
//! how each upstream node test is expected to behave.
//!
//! The schema intentionally over-allocates statuses for M1 (only `pass`
//! is exercised) so that follow-up PRs widening coverage do not have to
//! revisit the parser shape.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Top-level expectations file. Each map key is the upstream test name
/// (e.g. `test_abs`); each value declares the expected outcome.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct Expectations {
    pub entries: BTreeMap<String, Entry>,
}

/// One row of `expectations.toml`. Only `status` is required.
#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    pub status: Status,
    /// Free-form explanation of why the test is in this state.
    #[serde(default)]
    pub reason: Option<String>,
    /// Linked tracking issue or PR (e.g. `#314`).
    #[serde(default)]
    pub tracking: Option<String>,
    /// `true` means we will not fix this (out of scope, upstream-only
    /// dtype, etc.). Defaults to `false` — i.e. an intentional gap we
    /// plan to close.
    #[serde(default)]
    pub wontfix: bool,
}

/// Declared expected outcome of a single upstream test.
///
/// `Pass` is the only status the M1 runner actually exercises. The
/// remaining variants are parsed but treated as "skip this test for
/// now"; M2 will wire them through to the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Codegen succeeds, compile succeeds, output matches reference.
    Pass,
    /// `onnx2burn` panics or refuses the model (unsupported op, etc.).
    SkipCodegen,
    /// Codegen succeeds but the generated Rust does not compile.
    SkipCompile,
    /// Compiles and runs but produces incorrect output.
    FailCompare,
    /// Intermittent — do not gate CI on it.
    Flaky,
}

impl Expectations {
    /// Parse an `expectations.toml` file from disk.
    pub fn load(path: &Path) -> Result<Self, ExpectationsError> {
        let text = std::fs::read_to_string(path)?;
        let parsed: Self = toml::from_str(&text)?;
        Ok(parsed)
    }

    /// Look up the expected outcome for a given test name.
    pub fn get(&self, test_name: &str) -> Option<&Entry> {
        self.entries.get(test_name)
    }
}

#[derive(Debug)]
pub enum ExpectationsError {
    Io(std::io::Error),
    Parse(toml::de::Error),
}

impl std::fmt::Display for ExpectationsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error reading expectations.toml: {e}"),
            Self::Parse(e) => write!(f, "toml parse error: {e}"),
        }
    }
}

impl std::error::Error for ExpectationsError {}

impl From<std::io::Error> for ExpectationsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<toml::de::Error> for ExpectationsError {
    fn from(e: toml::de::Error) -> Self {
        Self::Parse(e)
    }
}
