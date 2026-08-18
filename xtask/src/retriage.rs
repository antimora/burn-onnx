//! `cargo xtask retriage` — re-run the `skip-codegen` and `skip-compile`
//! rows of `expectations.toml` against the current tree and rewrite
//! them to match reality.
//!
//! `crates/onnx-official-tests/build.rs` only exercises `pass` and
//! `fail-compare` rows. Every `skip-*` row is read as documentation and
//! never tried, so those rows drift the moment someone fixes the bug
//! that put them there. The drift is one-directional and invisible: the
//! file always claims the tree is worse than it is.
//!
//! `update-expectations` covers the other direction — demoting `pass`
//! rows that started failing — and leaves promotion to manual edits, on
//! the grounds that trying every skipped row is prohibitively expensive.
//! It isn't: codegen over the whole skip set is a few hundred process
//! spawns and finishes in well under a minute.
//!
//! The command runs in stages, each feeding the next:
//!
//! 1. **Codegen.** Run `onnx2burn` on every selected row. Rows that
//!    still fail keep `skip-codegen` and get a freshly captured reason
//!    (a `skip-compile` row that fails here was mislabeled). Rows that
//!    succeed are promoted to `pass` optimistically.
//! 2. **Compile.** Build the test crate. Any rustc error inside a
//!    generated model demotes that row to `skip-compile`, carrying the
//!    rustc diagnostic as its reason. Repeated until the crate builds,
//!    because one broken model can mask errors in later ones.
//! 3. **Hand off.** Whatever survives is a `pass` claim about output
//!    correctness, which `update-expectations` is already built to
//!    check.
//!
//! `--dry-run` reports stage 1 without touching the file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use tracel_xtask::prelude::*;

use crate::expectations_schema::{Expectations, Status};

/// Maximum number of build attempts in stage 2. Each round removes at
/// least one broken model from the crate, so this only bounds the
/// pathological case where rustc surfaces one model at a time.
const MAX_COMPILE_ROUNDS: usize = 12;

/// Longest reason string written back to the file.
const MAX_REASON_LEN: usize = 220;

/// Arguments for the `retriage` subcommand.
#[derive(clap::Args)]
pub struct RetriageArgs {
    /// Report the planned rewrite without modifying any file. Only
    /// stage 1 runs: stage 2 needs the promotions on disk to build them.
    #[arg(long)]
    pub dry_run: bool,

    /// Stop after stage 1. Rows whose codegen succeeded stay marked
    /// `pass` with no compile check, so the file may claim more than the
    /// tree delivers. Useful for looking at codegen drift on its own.
    #[arg(long)]
    pub codegen_only: bool,

    /// Re-check only the first N eligible rows, in file order, for
    /// sampling the drift without paying for the full sweep.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Optional `tracking` value to embed in every rewritten row, e.g.
    /// `"#456"`. Omitted by default: a captured reason is
    /// self-documenting, and a stale issue reference is worse than none.
    #[arg(long)]
    pub tracking: Option<String>,
}

/// What the sweep decided about one row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Verdict {
    status: Status,
    reason: Option<String>,
}

pub fn handle_command(args: RetriageArgs) -> anyhow::Result<()> {
    let repo_root = repo_root();
    let crate_dir = repo_root.join("crates/onnx-official-tests");
    let expectations_path = crate_dir.join("expectations.toml");

    let original = std::fs::read_to_string(&expectations_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", expectations_path.display()))?;
    let parsed = Expectations::from_toml(expectations_path.clone(), &original)
        .map_err(|e| anyhow::anyhow!("parse expectations: {e}"))?;

    let mut eligible: Vec<String> = parsed
        .entries
        .iter()
        .filter(|(_, e)| matches!(e.status, Status::SkipCodegen | Status::SkipCompile))
        .filter(|(_, e)| !e.wontfix)
        .map(|(name, _)| name.clone())
        .collect();
    if let Some(limit) = args.limit {
        eligible.truncate(limit);
    }

    if eligible.is_empty() {
        info!("No skip-codegen or skip-compile rows to re-check.");
        return Ok(());
    }
    info!("Re-checking {} skipped row(s)", eligible.len());

    // --- Stage 1: codegen ------------------------------------------
    let onnx2burn = build_onnx2burn(&repo_root)?;
    let scratch = repo_root.join("target/retriage");
    let _ = std::fs::remove_dir_all(&scratch);

    let mut verdicts: BTreeMap<String, Verdict> = BTreeMap::new();
    let mut codegen_ok: Vec<String> = Vec::new();
    for name in &eligible {
        let model = crate_dir.join("vendor/node").join(name).join("model.onnx");
        if !model.is_file() {
            warn!("{name}: no vendored model.onnx, leaving the row alone");
            continue;
        }
        match run_codegen(&onnx2burn, &model, &scratch.join(name)) {
            Ok(()) => codegen_ok.push(name.clone()),
            Err(reason) => {
                verdicts.insert(
                    name.clone(),
                    Verdict {
                        status: Status::SkipCodegen,
                        reason: Some(reason),
                    },
                );
            }
        }
    }
    info!(
        "Codegen: {} succeeded, {} still fail",
        codegen_ok.len(),
        verdicts.len()
    );

    for name in &codegen_ok {
        verdicts.insert(
            name.clone(),
            Verdict {
                status: Status::Pass,
                reason: None,
            },
        );
    }

    if args.dry_run {
        report(&parsed, &verdicts);
        warn!("--dry-run set; no files were modified");
        if !args.codegen_only {
            warn!("stage 2 (compile) needs the promotions on disk, so it did not run");
        }
        return Ok(());
    }

    let mut text = apply_verdicts(&original, &verdicts, args.tracking.as_deref());
    std::fs::write(&expectations_path, &text)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", expectations_path.display()))?;

    // --- Stage 2: compile ------------------------------------------
    if !args.codegen_only && !codegen_ok.is_empty() {
        let mut remaining: BTreeSet<String> = codegen_ok.iter().cloned().collect();
        for round in 1..=MAX_COMPILE_ROUNDS {
            let broken = compile_and_collect_broken_models(&remaining)?;
            if broken.is_empty() {
                info!("Compile: clean after {round} round(s)");
                break;
            }
            info!(
                "Compile round {round}: demoting {} model(s) to skip-compile",
                broken.len()
            );

            let round_verdicts: BTreeMap<String, Verdict> = broken
                .into_iter()
                .map(|(name, diag)| {
                    (
                        name,
                        Verdict {
                            status: Status::SkipCompile,
                            reason: Some(diag),
                        },
                    )
                })
                .collect();
            for (name, verdict) in &round_verdicts {
                remaining.remove(name);
                verdicts.insert(name.clone(), verdict.clone());
            }

            text = apply_verdicts(&text, &round_verdicts, args.tracking.as_deref());
            std::fs::write(&expectations_path, &text)
                .map_err(|e| anyhow::anyhow!("write {}: {e}", expectations_path.display()))?;

            if round == MAX_COMPILE_ROUNDS {
                warn!(
                    "still not building after {MAX_COMPILE_ROUNDS} round(s); \
                     re-run to keep narrowing"
                );
            }
        }
    }

    report(&parsed, &verdicts);
    info!("Rewrote {}", expectations_path.display());
    info!(
        "Next: `cargo xtask update-expectations` to demote any promoted row whose \
         output does not match the reference tensors."
    );
    Ok(())
}

/// Build the codegen binary once and return its path.
fn build_onnx2burn(repo_root: &Path) -> anyhow::Result<PathBuf> {
    info!("Building onnx2burn...");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "-p",
            "burn-onnx",
            "--bin",
            "onnx2burn",
        ])
        .current_dir(repo_root)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn cargo: {e}"))?;
    if !status.success() {
        return Err(anyhow::anyhow!("building onnx2burn failed"));
    }
    Ok(repo_root.join("target/release/onnx2burn"))
}

/// Run codegen for one model, returning the extracted failure reason on
/// error. A separate process per model means a codegen panic costs an
/// exit code rather than the whole sweep.
fn run_codegen(onnx2burn: &Path, model: &Path, out_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create out dir: {e}"))?;
    let output = Command::new(onnx2burn)
        .arg(model)
        .arg(out_dir)
        .env("RUST_LOG", "error")
        .output()
        .map_err(|e| format!("spawn onnx2burn: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(extract_panic_reason(&String::from_utf8_lossy(
            &output.stderr,
        )))
    }
}

/// Pull the operator-level complaint out of an onnx2burn panic.
///
/// The message sits on the line after `panicked at <loc>:`, wrapped in
/// framing that is identical for every failure. The framing is stripped
/// so reasons stay comparable across rows and close in shape to the
/// hand-written ones already in the file.
fn extract_panic_reason(stderr: &str) -> String {
    let stripped = strip_ansi(stderr);
    let lines: Vec<&str> = stripped.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("panicked at"))
        .map(|i| i + 1)
        .unwrap_or(0);

    // Take the message body: up to the backtrace note or a blank line.
    // Multi-line messages (custom-op coverage lists) are joined so the
    // reason stays a single TOML string.
    let body: Vec<&str> = lines
        .get(start..)
        .unwrap_or(&[])
        .iter()
        .map(|l| l.trim())
        .take_while(|l| !l.is_empty() && !l.starts_with("note:"))
        .collect();

    let joined = body.join(" ");
    let mut msg = joined.trim();

    // `Failed to parse ONNX file '<path>': ` names the vendored path,
    // which differs per row and would make otherwise-identical reasons
    // look distinct.
    if let Some(rest) = msg
        .strip_prefix("Failed to parse ONNX file ")
        .and_then(|r| r.split_once("': "))
        .map(|(_, rest)| rest)
    {
        msg = rest;
    }
    msg = msg.strip_prefix("Type inference failed: ").unwrap_or(msg);

    if msg.is_empty() {
        return "codegen failed with no message on stderr".to_string();
    }
    truncate_reason(msg)
}

/// Drop ANSI colour codes so they never reach the TOML file.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Consume through the final byte of the escape sequence.
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Keep reasons to one readable line and escape what TOML cares about.
fn truncate_reason(msg: &str) -> String {
    let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
    if escaped.chars().count() > MAX_REASON_LEN {
        escaped.chars().take(MAX_REASON_LEN - 3).collect::<String>() + "..."
    } else {
        escaped
    }
}

/// Build the test crate and map rustc errors back to the models that
/// produced them.
///
/// Generated models live at `$OUT_DIR/model/<test_name>.rs`, so the
/// diagnostic's file name identifies the row. Only names in
/// `candidates` are reported: an error inside a model that was already
/// passing is a real regression and must not be quietly reclassified.
fn compile_and_collect_broken_models(
    candidates: &BTreeSet<String>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let output = Command::new("cargo")
        .args([
            "build",
            "-p",
            "onnx-official-tests",
            "--tests",
            "--message-format=short",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn cargo: {e}"))?;
    if output.status.success() {
        return Ok(BTreeMap::new());
    }

    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    let mut broken: BTreeMap<String, String> = BTreeMap::new();
    for line in stderr.lines() {
        let Some((path, message)) = parse_short_diagnostic(line) else {
            continue;
        };
        let Some(name) = model_name_from_path(path) else {
            continue;
        };
        if candidates.contains(&name) {
            broken
                .entry(name)
                .or_insert_with(|| truncate_reason(message));
        }
    }

    if broken.is_empty() {
        return Err(anyhow::anyhow!(
            "onnx-official-tests failed to build but no error was attributed to a \
             promoted model; fix the build manually and re-run"
        ));
    }
    Ok(broken)
}

/// Split one `--message-format=short` error line into its path and
/// message. The format is `file:line:col: error[CODE]: message`;
/// warnings and notes are ignored.
fn parse_short_diagnostic(line: &str) -> Option<(&str, &str)> {
    let (path, rest) = line.split_once(".rs:")?;
    let path_end = path.len() + 3;
    let after_span = rest.split_once(": ")?.1;
    let message = after_span.strip_prefix("error")?;
    // Drop an optional `[E0433]` code and the separating colon.
    let message = match message.split_once(": ") {
        Some((_, tail)) => tail,
        None => return None,
    };
    Some((&line[..path_end], message.trim()))
}

/// `.../out/model/test_foo.rs` -> `test_foo`.
fn model_name_from_path(path: &str) -> Option<String> {
    let (_, tail) = path.rsplit_once("/model/")?;
    tail.strip_suffix(".rs").map(str::to_string)
}

/// Rewrite the matching rows in place, preserving every other line.
fn apply_verdicts(
    original: &str,
    verdicts: &BTreeMap<String, Verdict>,
    tracking: Option<&str>,
) -> String {
    let mut out = String::with_capacity(original.len() + verdicts.len() * 160);
    let mut lines = original.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(name) = parse_header(line)
            && let Some(verdict) = verdicts.get(name)
        {
            // Drop the old body so the replacement is clean. The blank
            // separator line is left in place by the peek guard.
            while let Some(&peek) = lines.peek() {
                let trimmed = peek.trim();
                if trimmed.is_empty() || trimmed.starts_with('[') {
                    break;
                }
                lines.next();
            }
            out.push_str(&format!("[{name}]\n"));
            out.push_str(&format!("status = \"{}\"\n", verdict.status.as_str()));
            if let Some(reason) = &verdict.reason {
                out.push_str(&format!("reason = \"{reason}\"\n"));
            }
            if let Some(tracking) = tracking {
                out.push_str(&format!("tracking = \"{tracking}\"\n"));
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `[test_name]` -> `test_name`, ignoring anything else.
fn parse_header(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .filter(|name| !name.is_empty() && !name.contains(['[', ']', '.']))
}

/// Summarise the sweep as a from -> to tally plus the promotion list.
fn report(before: &Expectations, verdicts: &BTreeMap<String, Verdict>) {
    let mut transitions: BTreeMap<(Status, Status), usize> = BTreeMap::new();
    let mut promoted: Vec<&str> = Vec::new();
    for (name, verdict) in verdicts {
        let Some(entry) = before.entries.get(name) else {
            continue;
        };
        *transitions
            .entry((entry.status, verdict.status))
            .or_default() += 1;
        if verdict.status == Status::Pass {
            promoted.push(name);
        }
    }

    info!("Re-triage summary:");
    for ((from, to), count) in &transitions {
        let marker = if from == to { "  " } else { "->" };
        info!(
            "  {marker} {:>4}  {} -> {}",
            count,
            from.as_str(),
            to.as_str()
        );
    }
    if !promoted.is_empty() {
        info!("Promoted to pass ({}):", promoted.len());
        for name in &promoted {
            info!("  - {name}");
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level under the repo root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The common single-line case: framing stripped, operator
    /// complaint kept.
    #[test]
    fn extract_reason_strips_framing() {
        let stderr = "\
thread 'main' (123) panicked at crates/burn-onnx/src/model_gen.rs:397:33:
Failed to parse ONNX file 'vendor/node/test_x/model.onnx': Type inference failed: Node 'resize1' (Resize): Invalid attribute 'axes': custom axes attribute is not supported
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
";
        assert_eq!(
            extract_panic_reason(stderr),
            "Node 'resize1' (Resize): Invalid attribute 'axes': custom axes attribute is not supported"
        );
    }

    /// Multi-line panic bodies are joined into one TOML-safe string.
    #[test]
    fn extract_reason_joins_multiline_body() {
        let stderr = "\
thread 'main' (123) panicked at crates/burn-onnx/src/model_gen.rs:397:33:
Failed to parse ONNX file 'vendor/node/test_adagrad/model.onnx': model contains 1 custom op(s) with no covering inference hook:
  - ai.onnx.preview.training::Adagrad used by 1 node(s)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
";
        let reason = extract_panic_reason(stderr);
        assert!(
            reason.starts_with("model contains 1 custom op(s)"),
            "{reason}"
        );
        assert!(reason.contains("Adagrad used by 1 node(s)"), "{reason}");
        assert!(!reason.contains('\n'));
    }

    /// Colour codes from the logger must not reach the file.
    #[test]
    fn extract_reason_strips_ansi() {
        let stderr =
            "panicked at x.rs:1:1:\n\u{1b}[31mERROR\u{1b}[0m something went wrong\nnote: bt\n";
        let reason = extract_panic_reason(stderr);
        assert_eq!(reason, "ERROR something went wrong");
    }

    /// A panic-free failure still yields a usable reason rather than an
    /// empty string, which would produce `reason = ""`.
    #[test]
    fn extract_reason_handles_empty_stderr() {
        assert_eq!(
            extract_panic_reason(""),
            "codegen failed with no message on stderr"
        );
    }

    /// Quotes and backslashes are escaped so the rewritten row parses.
    #[test]
    fn reasons_are_toml_escaped() {
        let escaped = truncate_reason(r#"expected "Tensor", got C:\path"#);
        assert_eq!(escaped, r#"expected \"Tensor\", got C:\\path"#);
    }

    #[test]
    fn long_reasons_are_truncated() {
        let reason = truncate_reason(&"x".repeat(500));
        assert_eq!(reason.chars().count(), MAX_REASON_LEN);
        assert!(reason.ends_with("..."));
    }

    #[test]
    fn short_diagnostics_parse() {
        let line =
            "/t/out/model/test_size.rs:60:27: error[E0609]: no field `shape` on type `[i64; 4]`";
        let (path, message) = parse_short_diagnostic(line).unwrap();
        assert_eq!(path, "/t/out/model/test_size.rs");
        assert_eq!(message, "no field `shape` on type `[i64; 4]`");
        assert_eq!(model_name_from_path(path).unwrap(), "test_size");
    }

    /// Warnings share the line shape but must not demote anything.
    #[test]
    fn short_diagnostics_ignore_warnings() {
        let line = "/t/out/model/test_size.rs:60:27: warning: unused variable: `x`";
        assert!(parse_short_diagnostic(line).is_none());
    }

    /// Errors outside the generated-model directory belong to the
    /// harness, not to a row, and are left unattributed.
    #[test]
    fn diagnostics_outside_model_dir_are_unattributed() {
        assert!(model_name_from_path("/t/src/lib.rs").is_none());
    }

    #[test]
    fn apply_verdicts_rewrites_only_named_rows() {
        let original = "\
# leading comment
[test_a]
status = \"skip-compile\"
reason = \"stale\"
tracking = \"#314\"

[test_b]
status = \"pass\"
";
        let verdicts = BTreeMap::from([(
            "test_a".to_string(),
            Verdict {
                status: Status::Pass,
                reason: None,
            },
        )]);
        let out = apply_verdicts(original, &verdicts, None);
        assert!(out.contains("# leading comment"));
        assert!(out.contains("[test_a]\nstatus = \"pass\"\n"));
        // The stale reason and tracking are gone with the old body.
        assert!(!out.contains("stale"));
        assert!(!out.contains("#314"));
        // Untouched rows survive verbatim.
        assert!(out.contains("[test_b]\nstatus = \"pass\""));
    }

    /// The rewritten file must still parse, including reasons that
    /// contain quotes.
    #[test]
    fn rewritten_file_reparses() {
        let original = "[test_a]\nstatus = \"skip-compile\"\n";
        let verdicts = BTreeMap::from([(
            "test_a".to_string(),
            Verdict {
                status: Status::SkipCodegen,
                reason: Some(truncate_reason(r#"expected "Tensor for split sizes""#)),
            },
        )]);
        let out = apply_verdicts(original, &verdicts, Some("#456"));
        let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &out).unwrap();
        let entry = &parsed.entries["test_a"];
        assert_eq!(entry.status, Status::SkipCodegen);
        assert_eq!(
            entry.reason.as_deref(),
            Some(r#"expected "Tensor for split sizes""#)
        );
        assert_eq!(entry.tracking.as_deref(), Some("#456"));
    }

    #[test]
    fn headers_are_recognised() {
        assert_eq!(parse_header("[test_abs]"), Some("test_abs"));
        assert_eq!(parse_header("status = \"pass\""), None);
        assert_eq!(parse_header("[[array]]"), None);
    }
}
