//! Build script: stage every vendored upstream node test into a uniquely
//! named `.onnx` file and run `burn_onnx::ModelGen` over the staged copies.
//!
//! Why staging? Each upstream test directory ships its model as
//! `vendor/node/test_<name>/model.onnx`. `ModelGen` derives the output
//! file name from the input file's stem, so handing it 21 paths all
//! named `model.onnx` would emit 21 colliding `model.rs` files. Copying
//! each one to `$OUT_DIR/staged/test_<name>.onnx` first gives every
//! generated module a unique name that matches its upstream test.

use burn_onnx::ModelGen;
use std::fs;
use std::path::PathBuf;

/// Upstream node tests vendored under `vendor/node/`. The list lives in
/// the build script (rather than being scanned dynamically) so adding or
/// removing a test is a deliberate, reviewable change.
const TESTS: &[&str] = &[
    "test_abs",
    "test_add",
    "test_ceil",
    "test_cos",
    "test_div",
    "test_exp",
    "test_floor",
    "test_log",
    "test_mul",
    "test_neg",
    "test_pow",
    "test_reciprocal",
    "test_relu",
    "test_round",
    "test_sigmoid",
    "test_sin",
    "test_softplus",
    "test_softsign",
    "test_sqrt",
    "test_sub",
    "test_tanh",
];

fn main() {
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=expectations.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let staged = out_dir.join("staged");
    fs::create_dir_all(&staged).expect("create staged dir");

    let mut model_gen = ModelGen::new();

    for name in TESTS {
        let src: PathBuf = format!("vendor/node/{name}/model.onnx").into();
        assert!(src.exists(), "vendored model missing: {}", src.display());
        let dst = staged.join(format!("{name}.onnx"));
        fs::copy(&src, &dst)
            .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", src.display(), dst.display()));
        model_gen.input(dst.to_str().expect("non-utf8 staged path"));
    }

    model_gen.out_dir("model/").run_from_script();
}
