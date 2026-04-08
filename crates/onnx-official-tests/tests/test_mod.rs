//! Integration tests that run the upstream ONNX backend node tests
//! vendored under `crates/onnx-official-tests/vendor/node/`.
//!
//! Each `#[test]` in this file corresponds to one upstream test
//! directory. The test:
//!
//!   1. Constructs the burn-onnx-generated `Model` for that test.
//!   2. Loads `test_data_set_0/input_*.pb` into burn `Tensor`s.
//!   3. Calls `model.forward(...)`.
//!   4. Loads `test_data_set_0/output_0.pb` and compares with
//!      `assert_approx_eq` at `Tolerance::default()`.
//!
//! The set of tests run here is mirrored in `expectations.toml` and the
//! `TESTS` array in `build.rs`. The runtime helper at the bottom of this
//! file (`verify_expectations_match_tests`) asserts they stay in sync,
//! so adding a vendored test only in `build.rs` (or only in
//! `expectations.toml`) will cause a clear test failure rather than
//! silent drift.

#![allow(clippy::approx_constant)]

mod backend;

use std::path::PathBuf;

use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};
use onnx_official_tests::expectations::{Expectations, Status};
use onnx_official_tests::pb_loader::{FloatTensor, load_float_tensor};

use crate::backend::TestBackend;

type FT = FloatElem<TestBackend>;

/// Pull in every generated `Model` module that `build.rs` produced under
/// `$OUT_DIR/model/`. The list here must stay in lockstep with the
/// `TESTS` array in `build.rs` and the entries in `expectations.toml`;
/// the `verify_expectations_match_tests` test enforces that.
macro_rules! include_node_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[allow(clippy::type_complexity, non_snake_case)]
            pub mod $name {
                include!(concat!(
                    env!("OUT_DIR"),
                    concat!("/model/", stringify!($name), ".rs"),
                ));
            }
        )*

        const VENDORED_TESTS: &[&str] = &[$(stringify!($name)),*];
    };
}

include_node_tests!(
    test_abs,
    test_add,
    test_ceil,
    test_cos,
    test_div,
    test_exp,
    test_floor,
    test_log,
    test_mul,
    test_neg,
    test_pow,
    test_reciprocal,
    test_relu,
    test_round,
    test_sigmoid,
    test_sin,
    test_softplus,
    test_softsign,
    test_sqrt,
    test_sub,
    test_tanh,
);

/// Path to a vendored test directory, relative to the crate root.
fn vendor_dir(test_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vendor/node")
        .join(test_name)
}

/// Load `input_<idx>.pb` from a vendored test as an `f32` tensor of the
/// expected rank `D`. The rank is provided by the caller (per test) so
/// the model's `forward()` signature stays statically typed; a runtime
/// rank mismatch is a bug in the macro invocation, not in the data.
fn load_input<const D: usize>(test_name: &str, idx: usize) -> Tensor<TestBackend, D> {
    let path = vendor_dir(test_name)
        .join("test_data_set_0")
        .join(format!("input_{idx}.pb"));
    let raw =
        load_float_tensor(&path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    assert_eq!(
        raw.shape.len(),
        D,
        "rank mismatch for {test_name} input_{idx}: \
         macro says {D}, .pb has {} (shape {:?})",
        raw.shape.len(),
        raw.shape,
    );
    let data = TensorData::new(raw.values, raw.shape);
    Tensor::<TestBackend, D>::from_data(data, &Default::default())
}

/// Load `output_0.pb` from a vendored test as a `TensorData`. Burn's
/// `assert_approx_eq` consumes a `TensorData`, so we don't need to wrap
/// it in a `Tensor` first.
fn load_expected_output(test_name: &str) -> TensorData {
    let path = vendor_dir(test_name)
        .join("test_data_set_0")
        .join("output_0.pb");
    let FloatTensor { shape, values } =
        load_float_tensor(&path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    TensorData::new(values, shape)
}

/// Compare a model output to its `output_0.pb` reference, using burn's
/// default float tolerance.
fn assert_matches_reference<const D: usize>(test_name: &str, output: Tensor<TestBackend, D>) {
    let expected = load_expected_output(test_name);
    output
        .to_data()
        .assert_approx_eq::<FT>(&expected, Tolerance::default());
}

/// Sanity-check that on-disk expectations agree with the compiled-in
/// list. This catches the easy mistakes (forgot to update one or the
/// other) before they cause confusing failures elsewhere.
#[test]
fn verify_expectations_match_tests() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("expectations.toml");
    let expectations =
        Expectations::load(&path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));

    let declared: std::collections::BTreeSet<&str> =
        expectations.entries.keys().map(String::as_str).collect();
    let compiled: std::collections::BTreeSet<&str> = VENDORED_TESTS.iter().copied().collect();

    let only_in_compiled: Vec<&&str> = compiled.difference(&declared).collect();
    let only_in_declared: Vec<&&str> = declared.difference(&compiled).collect();

    assert!(
        only_in_compiled.is_empty() && only_in_declared.is_empty(),
        "expectations.toml and the include_node_tests! list have drifted.\n  \
         Only in include_node_tests!: {only_in_compiled:?}\n  \
         Only in expectations.toml:  {only_in_declared:?}"
    );

    // M1 currently only handles `pass`. If a future PR marks something
    // differently before wiring up the corresponding harness path, fail
    // loudly so the gap is obvious.
    for (name, entry) in &expectations.entries {
        assert_eq!(
            entry.status,
            Status::Pass,
            "expectations.toml entry `{name}` has status {:?}, but the \
             M1 runner only handles `pass`. Add the harness branch \
             before introducing this status.",
            entry.status,
        );
    }
}

// ----------------------------------------------------------------------
// Per-test cases. Each one mirrors the upstream test name verbatim so
// failures point straight at the corresponding `vendor/node/<name>/`
// directory.
//
// All M1 tests share two shapes:
//   - unary  : forward(Tensor<B, 3>)               -> Tensor<B, 3>
//   - binary : forward(Tensor<B, 3>, Tensor<B, 3>) -> Tensor<B, 3>
//
// We keep the bodies inline (instead of factoring through a trait or
// macro) so that the model module being exercised is unambiguous in the
// test source, which matters when triaging a regression.
// ----------------------------------------------------------------------

macro_rules! unary_node_test {
    ($name:ident, $rank:literal) => {
        #[test]
        fn $name() {
            let device = Default::default();
            let model = $name::Model::<TestBackend>::new(&device);
            let input = load_input::<$rank>(stringify!($name), 0);
            let output = model.forward(input);
            assert_matches_reference::<$rank>(stringify!($name), output);
        }
    };
}

macro_rules! binary_node_test {
    ($name:ident, $rank:literal) => {
        #[test]
        fn $name() {
            let device = Default::default();
            let model = $name::Model::<TestBackend>::new(&device);
            let lhs = load_input::<$rank>(stringify!($name), 0);
            let rhs = load_input::<$rank>(stringify!($name), 1);
            let output = model.forward(lhs, rhs);
            assert_matches_reference::<$rank>(stringify!($name), output);
        }
    };
}

// Per-test ranks below are dictated by the upstream `model.onnx` graph
// I/O shapes — most use [3, 4, 5] (rank 3), but `test_round` uses a 1D
// input. If we add a test with a different rank, the macro invocation's
// const-generic argument is the only thing to update.

unary_node_test!(test_abs, 3);
unary_node_test!(test_ceil, 3);
unary_node_test!(test_cos, 3);
unary_node_test!(test_exp, 3);
unary_node_test!(test_floor, 3);
unary_node_test!(test_log, 3);
unary_node_test!(test_neg, 3);
unary_node_test!(test_reciprocal, 3);
unary_node_test!(test_relu, 3);
unary_node_test!(test_round, 1);
unary_node_test!(test_sigmoid, 3);
unary_node_test!(test_sin, 3);
unary_node_test!(test_softplus, 3);
unary_node_test!(test_softsign, 3);
unary_node_test!(test_sqrt, 3);
unary_node_test!(test_tanh, 3);

binary_node_test!(test_add, 3);
binary_node_test!(test_div, 3);
binary_node_test!(test_mul, 3);
binary_node_test!(test_pow, 3);
binary_node_test!(test_sub, 3);
