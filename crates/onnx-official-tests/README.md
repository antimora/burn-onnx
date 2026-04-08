# onnx-official-tests

Native regression gate against the upstream
[ONNX backend test suite](https://github.com/onnx/onnx/tree/main/onnx/backend/test).

This is the M1 scaffold for issue #315. It runs a small, hand-picked set
of upstream node tests against `burn-onnx`-generated Rust code on every
`cargo test`, with no Docker, no Python, and no network.

```sh
cargo test -p onnx-official-tests
```

## What's in here

```text
crates/onnx-official-tests/
├── Cargo.toml
├── build.rs                # stages each vendored model.onnx and runs ModelGen
├── expectations.toml       # declarative source of truth for each test
├── vendor/node/            # upstream test data, vendored from onnx v1.19.0
│   └── test_<name>/
│       ├── model.onnx
│       └── test_data_set_0/{input_*.pb, output_0.pb}
├── src/
│   ├── lib.rs
│   ├── expectations.rs     # parse expectations.toml
│   └── pb_loader.rs        # decode TensorProto .pb files into f32 buffers
└── tests/
    ├── backend.rs          # selects burn backend by feature flag
    └── test_mod.rs         # one #[test] per vendored test, plus a drift check
```

## Vendored test data

The data under `vendor/node/` is copied verbatim from
[`onnx/onnx`](https://github.com/onnx/onnx) at tag `v1.19.0`. Twenty-one
test directories are included for the M1 scaffold:

- Unary float ops: `test_abs`, `test_ceil`, `test_cos`, `test_exp`,
  `test_floor`, `test_log`, `test_neg`, `test_reciprocal`, `test_relu`,
  `test_round`, `test_sigmoid`, `test_sin`, `test_softplus`,
  `test_softsign`, `test_sqrt`, `test_tanh`
- Binary float ops: `test_add`, `test_div`, `test_mul`, `test_pow`,
  `test_sub`

To refresh or extend, fetch the matching `onnx/onnx` release tarball,
copy `onnx/backend/test/data/node/test_<name>/` into `vendor/node/`,
add the test name to:

1. `TESTS` in `build.rs`
2. `include_node_tests!` in `tests/test_mod.rs` (with the right rank
   for the macro invocation)
3. An entry in `expectations.toml`

The `verify_expectations_match_tests` integration test fails loudly if
the three lists drift. A future PR will replace these manual steps with
`cargo xtask refresh-onnx-tests` (M2 in #315).

## Expectations

`expectations.toml` is the declarative source of truth. Each entry sets
a `status` for one upstream test. M1 only exercises `pass`, but the
parser already understands the full set so M2 can wire them through
without revisiting the schema:

| status         | meaning                                                       |
|----------------|---------------------------------------------------------------|
| `pass`         | codegen + compile + output match the reference                |
| `skip-codegen` | onnx2burn refuses or panics on this model (M2+)               |
| `skip-compile` | codegen succeeds but the generated Rust does not compile (M2+)|
| `fail-compare` | compiles and runs, but produces incorrect output (M2+)        |
| `flaky`        | intermittent, ignored by the gate (M2+)                       |

Optional fields: `reason`, `tracking` (issue/PR ref), `wontfix` (bool).

## What this crate is not

- Not a replacement for `crates/onnx-tests/` — those hand-crafted tests
  exercise targeted regressions and stay the canonical place for "did
  this specific bug come back?".
- Not a replacement for `scoreboard/` — that submits to the upstream
  ONNX Backend Scoreboard via Docker. This crate is the local CI gate.
- Not yet wired up in CI — that's M3 in #315.

## Roadmap

Tracked in [#315](https://github.com/tracel-ai/burn-onnx/issues/315):

- **M1** *(this PR)*: scaffold + ~20 known-passing tests + drift check.
- **M2**: vendor the full ~1800 upstream tests, populate expectations
  with `skip-*` entries from the #314 analysis, replace the per-test
  macros with a data-driven harness.
- **M3**: GitHub Actions matrix workflow.
- **M4**: PR-comment delta diff against `main`.
- **M5**: `--update-expectations` accept-new-greens local workflow.
