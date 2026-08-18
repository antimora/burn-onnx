# burn-onnx roadmap

Prioritized work queue derived from a measured sweep of the open issues and the
`onnx-official-tests` scoreboard on 2026-08-18. Test counts come from re-running every non-passing
entry in `crates/onnx-official-tests/expectations.toml` through `onnx2burn` on `main`, then
compile-checking the output against `burn 0.22.0-pre.1` with the `flex` backend.

## Scoreboard drift

`expectations.toml` has 1615 entries: 722 pass, 484 `skip-codegen`, 230 `skip-compile`, 179
`fail-compare`.

`build.rs` only verifies `pass` and `fail-compare` entries. `skip-codegen`, `skip-compile` and
`flaky` rows are read as documentation and never exercised, so they rot silently. Measured state of
those 714 rows on `main`:

| Claimed            | Measured                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| 230 `skip-compile` | 192 codegen fine; 104 of those also compile clean                                                   |
| 230 `skip-compile` | 38 actually fail codegen (wrong status, not just wrong reason)                                      |
| 484 `skip-codegen` | 37 now codegen fine (33 Mod-Shape, 4 QLinearMatMul)                                                 |
| 15 `skip-codegen`  | reason string stale: they now fail on training-domain ops, not the opset-domain check fixed in #434 |

Roughly 141 rows understate reality. The published pass rate is too low, and every PR that quotes a
delta against this file is quoting a stale baseline.

## Tier 1

### 1. Fix `Size` codegen

`crates/burn-onnx/src/burn/node/size.rs:17` emits:

```rust
let #output = #input.shape.num_elements();
```

`shape` is a method on `Tensor`, not a field, and `num_elements()` returns `usize` while the
generated signature declares `-> i64`. Every model containing an ONNX `Size` node produces Rust that
does not compile.

For a `Shape(N)` input it is worse: the generated code calls `.shape` on an `[i64; N]` array, which
has no such member. The answer there is the compile-time constant `N`.

Why it survived: the inline snapshot at `size.rs:38` asserts the broken output, and while
`tests/size/size.onnx` *is* registered in `crates/onnx-tests/build.rs`, `tests/test_mod.rs` never
declared `pub mod size;`. build.rs therefore generated the model on every build and nothing ever
`include!`d it, so rustc never saw the broken code. It is the only test directory in the repo
missing from `test_mod.rs` (`loop` and `mod` are there as raw idents).

Unblocks 21 official tests: `test_size`, `test_size_example`, and 19 `rms_normalization_*_expanded`.

**Status: done.** `size.rs` now branches on input type: Tensor -> `.shape().num_elements() as i64`,
`Shape(N)` -> the constant `N`, either scalar form -> 1. A new `Size(Shape(N)) -> constant N` rule
in `simplify/constant_shape.rs` folds the pattern away entirely and lets dead-node elimination drop
the feeding `Shape` node. Unlike the Gather and Slice rules it never consults `static_shape`, only
the rank, so it is safe under dynamic dimensions. `tests/size/` is wired up with the existing `size`
model plus a new `size_shape` model. Scoreboard moved 21 rows off `skip-compile`: 8 to `pass`, 13 to
`fail-compare` (see item 5).

### 2. Re-triage the scoreboard

Promote the clean and stale rows, correct the wrong reason strings, and demote the mislabeled
`skip-compile` rows to `skip-codegen`. Add a `cargo xtask retriage` that re-runs every `skip-*` row,
so the file cannot drift again.

**Status: done.** `cargo xtask retriage` runs codegen per row in its own process, promotes what
succeeds, then builds the test crate and demotes anything rustc rejects, carrying the diagnostic as
the row's reason. It converged on all 693 skipped rows in two compile rounds:

| Transition | Rows |
|---|---:|
| skip-compile -> pass | 101 |
| skip-compile -> skip-codegen (mislabeled) | 38 |
| skip-codegen -> skip-compile | 33 |
| skip-codegen -> pass | 4 |

`cargo xtask update-expectations` then demoted the 24 promotions whose output did not match the
reference tensors. Net against `main`:

| Status | Before | After |
|---|---:|---:|
| pass | 722 | 811 |
| fail-compare | 179 | 216 |
| skip-codegen | 484 | 485 |
| skip-compile | 230 | 103 |

Harness tests actually executing and passing went from 663 to 709. The reasons are worth more than
the counts: the single 207-row bucket reading "burn-onnx emits uncompilable generated code
(references alloc::\* from no_std, or emits unresolved variable bindings)" is gone, replaced by
exact rustc diagnostics. Item 8's clusters can now be read straight out of the file — the largest
being 34 rows of `expected Tensor<1, Bool>, found Tensor<1, Int>`, 22 of `expected f32, found
Tensor<0>`, and 22 of `expected f16, found f32`.

Two supporting fixes fell out of the sweep:

- retriage attributes errors in the generated `harness.rs` to the enclosing `fn`, not just errors in
  a generated model. Three `constantofshape` rows compile fine but the driver cannot call them: a
  Shape-typed graph input arrives as `[i64; N]` where the driver built a `Tensor<1, Int>`. Without
  the attribution the sweep aborted with "no error attributed to a promoted model".
- The `fail-compare` harness body in `onnx-official-tests/build.rs` now guards model construction
  with `catch_unwind`. `test_gru_batchwise` panics in `from_file` (the bpk is missing every
  `GateController` weight, which is its own bug worth a ticket), and that panic escaped the
  per-comparison guard and took down `verify_fail_compare_still_fails` for every other entry.

### 3. Upsample (#415)

A user is blocked importing a public model (`fastdepth_7.onnx`). Upsample is deprecated but common
in older exports, and is a strict subset of Resize (opset 7: `scales` attribute; opset 9: `scales`
input; modes nearest/linear). Currently a placeholder in
`crates/onnx-ir/src/node/unsupported.rs:94`.

### 4. Domain-aware unsupported-op error (#433)

`Unknown node type: VariantNotFound` for `TreeEnsembleRegressor` gives the user nothing to act on;
the reporter had to work out on their own that `ai.onnx.ml` is a separate domain.

**Status: already fixed on main**, after the 0.21.0 release the issue was filed against.
`proto_conversion.rs` now maps any unrecognised standard-domain op to `NodeType::Custom` rather
than unwrapping a `FromStr`, and the custom-op coverage check reports it by domain. Checked with a
hand-built `ai.onnx.ml::TreeEnsembleRegressor` model:

```
INFO onnx_ir::proto_conversion: Custom-domain op 'ai.onnx.ml::TreeEnsembleRegressor'
  (node 'tree1'); treating as custom op
...
model contains 1 custom op(s) with no covering inference hook:
  - ai.onnx.ml::TreeEnsembleRegressor used by 1 node(s)
Register hooks via ModelGen::register_custom_op.
```

Remaining work is issue hygiene: confirm against the reporter's attached model, then close #433
against the next release and fold the operator request into #162.

## Tier 2

### 5. Reduce family comparison failures (88 tests)

The largest `fail-compare` bucket, and the worst failure mode to leave sitting: these compile and
run, and produce wrong numbers.

| Family      | Tests |
| ----------- | ----: |
| reduce_sum  |    24 |
| reduce_l1   |    14 |
| reduce_l2   |    14 |
| reduce_log  |     9 |
| reduce_max  |     7 |
| reduce_min  |     7 |
| reduce_prod |     7 |
| reduce_mean |     6 |

Likely a shared root cause in keepdims / empty-axes / `noop_with_empty_axes`.

Item 1 produced a sharp reproducer for part of this. Once the 19 `rms_normalization_*_expanded`
models compile, exactly 6 pass: the `axis0` and `axis_negative_<rank>` variants. Every other axis is
off by a uniform relative factor. The generated code is

```rust
let reducemean1_out1 = { mul1_out1.mean().expand([1; 2usize]) };
```

`mean()` with no argument reduces over *all* elements, so it is only correct when `axis` names the
first dimension and the reduction therefore covers the whole tensor. It should reduce over the axes
from `axis` onward. Those 13 are now tracked as `fail-compare` against #311 and are the cheapest way
into this bucket.

### 6. Runtime weight inputs: LayerNorm (#352, 19 tests) + Conv/ConvTranspose (#346, 12 tests)

Both are the same fix: route through the functional API (`burn::tensor::module::conv2d`) instead of
a baked-in `Param` field. Five ops have now hit this pattern; extract the shared
`runtime_scalar_to_native(arg, target_dtype, scope)` helper in `argument_helpers.rs` proposed in the
#314 thread before doing these two.

### 7. RMSNormalization (19 tests, +19 more via item 1)

Burn has `RmsNorm` natively and ONNX 23 made this a first-class op. High real-world relevance:
Llama, Qwen and Gemma all use it.

### 8. Remaining compile-error clusters

Independent and small, from the compile sweep:

| Tests | Bug                                                                                                                                                |
| ----: | -------------------------------------------------------------------------------------------------------------------------------------------------- |
|     6 | `blackman/hamming/hannwindow_expanded`: `cos`/`powf` on `Tensor<Int>` and `i64 - f32` mixing; missing cast before float math                       |
|     3 | `dynamicquantizelinear_expanded`: `use of moved value`; clone tracking missed, `arg_to_ident` used where `scope.arg` was needed                    |
|     2 | `maxpool_with_argmax`: `cannot find value maxpool2d1_out2`; second output never bound                                                              |
|     2 | `pow_types_int32/int64_float32`: `powf` on Int tensor                                                                                              |
|   ~10 | generated signatures name `half::f16`, but nothing guarantees `half` is a dependency of the consuming crate; this bites real users, not only tests |

## Tier 3

- **#50 / #51 metal backend.** Dozens of ops fail and YOLO11x diverges by 295 max-abs. Correctness
  on the backend people ship on outweighs op-count wins. Root cause probably belongs upstream in
  burn, but it surfaces here.
- **#280 shape propagation through Where/Mul/ConstantOfShape.** Blocks RF-DETR without an `onnxsim`
  pre-pass.
- **#371 Kokoro residual 1.3x.** Established as f32 drift through HiFi-GAN resblocks, not fixable
  here. Close or move to burn.

## Deprioritized

- **NegativeLogLikelihoodLoss (52) + SoftmaxCrossEntropyLoss (34) + nllloss fail-compare (12).** 98
  tests, but training-loss ops in an inference-focused importer. Large count, small user value.
- **#433 TreeEnsembleRegressor / #162 ONNX-ML.** The reporter reached the right conclusion
  themselves. Fix the error message (item 4), then close.
- **Float8 / Float4 / INT4 cast tests (~30).** Blocked on backend dtype support, not on burn-onnx.

## Order

1 -> 2 -> (3 and 4) -> 5 -> 6 -> 7. Items 1-4 are each about a day and independently shippable. Item
2 early, so 5-7 are measurable.

Items 1 and 2 are done, and 4 turned out to be already fixed. Item 3 is next.
