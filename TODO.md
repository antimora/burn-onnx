# burn-onnx roadmap

Prioritized work queue derived from a measured sweep of the open issues and the
`onnx-official-tests` scoreboard on 2026-08-18. Test counts come from re-running every non-passing
entry in `crates/onnx-official-tests/expectations.toml` through `onnx2burn`, then compile-checking
the output against `burn 0.22.0-pre.1` with the `flex` backend.

The `Size` codegen fix and the scoreboard re-triage that produced this baseline landed in #457, and
the domain-aware unsupported-op error (#433) turned out to be already fixed on `main`. Counts below
are the current state of `expectations.toml`, including the Upsample promotion (item 1) and the
runtime-`axes` reduce fix (item 2).

## Scoreboard baseline

`expectations.toml` has 1615 entries:

| Status         | Rows |
| -------------- | ---: |
| `pass`         |  921 |
| `fail-compare` |  107 |
| `skip-codegen` |  484 |
| `skip-compile` |  103 |

819 of the 921 `pass` rows execute as harness tests. The other 102 are codegen-only: build.rs skips
harness generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader cannot construct.

### Why skip counts rot

`build.rs` only verifies `pass` and `fail-compare` entries. `skip-codegen`, `skip-compile` and
`flaky` rows are read as documentation and never exercised, so they go stale the moment someone
fixes the bug behind them, always in the pessimistic direction. Measured on `main` before #457: of
230 claimed `skip-compile` rows, 192 built fine and 101 of those went on to pass, while 38 were not
compile failures at all but codegen failures wearing the wrong status.

`cargo xtask retriage` now re-checks every `skip-*` row, so this cannot silently recur — but nothing
runs it automatically. Run it before trusting a skip count, and after any fix that could plausibly
unblock a family.

### What "pass" does and does not mean

921 rows are marked `pass`; 819 of them execute as harness tests. The other 102 are codegen-only:
`build.rs` skips harness generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb` loader
cannot construct, and `update-expectations` can only demote a row whose test failed. A codegen-only
row is therefore unfalsifiable once promoted, and its output is never compared against the
reference tensors. `retriage` counts them separately when reporting promotions rather than folding
them into the total.

`test_size` and `test_size_example` are in that group (the Size fix is verified by the
`crates/onnx-tests/tests/size/` integration tests, not by the official suite), as are 26
`test_castlike_*` rows converting to FLOAT8/INT4 variants. Extending the harness to cover them is
separate work; the honest reading of 921 is "921 compile, 819 match".

Item 2 turned four of those unfalsifiable rows into real tests and immediately found a bug in two
of them, which is the concrete cost of the category: `test_reduce_log_sum_exp_do_not_keepdims_*`
were marked `pass` while producing a rank-0 output the harness declines to drive. Correcting the
inferred rank gave them a driver, and the driver failed.

## Tier 1

### 1. Upsample (#415)

A user is blocked importing a public model (`fastdepth_7.onnx`). Upsample is deprecated but common
in older exports, and is a strict subset of Resize (opset 7: `scales` attribute; opset 9: `scales`
input; modes nearest/linear).

**Status: done.** ONNX deprecated Upsample *into* Resize: Resize opset 10 is Upsample with the
coordinate mapping (`asymmetric`) and nearest rounding (`floor`) spelled out as attributes. So
`node/upsample.rs` extracts Upsample's own attributes across all three shapes it has had -
`height_scale`/`width_scale` (opset 1), the `scales` float-list attribute (opset 7), the `scales`
input (opset 9+, static or runtime) - and builds a `ResizeNode` with those two modes pinned. The
enum entry is `Upsample => resize::ResizeNode` and burn-onnx just lists `Upsample` in its dispatch
macro, so there is no second copy of the interpolate codegen. Node naming still comes from the
`RawNode`, so generated code reads `self.upsample1`, not `resize1`.

Beyond what Resize does today, the processor computes the output static shape as
`floor(dim * scale)` rather than copying the input's, rejects scaling the batch or channel
dimension instead of silently dropping those scales, and rejects ranks Burn's interpolate cannot
serve (anything but 3 and 4, or anything but 4 when the scales are a runtime input) rather than
emitting code that references a field that was never created.

A multi-agent review caught a wrong claim in the first draft of this work, which said nearest mode
was exact and only linear diverged. Both halves were wrong, and the corrected behavior is the more
interesting part of the change:

- **`mode="linear"` is refused**, not warned about. Burn's bilinear samples at half-pixel
  coordinates (ONNX's `half_pixel`); Upsample mandates `asymmetric`. Every interior sample differs
  at every scale other than 1, so this is "always wrong", not "may differ". A `log::warn!` was also
  the wrong channel: cargo swallows build-script stdout unless the line is `cargo:warning=`
  prefixed, so the primary `build.rs` path showed the user nothing at all.
- **Nearest is refused when a scale does not divide its dimension evenly.** ONNX picks a source
  element by scale, `floor(o / scale)`; Burn's kernel picks it by output size,
  `floor(o * in / out)` with `out = floor(in * scale)`. Those agree only when `in * scale` is
  whole. Verified end to end: `scale=1.75` on width 5 gives Burn `[0,0,1,1,2,3,3,4]` against
  onnxruntime's `[0,0,1,1,2,2,3,4]`. Where the product is provable the model is rejected with the
  dimension, the scale and the reason; where it is not (runtime scales, dynamic dims) it warns.
  Integer scales, which is what fastdepth and most real exports use, are unaffected.

Every test in the first draft used an integer scale, which is exactly the case where the two
formulas coincide, so the suite could not have caught this. `ReferenceEvaluator` cannot either: its
Upsample is `np.repeat` and raises on non-integer scales, making onnxruntime the only usable oracle.

Two smaller review fixes: opset 1 spells linear mode `bilinear` (the rename came at opset 7), which
was being rejected as an unknown mode rather than for the real reason; and spatial scales are now
checked against the spec's "greater than or equal to 1", since a scale below 1 or a NaN reaches
`as usize` in generated code and saturates to a zero-size dimension.

Scoreboard: `test_upsample_nearest` moved from `skip-codegen` to `pass`. Opset compliance grew from
472 to 476 op-version combinations (Upsample at opsets 1, 7, 9, 10).

## Tier 2

### 2. Reduce family comparison failures (99 tests)

**Status: done (#459).** 96 of the 99 rows shared one root cause, and it was the one #459 named. Opset 18
moved `axes` from an attribute to an input; when that input is a graph input rather than a constant
its value is unknown at build time, and `ReduceConfig::dims` was a plain `Vec<usize>` that recorded
that case as an empty vector — the same value ONNX uses for "no axes given, reduce everything".
Codegen then emitted `.sum()` / `.mean()` with no dimension argument and dropped the axes input on
the floor. Every one of those models supplies `axes` this way; none of them was a keepdims or
broadcasting bug. The other 3 are a separate bug, described at the end of this item.

The fix is to make the three meanings of "empty axes" distinguishable, which they are not in a
`Vec`:

| ONNX                                  | `ReduceConfig::axes`  | Behavior                |
| ------------------------------------- | --------------------- | ----------------------- |
| `axes` absent, or an empty list       | `Static(vec![])`      | reduce every axis       |
| empty list with `noop_with_empty_axes`| `Static(vec![])`      | skip the reduction      |
| `axes` supplied at run time           | `Runtime(input_ref)`  | reduce what it names    |

`ReduceConfig` now carries `axes: ReduceAxes`, the `Static(..) | Runtime(RuntimeInputRef)` enum that
22 other node files already use, plus the `noop_with_empty_axes` attribute it was previously
discarding. "Skip the reduction" is not quite "identity": the spec says other operations still
happen, so `ReduceSumSquare` still squares and `ReduceL1` still takes an absolute value.

Reducing over axes that are not compile-time constants sounds like it should be impossible against
Burn's statically-ranked tensors, and it very nearly is — but only the output *rank* has to be
static, not the axis values. Burn 0.22's `sum_dims`/`mean_dims`/`max_dims`/`min_dims`/`prod_dims`
and `squeeze_dims::<D2>` all take `&[impl AsIndex]`, a runtime slice, and `AsIndex::try_dim_index`
wraps negative entries itself, so a runtime axis and a negative axis both come out correct with no
work in the generated code. The rank comes from two places: with `keepdims=1` it is the input rank,
and with `keepdims=0` it is `input_rank - len(axes)`, where the *length* is in the axes input's
static shape even when its values are not. All 99 models declare that length. When they do not
(a `Range`-computed axes list) and `keepdims` is off, onnx-ir now refuses the model instead of
guessing.

Generated code for a runtime-axes reduce reads the axes once and hands the slice to Burn:

```rust
let __axes: alloc::vec::Vec<i64> = axes.into_data().iter::<i64>().collect();
data.abs().sum_dims(&__axes).squeeze_dims::<2usize>(&__axes)
```

Two things surfaced while doing it that were not in the original diagnosis:

- **LogSumExp was wrong for `keepdims=0` independently of the axes bug.** It subtracts a running
  max from the input, so that max has to keep its rank to broadcast back; squeezing it first made
  `expand(input_shape)` either wrong or a hard `Squeeze` panic. Both intermediate reductions now run
  with keepdims and the reduced axes are dropped once at the end. This was invisible before because
  the two affected rows inferred a rank-0 output, which `build.rs` declines to generate a driver for
  — they were marked `pass` and never ran.
- **Out-of-range axes were accepted.** `dim as usize` on a negative that did not wrap left a huge
  index; `extract_config` now returns `ProcessError::InvalidAttribute` naming the axis and the rank.

Scoreboard: 109 rows promoted from `fail-compare` to `pass` — 96 of the 99 reduce rows and all 13
remaining `rms_normalization_*_expanded` rows, which used `Shape -> Size -> Range` to build their
axes at run time. Harness tests went from 706 to 819, all green.

A multi-agent review after the fact turned up two more silent-wrong-answer cases in the first
draft of this work, both now fixed and both worth recording because they are the same shape as the
bug being fixed:

- **`noop_with_empty_axes` was lowered to a bare identity.** The spec says the reduction is skipped
  but "other operations will be performed", so `ReduceSumSquare` must still square, `ReduceL1` and
  `ReduceL2` must still take an absolute value, and `ReduceLogSum` must still take a log. Only the
  five plain reductions are genuine identities. Modelling this as "reduce over no axes" rather than
  an early return makes each composite land on the right answer through machinery that already
  exists - `ReduceL2` becomes `sqrt(square(x))`, `ReduceLogSumExp` becomes `x + log(exp(x - x))`.
  The reductions that *are* identities now implement `NodeProcessor::is_noop`, so the framework
  drops those nodes in post-processing instead of codegen emitting a rebinding; the codegen path
  stays for `simplify(false)`, where only Identity is eliminated.
- **A runtime axes list that is empty at run time.** Burn's `*_dims` fold over an empty slice is the
  identity, but ONNX reads empty axes as "every dimension" unless `noop_with_empty_axes` is set.
  Only reachable when the axes input has no statically known length, which is exactly the case the
  opset 18 input shape exists for. The generated code now resolves the list where its length is
  finally known.

Two structural validations were added alongside: an axis count larger than the input rank used to
underflow `tensor_rank - axis_count` and panic with no node name, and duplicate axes built cleanly
and then panicked inside Burn's `squeeze_dims`, which deduplicates and so disagreed with the rank
onnx-ir had declared.

Note for the `build_node` item in Tier 3: Reduce is now the second operator, after Upsample, with
validation that can first become reachable in `build_node` (an out-of-range axis behind a
`Constant -> Identity -> Reduce` chain). Its panic at least names the node and formats the error
with `Display` now, but the underlying hazard is unchanged.

The 3 reduce rows left are `test_reduce_max_empty_set`, `test_reduce_min_empty_set` and
`test_reduce_log_sum_exp_empty_set`, which are a different bug: reducing over a zero-size dimension,
where ONNX mandates the identity element (`-inf` for max, `+inf` for min) and Burn's kernels return
something else. Sum, prod, L1, L2 and LogSum over an empty set all pass, because their identity
elements are 0 and 1 and Burn agrees. This belongs upstream in burn, not here.

Also fixed in the same pass, since it was blocking clean `retriage` runs: **#460**, non-deterministic
attribute-validation errors. `Attributes` was a `HashMap`, whose iteration order Rust reseeds per
process, so a model with two rejected attributes reported whichever one the loop happened to reach
first. It is now a `BTreeMap`; the type change is one line and the fallout was the 5 construction
sites the issue predicted plus their test helpers. `test_resize_downsample_sizes_nearest_not_smaller`
reported `axes` on 8 of 8 runs afterwards, against 1-of-6 before.

### 3. Runtime weight inputs: LayerNorm (#352, 19 tests) + Conv/ConvTranspose (#346, 12 tests)

Both are the same fix: route through the functional API (`burn::tensor::module::conv2d`) instead of
a baked-in `Param` field. Five ops have now hit this pattern; extract the shared
`runtime_scalar_to_native(arg, target_dtype, scope)` helper in `argument_helpers.rs` proposed in the
#314 thread before doing these two.

### 4. RMSNormalization (19 tests)

Burn has `RmsNorm` natively and ONNX 23 made this a first-class op. High real-world relevance:
Llama, Qwen and Gemma all use it. The 19 `_expanded` rows this item used to also claim now pass on
the decomposition alone (item 2); the 19 left are the native op, still `skip-codegen`.

### 5. Remaining compile-error clusters

All 103 remaining `skip-compile` rows carry the rustc diagnostic that produced them, so this table
is a `grep` of `expectations.toml` rather than an estimate. Sorted by blast radius:

| Rows | Diagnostic | Example | Read |
| ---: | --- | --- | --- |
| 34 | `expected Tensor<1, Bool>, found Tensor<1, Int>` | `test_attention_3d_attn_mask_expanded` | the Mod/And-on-Shape chain lands an Int tensor where a mask is wanted. Biggest single win left in the bucket, and it is the attention-expanded family. |
| 22 | `expected f32, found Tensor<0>` | `test_cast_FLOAT8E4M3FN_to_FLOAT` | rank-0 output typed as a scalar in the signature but produced as a tensor. |
| 22 | `expected f16, found f32` | `test_cast_FLOAT8E4M3FN_to_FLOAT16` | the f16 cast result is never narrowed. Same family as the row above; likely one fix for both 44. |
| 3 | `no method named add found for type f32` | `test_blackmanwindow_expanded` | scalar/tensor mixing in the window ops. |
| 3 | `expected Tensor<1>, found f32` | `test_hammingwindow_symmetric_expanded` | same family. |
| 3 | `use of moved value: div1_out1` | `test_dynamicquantizelinear_expanded` | clone tracking missed; `arg_to_ident` used where `scope.arg` was needed. |
| 2 | `expected bool, found Tensor<0, Bool>` | `test_equal_string` | rank-0 bool, same shape as the f32 case above. |
| 2 | `Tensor<3>: ElementConversion is not satisfied` | `test_gelu_default_2_expanded` | a tensor passed to a scalar-taking API. |
| 2 | `cannot find value maxpool2d1_out2` | `test_maxpool_with_argmax_2d_precomputed_pads` | MaxPool's second output is never bound. |
| 2 | `expected Tensor<1>, found Tensor<1, Int>` | `test_pow_types_float32_int32` | missing cast before a binary op. |
| 2 | `no method named powf on Tensor<Int>` | `test_pow_types_int32_float32` | Pow with an int base needs a cast first. |
| 3 | `expected [i64; N], found Tensor<1, Int>` | `test_constantofshape_float_ones` | not a codegen bug: the model compiles, the generated *harness* cannot call it, because a Shape-typed graph input arrives as `[i64; N]` where the driver built a `Tensor<1, Int>`. Fixing it means teaching build.rs to read the generated `forward` signature rather than inferring argument types from the ONNX proto. |
| 1 each | `fmod` on `Tensor<Int>`, `expected Tensor<4>, found Tensor<4, Int>`, `can't compare f32 with {integer}` | `test_mod_int64_fmod` | one-offs. |

The `half::f16` problem noted during the first sweep is not in this table: generated signatures do
name `half::f16`, but `onnx-official-tests` happens to depend on `half` already. It still bites a
consuming crate that does not, and is worth fixing independently of these rows.

### 6. GRU/LSTM/RNN discard runtime weights (#458)

Surfaced by the re-triage, previously hidden behind a `skip-compile` row. `test_gru_batchwise`
compiles, then panics in `Model::from_file`:

```
Validation error: Missing tensors: [
  ("gru1.new_gate.hidden_transform.weight", "Struct:Model.Struct:Gru.Struct:GateController.Struct:Linear"),
  ("gru1.new_gate.input_transform.weight",  ...),
  ("gru1.reset_gate.hidden_transform.weight", ...),
  ("gru1.reset_gate.input_transform.weight",  ...),
  ("gru1.update_gate.hidden_transform.weight", ...),
  ("gru1.update_gate.input_transform.weight",  ...),
]
```

Root-caused while filing #458, and it is worse than "unloadable". Every RNN-family test in the
upstream suite supplies `W`/`R` as runtime graph inputs rather than initializers.
`collect_gru_snapshots` returns an empty snapshot list when the weights are not statically available
(`gru.rs:52`, `:55`; same shape in `lstm.rs:87`/`:90` and `rnn.rs:82`/`:85`), but `field()` still
emits the module. So the generated `forward` accepts `w` and `r` as parameters and never reads them:

```rust
pub fn forward(&self, x: Tensor<3>, w: Tensor<3>, r: Tensor<3>) -> Tensor<3> {
    let gru_output = self.gru1.forward(x.swap_dims(0, 1), None);
    //                    ^^^^ w and r are dropped on the floor
```

`from_file` panics on the missing tensors, but `Model::new` does not: `GruConfig::init` gives the
module fully random weights and inference proceeds. That silent path is the reason this is a bug
rather than a gap. Same family as item 3's Conv/LayerNorm runtime weights (#346, #352), except
those reject the model with a clear error instead of accepting it and computing nonsense.

## Tier 3

- **#50 / #51 metal backend.** Dozens of ops fail and YOLO11x diverges by 295 max-abs. Correctness
  on the backend people ship on outweighs op-count wins. Root cause probably belongs upstream in
  burn, but it surfaces here.
- **Resize shares every gap Upsample just closed (surfaced by the item 1 review).** None of it is
  new and none of it blocked item 1, but it is all in `crates/burn-onnx/src/import/burn/node/resize.rs`
  and its processor:
  - Accepts `asymmetric` linear and computes half-pixel values (#311 already tracks
    `test_resize_upsample_scales_cubic_asymmetric` as `fail-compare`).
  - Does not validate that a nearest scale divides its dimension, so it has the same silent pixel
    shift Upsample now refuses.
  - The runtime path emits `input_dims[3]` unconditionally, so a rank-3 runtime-scales Resize emits
    code that does not compile.
  - Drops `scales[0]`/`scales[1]` in the runtime path, so a batch or channel scale that the static
    path hard-rejects is silently ignored when the same tensor arrives as a graph input.
  - Leaves `nearest_mode` at the opset 11 default on its opset 10 path, where the spec says floor.

  Note `coordinate_transformation_mode` and `nearest_mode` are recorded but never read by codegen
  (only `align_corners` is derived), so pinning them documents intent without changing behavior.
- **`build_node` cannot report an error, so late-lifted constants panic.** `lift_constants` runs
  again after identity elimination (`post_processing.rs:265`) and type inference does not re-run
  after it, so `Constant -> Identity -> Op` reaches `build_node` with a value that was Dynamic
  during `infer_types`. Any validation that first becomes possible there can only panic:
  `NodeProcessor::build_node` returns `Node`, not `Result<Node>`. Every processor in the crate has
  this shape (`.expect("Config extraction failed")`); Upsample is just the first to have checks
  that can realistically fire there. Reproduced with a `Constant -> Identity -> Upsample` graph
  carrying scales of 1.75.
- **#280 shape propagation through Where/Mul/ConstantOfShape.** Blocks RF-DETR without an `onnxsim`
  pre-pass.
- **#371 Kokoro residual 1.3x.** Established as f32 drift through HiFi-GAN resblocks, not fixable
  here. Close or move to burn.

## Deprioritized

- **NegativeLogLikelihoodLoss (52) + SoftmaxCrossEntropyLoss (34) + nllloss fail-compare (12).** 98
  tests, but training-loss ops in an inference-focused importer. Large count, small user value.
- **#433 TreeEnsembleRegressor / #162 ONNX-ML.** The reporter reached the right conclusion
  themselves, and the error message they hit is already fixed on `main`: `proto_conversion.rs` maps
  any unrecognised standard-domain op to `NodeType::Custom` instead of unwrapping a `FromStr`, and
  the custom-op coverage check reports it by domain. Remaining work is issue hygiene: confirm
  against the reporter's attached model, close #433 against the next release, and fold the operator
  request into #162.
- **Float8 / Float4 / INT4 cast tests (~30).** Blocked on backend dtype support, not on burn-onnx.

## Order

Items 1 and 2 are done, and #460 landed alongside item 2. Item 6 (#458) is next: it is the last
known silent-wrong-answer bug on the board, and the argument that pulled item 2 ahead of the
test-count work applies to it unchanged — a GRU/LSTM/RNN whose runtime `W`/`R` are dropped still
runs, with fully random weights, under `Model::new`.

Then 3 -> 4, each measurable against an honest baseline. Item 3 shares item 6's shape (runtime
weights that codegen assumes are static), so doing 6 first may well produce the helper 3 needs.

Item 5's top three rows (34 + 22 + 22 = 78 of the 103 remaining `skip-compile` rows) are probably
two fixes, which makes that bucket the largest remaining test-count win now that item 2 is spent.
