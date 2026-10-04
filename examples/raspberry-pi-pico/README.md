# ONNX Inference on the Raspberry Pi Pico 2

Runs an imported ONNX model on a microcontroller: a Cortex-M33 with 520 KB of RAM, no operating
system, and no `std`. The model is a tiny TensorFlow network that approximates `y = sin(x)`; the
firmware sweeps `x` from 0 to 2 and logs each prediction over USB serial.

What makes it work:

- **`no_std` generated code.** The code `burn-onnx` generates only needs `alloc`, so it compiles
  for bare-metal targets.
- **Embedded weights.** `build.rs` uses `LoadStrategy::Embedded`, which compiles the weights into
  the firmware image; `Model::from_embedded(&device)` reads them in place, with no filesystem.
- **Flex backend.** Burn's pure-Rust CPU backend, built with `critical-section` for targets without
  atomic pointer support.
- **A small heap.** `embedded-alloc` provides a 100 KB heap for tensor allocations. Adjust
  `HEAP_SIZE` in `src/bin/main.rs` for larger models.

## Setup

1. Install the Pico 2 target:

   ```sh
   rustup target add thumbv8m.main-none-eabihf
   ```

2. Install a flashing tool. Either `elf2flash` (the default runner, flashes over USB in BOOTSEL
   mode):

   ```sh
   cargo install elf2flash
   ```

   or [`probe-rs`](https://probe.rs/docs/getting-started/installation/) with a
   [compatible debug probe](https://probe.rs/docs/getting-started/probe-setup/). To use `probe-rs`,
   uncomment its `runner` line in `.cargo/config.toml` and comment out the `elf2flash` one.

## Running

With the Pico connected (hold BOOTSEL while plugging it in when using `elf2flash`):

```sh
cargo run --release
```

The onboard LED turns on once the firmware starts, and each prediction is logged over USB serial
as `input: <x> - output: [<y>]`, with `y` tracking `sin(x)`.

For the original Raspberry Pi Pico (RP2040), switch the `embassy-rp` feature from `rp235xb` to
`rp2040` in `Cargo.toml`, set the build target to `thumbv6m-none-eabi` in `.cargo/config.toml`
(the line is already there, commented out), and replace `memory.x` with the RP2040 layout (264 KB
of RAM, plus a `BOOT2` section). The RP2040 build has not been tested.

## Project structure

```text
raspberry-pi-pico
├── build.rs            # places memory.x and generates the model with LoadStrategy::Embedded
├── memory.x            # flash and RAM layout of the RP2350
├── src
│   ├── bin/main.rs     # firmware: heap, USB logger, inference loop
│   ├── lib.rs
│   └── model
│       ├── mod.rs      # include!s the generated sine.rs
│       └── sine.onnx
└── tensorflow
    └── train.py        # trains the model and exports it to src/model/sine.onnx
```

## Retraining the model

`tensorflow/train.py` declares its dependencies inline, so `uv` can run it directly. It writes to a
path relative to the `tensorflow` directory, so run it from there:

```sh
cd tensorflow && uv run train.py
```

It trains the network and writes `src/model/sine.onnx`. The next `cargo build` regenerates the Rust
code from it.
