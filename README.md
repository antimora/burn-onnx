<div align="center">

# Burn ONNX

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-onnx.svg)](https://crates.io/crates/burn-onnx)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/burn-onnx)
[![Test Status](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml/badge.svg)](https://github.com/tracel-ai/burn-onnx/actions/workflows/test.yml)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

**Import ONNX models into the [Burn](https://burn.dev) deep learning framework.**

</div>

## Overview

`burn-onnx` converts ONNX models to native Burn code, allowing you to run models from PyTorch,
TensorFlow, and other frameworks on any Burn backend - from WebAssembly to CUDA.

**Key features:**

- Generates readable, modifiable Rust source code from ONNX models
- Produces `burnpack` weight files for efficient loading
- Works with any Burn backend (CPU, GPU, WebGPU, embedded)
- Supports both `std` and `no_std` environments

## Quick Start

Add to your `Cargo.toml`:

```toml
[build-dependencies]
burn-onnx = "0.21"
```

In your `build.rs`:

```rust
use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("model.onnx")
        .out_dir("src/model")
        .run_from_script();
}
```

Then use the generated model:

```rust
mod model;

use burn::backend::Wgpu;
use model::Model;

fn main() {
    let device = Default::default();
    let model: Model<Wgpu> = Model::default();

    let output = model.forward(input);
}
```

For detailed usage instructions, see the
[ONNX Import Guide](https://burn.dev/books/burn/onnx-import.html) in the Burn Book.

## Examples

| Example | Description |
|---------|-------------|
| [onnx-inference](examples/onnx-inference) | Basic ONNX model inference |
| [image-classification-web](examples/image-classification-web) | WebAssembly/WebGPU image classifier |

## Supported Operators

See the [Supported ONNX Operators](SUPPORTED-ONNX-OPS.md) table for the complete list of supported
operators.

## Contributing

We welcome contributions! Please read the [Development Guide](DEVELOPMENT-GUIDE.md) to get started.

For questions and discussions, join us on [Discord](https://discord.gg/uPEBbYYDB6).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
