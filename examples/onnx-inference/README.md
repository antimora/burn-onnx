# ONNX Inference

The smallest complete ONNX import: a PyTorch-trained MNIST classifier converted to Burn at build
time and run on a test image. `build.rs` turns `src/model/mnist.onnx` into Rust source plus a `.bpk`
weights file, and the binary loads both and classifies a digit.

## Usage

```sh
cargo run -- 15
```

The argument is an index into the MNIST test set (0 to 9999; defaults to 42):

```text
Image index: 15
Success!
Predicted: 5
Actual: 5
See the image online, click the link below:
https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row=15
```

The MNIST test set is downloaded on first run.

## How to import a model

These are the steps this crate follows, and the same ones apply to your own model.

1. Add the dependencies to `Cargo.toml`. The generated code loads weights through `burn-store`, so
   it is a regular dependency next to `burn`:

   ```toml
   [dependencies]
   burn = { version = "0.22", features = ["flex"] }
   burn-store = "0.22"

   [build-dependencies]
   burn-onnx = "0.22"
   ```

   This example also enables `burn`'s `dataset` and `vision` features to load MNIST.

2. Put the ONNX file in `src/model/mnist.onnx`.

3. Generate the code from `build.rs`:

   ```rust
   use burn_onnx::ModelGen;

   fn main() {
       ModelGen::new()
           .input("src/model/mnist.onnx")
           .out_dir("model/")
           .run_from_script();
   }
   ```

4. Include the generated file from `src/model/mod.rs`:

   ```rust
   pub mod mnist {
       include!(concat!(env!("OUT_DIR"), "/model/mnist.rs"));
   }
   ```

5. Expose the module from `src/lib.rs`:

   ```rust
   pub mod model;

   pub use model::mnist::*;
   ```

6. Use the model, as in [`src/bin/mnist_inference.rs`](src/bin/mnist_inference.rs):

   ```rust
   use burn::tensor::{Device, Tensor};
   use onnx_inference::mnist::Model;

   fn main() {
       let device: Device = Default::default();

       // Load the weights that build.rs wrote next to the generated code.
       let model: Model = Model::default();

       let input = Tensor::<4>::zeros([1, 1, 28, 28], &device);
       let output = model.forward(input);
       println!("{output}");
   }
   ```

7. `cargo build` generates the code and weights, then compiles everything. The generated file lands
   in `target/debug/build/onnx-inference-*/out/model/mnist.rs` if you want to read it.

## Re-exporting the model from PyTorch

`pytorch/mnist.py` trains the network and exports it to ONNX. Its dependencies are declared inline,
so `uv` installs them automatically:

```sh
cd pytorch && uv run mnist.py
```

This writes `mnist.onnx` to the current directory; copy it over `src/model/mnist.onnx` to use it.

## Resources

- [Burn Book: ONNX Import](https://burn.dev/books/burn/onnx-import.html)
- [Exporting a PyTorch model to ONNX](https://pytorch.org/docs/stable/onnx.html)
- [ONNX introduction](https://onnx.ai/onnx/intro/)
