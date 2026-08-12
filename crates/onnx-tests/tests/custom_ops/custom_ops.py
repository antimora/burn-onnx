#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: custom_ops.onnx
#
# The model chains three custom (non-built-in) operators whose semantics are
# supplied by CustomOp hooks registered in onnx-tests/build.rs:
#   x -> ScaleShift(test.custom, scale=2.0, shift=0.5)
#     -> AddWindow(test.custom, window constant initializer)
#     -> MyIdentity(default domain)
#     -> y
#
# There is no ONNX reference implementation for these ops; the ground truth is
# computed with numpy below and asserted in the Rust test with the same input.

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    scale, shift = 2.0, 0.5
    window = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    scale_shift = helper.make_node(
        "ScaleShift",
        inputs=["x"],
        outputs=["y1"],
        name="scale_shift1",
        domain="test.custom",
        scale=scale,
        shift=shift,
    )
    add_window = helper.make_node(
        "AddWindow",
        inputs=["y1", "window"],
        outputs=["y2"],
        name="add_window1",
        domain="test.custom",
    )
    identity = helper.make_node(
        "MyIdentity",
        inputs=["y2"],
        outputs=["y"],
        name="my_identity1",
        domain="",
    )

    graph = helper.make_graph(
        [scale_shift, add_window, identity],
        "custom_ops_graph",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[
            helper.make_tensor(
                "window", TensorProto.FLOAT, window.shape, window.tolist()
            )
        ],
    )

    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[
            helper.make_operatorsetid("", 16),
            helper.make_operatorsetid("test.custom", 1),
        ],
    )
    # No check_model: the checker rejects unknown op_types in the default
    # domain (MyIdentity), which is exactly the parser-tolerance case this
    # fixture exists to exercise.

    file_name = "custom_ops.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Ground truth for the Rust test input
    x = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
    )
    expected = (x * scale + shift) + window
    print(f"Test input: {x.tolist()}")
    print(f"Expected output: {expected.tolist()}")


if __name__ == "__main__":
    main()
