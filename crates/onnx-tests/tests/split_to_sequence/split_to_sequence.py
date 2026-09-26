#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""SplitToSequence read back by SequenceAt at constant positions.

This is how torch.onnx.export lowers `unbind` and `chunk`. Covers each form the
importer rewrites into tensor ops:
- a: no split, keepdims=0: positions 0 and -1
- b: 1-D split [1, 2, 3] on axis 1, keepdims=0 (ignored): positions 1 and -1
  (the latter as int32)
- c: scalar split 4 on axis 1, so chunks are 4 and 2 wide: position 1 (the short one)
- d: no split, keepdims=1 on axis 1: positions 2 and -1
- e: scalar split 1, keepdims=0 (torch's `unbind`): position 1

For e the spec ignores keepdims since split is given, and so does ReferenceEvaluator,
but onnxruntime drops the axis, which is what torch expects. Its expected value is
x[1], not the evaluator's output.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 17


def const(name, value):
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(np.array(value), name)
    )


def main():
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq_a"], axis=0, keepdims=0),
        const("p_a0", np.int64(0)),
        const("p_a1", np.int64(-1)),
        helper.make_node("SequenceAt", ["seq_a", "p_a0"], ["a0"]),
        helper.make_node("SequenceAt", ["seq_a", "p_a1"], ["a1"]),
        const("split_b", np.array([1, 2, 3], dtype=np.int64)),
        helper.make_node(
            "SplitToSequence", ["x", "split_b"], ["seq_b"], axis=1, keepdims=0
        ),
        const("p_b0", np.int64(1)),
        const("p_b1", np.int32(-1)),
        helper.make_node("SequenceAt", ["seq_b", "p_b0"], ["b0"]),
        helper.make_node("SequenceAt", ["seq_b", "p_b1"], ["b1"]),
        const("split_c", np.int64(4)),
        helper.make_node("SplitToSequence", ["x", "split_c"], ["seq_c"], axis=1),
        const("p_c0", np.int64(1)),
        helper.make_node("SequenceAt", ["seq_c", "p_c0"], ["c0"]),
        helper.make_node("SplitToSequence", ["x"], ["seq_d"], axis=1),
        const("p_d0", np.int64(2)),
        const("p_d1", np.int64(-1)),
        helper.make_node("SequenceAt", ["seq_d", "p_d0"], ["d0"]),
        helper.make_node("SequenceAt", ["seq_d", "p_d1"], ["d1"]),
        const("split_e", np.int64(1)),
        helper.make_node(
            "SplitToSequence", ["x", "split_e"], ["seq_e"], axis=0, keepdims=0
        ),
        const("p_e0", np.int64(1)),
        helper.make_node("SequenceAt", ["seq_e", "p_e0"], ["e0"]),
    ]

    output_shapes = {
        "a0": [6],
        "a1": [6],
        "b0": [4, 2],
        "b1": [4, 3],
        "c0": [4, 2],
        "d0": [4, 1],
        "d1": [4, 1],
        # The spec's shape inference says [1, 6] here.
        "e0": [6],
    }
    graph = helper.make_graph(
        nodes,
        "main_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 6])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in output_shapes.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    # No full_check: its shape inference rejects e0 for the reason above.
    onnx.checker.check_model(model)
    onnx.save(model, "split_to_sequence.onnx")

    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    outputs = ReferenceEvaluator(model).run(None, {"x": x})
    outputs[-1] = x[1]
    for name, value in zip(output_shapes, outputs):
        print(f"{name}: {value.tolist()}")


if __name__ == "__main__":
    main()
