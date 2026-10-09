#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: name_collision.onnx
#
# Distinct ONNX names that sanitize to the same Rust identifier must stay distinct values:
# "/c/INT64/[-1]" and "/c/INT64/[1]" both sanitize to "c_int64_1", and so do the node
# outputs "t:0" and "t/0" (to "t_0").

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 17


def main():
    initializers = [
        numpy_helper.from_array(np.array([-1], np.int64), "/c/INT64/[-1]"),
        numpy_helper.from_array(np.array([1], np.int64), "/c/INT64/[1]"),
    ]
    nodes = [
        helper.make_node("Unsqueeze", ["x", "/c/INT64/[-1]"], ["y"]),
        helper.make_node("Unsqueeze", ["x", "/c/INT64/[1]"], ["z"]),
        helper.make_node("Neg", ["x"], ["t:0"]),
        helper.make_node("Abs", ["x"], ["t/0"]),
        helper.make_node("Sub", ["t:0", "t/0"], ["w"]),
    ]
    graph = helper.make_graph(
        nodes,
        "name_collision",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 1]),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 1, 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 3]),
        ],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "name_collision.onnx")
    print("Finished exporting model to name_collision.onnx")

    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    y, z, w = ReferenceEvaluator(model).run(None, {"x": x})
    print(f"y shape: {y.shape}")
    print(f"z shape: {z.shape}")
    print(f"w: {w}")


if __name__ == "__main__":
    main()
