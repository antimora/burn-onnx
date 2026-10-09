#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_shape.onnx
#
# ScatterElements whose data input is a Shape output, as PyTorch exports index_put on size
# vectors. The result is a shape vector with entries replaced, so it stays on the host.
#   s2 = Shape(x) with entry 0 replaced by Shape(y)[0:1], fed to ConstantOfShape -> z
#   s3 = Shape(x) with 2 added to the last entry (negative index, add reduction)
#   s4 = Shape(x) scattered with runtime indices and updates (max reduction)

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18


def const(name, arr):
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(np.asarray(arr))
    )


def main():
    nodes = [
        helper.make_node("Shape", ["x"], ["s"]),
        const("idx", np.array([0], np.int64)),
        helper.make_node("Shape", ["y"], ["sy"]),
        const("st", np.array([0], np.int64)),
        const("en", np.array([1], np.int64)),
        helper.make_node("Slice", ["sy", "st", "en"], ["upd"]),
        helper.make_node("ScatterElements", ["s", "idx", "upd"], ["s2"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["s2"],
            ["z"],
            value=numpy_helper.from_array(np.array([0.0], np.float32)),
        ),
        const("last", np.array([-1], np.int64)),
        const("two", np.array([2], np.int64)),
        helper.make_node(
            "ScatterElements",
            ["s", "last", "two"],
            ["s3"],
            axis=0,
            reduction="add",
        ),
        helper.make_node(
            "ScatterElements",
            ["s", "i", "u"],
            ["s4"],
            axis=0,
            reduction="max",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "scatter_elements_shape",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["a", 3]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, ["b", 5]),
            helper.make_tensor_value_info("i", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("u", TensorProto.INT64, [2]),
        ],
        [
            helper.make_tensor_value_info("z", TensorProto.FLOAT, ["p", "q"]),
            helper.make_tensor_value_info("s3", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("s4", TensorProto.INT64, [2]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_shape.onnx")
    print("Finished exporting model to scatter_elements_shape.onnx")

    x = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros((4, 5), dtype=np.float32)
    i = np.array([0, -1], dtype=np.int64)
    u = np.array([7, 1], dtype=np.int64)
    z, s3, s4 = ReferenceEvaluator(model).run(None, {"x": x, "y": y, "i": i, "u": u})
    print(f"z shape: {z.shape}")
    print(f"s3: {s3}")
    print(f"s4: {s4}")


if __name__ == "__main__":
    main()
