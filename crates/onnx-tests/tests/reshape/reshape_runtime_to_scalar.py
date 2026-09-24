#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Reshape a [1, 1] tensor to a scalar with an empty shape passed as a runtime input
# instead of an initializer.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [0])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])

    node = helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape1")
    graph = helper.make_graph([node], "reshape_runtime_to_scalar", [x, shape], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])
    onnx.checker.check_model(model)
    onnx.save(model, "reshape_runtime_to_scalar.onnx")

    test_x = np.array([[2.5]], dtype=np.float32)
    test_shape = np.array([], dtype=np.int64)
    result = ReferenceEvaluator(model).run(None, {"x": test_x, "shape": test_shape})
    print(f"Input: {test_x}, output: {result[0]} (shape {result[0].shape})")


if __name__ == "__main__":
    main()
