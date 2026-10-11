#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

# Int64 Clip with a constant min and a runtime max: both bounds must be
# generated with the same Rust type, or it will fail to compile

min_node = helper.make_node(
    "Constant", [], ["min"], value=numpy_helper.from_array(np.array(1, dtype=np.int64))
)
clip_node = helper.make_node("Clip", ["x", "min", "max"], ["y"])
graph = helper.make_graph(
    [min_node, clip_node],
    "main_graph",
    [
        helper.make_tensor_value_info("x", TensorProto.INT64, [3]),
        helper.make_tensor_value_info("max", TensorProto.INT64, []),
    ],
    [helper.make_tensor_value_info("y", TensorProto.INT64, [3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save(model, "clip_int_static_min_runtime_max.onnx")

x = np.array([0, 3, 9], dtype=np.int64)
(y,) = ReferenceEvaluator(model).run(None, {"x": x, "max": np.int64(5)})
print(f"Test output: {y.tolist()}")
