#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Gemm at opset 6 in the Linear pattern (alpha=1, beta=1, transB=1, constant B and C).

Opsets before 7 carry a `broadcast` attribute on Gemm. onnx-ir fuses this Gemm into
Linear, which must not trip over the leftover attribute.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)
    weight = np.random.randn(4, 3).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)

    node = helper.make_node(
        "Gemm",
        ["input", "weight", "bias"],
        ["output"],
        alpha=1.0,
        beta=1.0,
        transB=1,
        broadcast=1,
    )
    graph = helper.make_graph(
        [node],
        "gemm_linear_opset6",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        initializer=[
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 6)], ir_version=4
    )
    onnx.checker.check_model(model)
    onnx.save(model, "gemm_linear_opset6.onnx")

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    (output,) = ReferenceEvaluator(model).run(None, {"input": x})
    print(f"Input: {x.tolist()}")
    print(f"Output: {output.tolist()}")


if __name__ == "__main__":
    main()
