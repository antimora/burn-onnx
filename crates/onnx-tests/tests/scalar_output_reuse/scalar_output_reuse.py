#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/scalar_output_reuse/scalar_output_reuse.onnx
#
# A scalar graph output that is also consumed by a later node:
#   x -> ReduceMax -> m (output)
#   m -> Neg -> n (output)

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18


def main():
    reduce_max = helper.make_node("ReduceMax", ["x"], ["m"], keepdims=0)
    neg = helper.make_node("Neg", ["m"], ["n"])

    graph = helper.make_graph(
        [reduce_max, neg],
        "scalar_output_reuse",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [
            helper.make_tensor_value_info("m", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("n", TensorProto.FLOAT, []),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx_name = "scalar_output_reuse.onnx"
    onnx.save(model, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    np.random.seed(42)
    x = np.random.randn(4).astype(np.float32)
    m, n = ReferenceEvaluator(model).run(None, {"x": x})
    print("Test input data: {}".format(x))
    print("Test output data: m={}, n={}".format(m, n))


if __name__ == "__main__":
    main()
