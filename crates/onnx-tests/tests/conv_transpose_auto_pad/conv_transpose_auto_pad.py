#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: conv_transpose_auto_pad.onnx
#
# ConvTranspose with constant weights whose pads come from auto_pad or output_shape:
#   y1: 1D SAME_UPPER, the odd pad lands at the end and is cropped off
#   y2: 2D SAME_LOWER with dilation, the odd pad lands at the start
#   y3: 3D SAME_UPPER with output_shape and output_padding
#   y4: 2D output_shape larger than the full result, grown at the end
# The reference evaluator is the ground truth; ONNX Runtime must agree with it.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def seq(shape, scale):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale - 1.0


def main():
    f = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("x1", f, [1, 1, 3]),
        helper.make_tensor_value_info("x2", f, [1, 1, 2, 3]),
        helper.make_tensor_value_info("x3", f, [1, 1, 2, 2, 2]),
        helper.make_tensor_value_info("x4", f, [1, 1, 2, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("y1", f, [1, 2, 6]),
        helper.make_tensor_value_info("y2", f, [1, 1, 4, 6]),
        helper.make_tensor_value_info("y3", f, [1, 1, 4, 3, 5]),
        helper.make_tensor_value_info("y4", f, [1, 1, 5, 4]),
    ]
    initializers = [
        numpy_helper.from_array(seq([1, 2, 3], 0.3), "w1"),
        numpy_helper.from_array(np.array([0.5, -0.5], dtype=np.float32), "b1"),
        numpy_helper.from_array(seq([1, 1, 3, 3], 0.2), "w2"),
        numpy_helper.from_array(seq([1, 1, 2, 2, 3], 0.1), "w3"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "b3"),
        numpy_helper.from_array(seq([1, 1, 2, 2], 0.5), "w4"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "b4"),
    ]
    nodes = [
        helper.make_node(
            "ConvTranspose", ["x1", "w1", "b1"], ["y1"], auto_pad="SAME_UPPER", strides=[2]
        ),
        helper.make_node(
            "ConvTranspose",
            ["x2", "w2"],
            ["y2"],
            auto_pad="SAME_LOWER",
            strides=[2, 2],
            dilations=[2, 1],
        ),
        helper.make_node(
            "ConvTranspose",
            ["x3", "w3", "b3"],
            ["y3"],
            auto_pad="SAME_UPPER",
            strides=[2, 1, 3],
            output_padding=[1, 0, 0],
            output_shape=[4, 3, 5],
        ),
        helper.make_node(
            "ConvTranspose", ["x4", "w4", "b4"], ["y4"], strides=[2, 2], output_shape=[5, 4]
        ),
    ]
    graph = helper.make_graph(
        nodes, "conv_transpose_auto_pad", inputs, outputs, initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "conv_transpose_auto_pad.onnx")

    feeds = {
        "x1": seq([1, 1, 3], 0.5),
        "x2": seq([1, 1, 2, 3], 0.4),
        "x3": seq([1, 1, 2, 2, 2], 0.3),
        "x4": seq([1, 1, 2, 2], 0.6),
    }
    expected = ReferenceEvaluator(model).run(None, feeds)
    actual = ort.InferenceSession(model.SerializeToString()).run(None, feeds)
    for name, value, ort_value in zip(["y1", "y2", "y3", "y4"], expected, actual):
        np.testing.assert_allclose(value, ort_value, atol=1e-5)
        print(f"{name} {list(value.shape)} sum={value.sum():.5f}: {np.round(value, 5).tolist()}")


if __name__ == "__main__":
    main()
