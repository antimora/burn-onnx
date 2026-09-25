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
#   y4: 2D output_shape one past the full result on the first axis, grown at the end
#   y5: two stacked 2D SAME_UPPER layers, the second sized from the first's output
#   y6: 1D output_shape without auto_pad and a positive total pad, odd unit at the start
#   y7: 2D VALID
# ONNX Runtime is the ground truth. The reference evaluator must agree on every output
# except y6, where its col2im shape check rejects the node.

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
        helper.make_tensor_value_info("x5", f, [1, 1, 2, 2]),
        helper.make_tensor_value_info("x6", f, [1, 1, 3]),
        helper.make_tensor_value_info("x7", f, [1, 1, 2, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("y1", f, [1, 2, 6]),
        helper.make_tensor_value_info("y2", f, [1, 1, 4, 6]),
        helper.make_tensor_value_info("y3", f, [1, 1, 4, 3, 5]),
        helper.make_tensor_value_info("y4", f, [1, 1, 5, 4]),
        helper.make_tensor_value_info("y5", f, [1, 1, 8, 8]),
        helper.make_tensor_value_info("y6", f, [1, 1, 6]),
        helper.make_tensor_value_info("y7", f, [1, 1, 5, 5]),
    ]
    initializers = [
        numpy_helper.from_array(seq([1, 2, 3], 0.3), "w1"),
        numpy_helper.from_array(np.array([0.5, -0.5], dtype=np.float32), "b1"),
        numpy_helper.from_array(seq([1, 1, 3, 3], 0.2), "w2"),
        numpy_helper.from_array(seq([1, 1, 2, 2, 3], 0.1), "w3"),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "b3"),
        numpy_helper.from_array(seq([1, 1, 2, 2], 0.5), "w4"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "b4"),
        numpy_helper.from_array(seq([1, 2, 3, 3], 0.1), "w5a"),
        numpy_helper.from_array(seq([2, 1, 3, 3], 0.05), "w5b"),
        numpy_helper.from_array(seq([1, 1, 3], 0.5), "w6"),
        numpy_helper.from_array(seq([1, 1, 3, 3], 0.25), "w7"),
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
        helper.make_node(
            "ConvTranspose", ["x5", "w5a"], ["h5"], auto_pad="SAME_UPPER", strides=[2, 2]
        ),
        helper.make_node(
            "ConvTranspose", ["h5", "w5b"], ["y5"], auto_pad="SAME_UPPER", strides=[2, 2]
        ),
        helper.make_node("ConvTranspose", ["x6", "w6"], ["y6"], strides=[2], output_shape=[6]),
        helper.make_node(
            "ConvTranspose",
            ["x7", "w7"],
            ["y7"],
            auto_pad="VALID",
            strides=[2, 2],
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
        "x5": seq([1, 1, 2, 2], 0.7),
        "x6": seq([1, 1, 3], 0.5),
        "x7": seq([1, 1, 2, 2], 0.5),
    }
    names = [out.name for out in outputs]
    expected = dict(zip(names, ort.InferenceSession(model.SerializeToString()).run(None, feeds)))

    # Cross-check against the reference evaluator on the graph without y6.
    checked = [node for node in nodes if node.output[0] != "y6"]
    reference_graph = helper.make_graph(
        checked,
        "reference",
        [i for i in inputs if i.name != "x6"],
        [o for o in outputs if o.name != "y6"],
        initializer=initializers,
    )
    reference_model = helper.make_model(
        reference_graph, opset_imports=[helper.make_opsetid("", 16)]
    )
    reference_feeds = {k: v for k, v in feeds.items() if k != "x6"}
    reference = ReferenceEvaluator(reference_model).run(None, reference_feeds)
    for output, value in zip(reference_graph.output, reference):
        np.testing.assert_allclose(value, expected[output.name], atol=1e-5)

    for name in names:
        value = expected[name]
        print(f"{name} {list(value.shape)} sum={value.sum():.5f}: {np.round(value, 5).tolist()}")

if __name__ == "__main__":
    main()
