#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_reverse_dynamic.onnx
#
# Reverse slicing on an axis whose size is only known at runtime. Codegen cannot
# resolve the bounds statically here, so it has to read them off the tensor.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

INT64_MIN = -9223372036854775808


def const(name: str, vals: list[int]):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_constant",
        value=helper.make_tensor(
            name=name,
            data_type=TensorProto.INT64,
            dims=[len(vals)],
            vals=vals,
        ),
    )


def main() -> None:
    # x[-1::-1] on the dynamic axis 0
    nodes = [
        const("starts", [-1]),
        const("ends", [INT64_MIN]),
        const("axes", [0]),
        const("steps", [-1]),
        helper.make_node(
            "Slice",
            name="slice_node",
            inputs=["input_tensor", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        ),
    ]

    graph_def = helper.make_graph(
        nodes=nodes,
        name="SliceReverseDynamicGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, ["N", 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3]),
        ],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="slice_reverse_dynamic",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "slice_reverse_dynamic.onnx")

    test_input = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = ReferenceEvaluator(model_def).run(None, {"input_tensor": test_input})[0]
    print(f"Test input:\n{test_input}")
    print(f"Expected output:\n{result}")


if __name__ == "__main__":
    main()
