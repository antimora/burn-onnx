#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""
Generate ONNX model with a Loop whose body leaves its loop-carried inputs untyped,
the pattern used by the ONNX function expansion of Range. Each body input takes its
type from the matching v_initial.

Body:
- prev_copy = Identity(prev); next = prev_copy + x, where x is an outer-scope reference
  and prev (f32 [2, 3]) is declared with an empty TypeProto
- count_next = count + one, where count (int64 [3]) has no type field at all
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def build_model():
    body_graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["prev"], ["prev_copy"]),
            helper.make_node("Add", ["prev_copy", "x"], ["next"]),
            helper.make_node(
                "Constant",
                [],
                ["one"],
                value=helper.make_tensor("one_value", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node("Add", ["count", "one"], ["count_next"]),
        ],
        name="loop_body",
        inputs=[
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            # No type: inferred from the Loop's v_initial input
            helper.make_value_info("prev", onnx.TypeProto()),
            onnx.ValueInfoProto(name="count"),
        ],
        outputs=[
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_value_info("next", onnx.TypeProto()),
            helper.make_value_info("count_next", onnx.TypeProto()),
        ],
    )

    loop_node = helper.make_node(
        "Loop",
        inputs=["M", "cond", "initial", "count_initial"],
        outputs=["final", "count_final"],
        body=body_graph,
    )

    graph = helper.make_graph(
        nodes=[loop_node],
        name="loop_untyped_carried_model",
        inputs=[
            helper.make_tensor_value_info("M", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("initial", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("count_initial", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("final", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("count_final", TensorProto.INT64, [3]),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="burn-onnx-test",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.checker.check_model(model)
    return model


def main():
    np.random.seed(42)
    model = build_model()
    onnx.save(model, "loop_untyped_carried.onnx")
    print("Saved loop_untyped_carried.onnx")

    initial = np.random.randn(2, 3).astype(np.float32)
    x = np.random.randn(2, 3).astype(np.float32)
    count_initial = np.array([10, 20, 30], dtype=np.int64)
    final, count_final = ReferenceEvaluator(model).run(
        None,
        {
            "M": np.array(3, dtype=np.int64),
            "cond": np.array(True),
            "initial": initial,
            "count_initial": count_initial,
            "x": x,
        },
    )

    print(f"initial: {initial.tolist()}")
    print(f"x: {x.tolist()}")
    print(f"final (M=3): {final.tolist()}")
    print(f"count_final (M=3): {count_final.tolist()}")


if __name__ == "__main__":
    main()
