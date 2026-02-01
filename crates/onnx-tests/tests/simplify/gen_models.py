#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates purpose-built ONNX models that exercise specific simplification
# patterns. Each model is designed so that simplified and unsimplified versions
# produce identical outputs, but the simplified version folds shape computations
# into constants at codegen time.

import onnx
from onnx import helper, TensorProto

OPSET = 16


def save(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, name)
    print(f"  {name}")


def shape_folding():
    """Shape(x) where x has fully static shape -> foldable to constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_out"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "shape_out",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_shape_folding.onnx",
    )


def gather_on_shape():
    """Gather(Shape(x), const_idx) -> single dimension extracted as scalar."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["idx"],
                value=helper.make_tensor("idx_val", TensorProto.INT64, [], [1]),
            ),
            helper.make_node("Gather", ["shape1", "idx"], ["dim_val"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "dim_val",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_gather_on_shape.onnx",
    )


def slice_on_shape():
    """Slice(Shape(x), [1], [3]) -> sub-array of shape dims."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("s", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor("e", TensorProto.INT64, [1], [3]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["axes"],
                value=helper.make_tensor("a", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Slice", ["shape1", "starts", "ends", "axes"], ["sliced"]
            ),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4, 5]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "sliced",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[2]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_slice_on_shape.onnx",
    )


def concat_shapes():
    """Concat(Shape(x), Shape(y)) -> concatenated shape array."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            helper.make_node("Concat", ["shape_x", "shape_y"], ["combined"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[4, 5, 6]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "combined",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[5]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_concat_shapes.onnx",
    )


def reshape_from_shape():
    """Reshape(x, Shape(y)) -> reshape using folded shape constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["shape_source"], ["target_shape"]),
            helper.make_node("Reshape", ["x", "target_shape"], ["reshaped"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[12]),
            ),
            helper.make_value_info(
                "shape_source",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "reshaped",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 4]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_reshape_from_shape.onnx",
    )


def binary_ops_on_shape():
    """Add(Shape(x), const) and Mul(Shape(x), const) on shape values."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["offset"],
                value=helper.make_tensor("off", TensorProto.INT64, [3], [10, 20, 30]),
            ),
            helper.make_node("Add", ["shape1", "offset"], ["added"]),
            helper.make_node(
                "Constant",
                [],
                ["scale"],
                value=helper.make_tensor("sc", TensorProto.INT64, [3], [2, 2, 2]),
            ),
            helper.make_node("Mul", ["shape1", "scale"], ["mulled"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "added",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
            helper.make_value_info(
                "mulled",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_binary_ops_on_shape.onnx",
    )


def cast_shape():
    """Cast(Shape(x), to=FLOAT) -> cast shape to float tensor."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node("Cast", ["shape1"], ["casted"], to=TensorProto.FLOAT),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "casted",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_cast_shape.onnx",
    )


def where_on_shapes():
    """Where(scalar_cond, Shape(x), Shape(y)) -> conditional shape selection.

    Uses scalar bool condition (the only supported pattern for Where with Shape
    outputs besides element-wise Shape,Shape,Shape).
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            helper.make_node("Where", ["cond", "shape_x", "shape_y"], ["selected"]),
        ],
        inputs=[
            helper.make_value_info(
                "cond",
                helper.make_tensor_type_proto(TensorProto.BOOL, shape=[]),
            ),
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[5, 6, 7]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "selected",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_where_on_shapes.onnx",
    )


def expand_from_shape():
    """Expand(x, Shape(y)) -> expand using folded shape constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["shape_source"], ["target_shape"]),
            helper.make_node("Expand", ["x", "target_shape"], ["expanded"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 1, 4]),
            ),
            helper.make_value_info(
                "shape_source",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "expanded",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_expand_from_shape.onnx",
    )


def constant_of_shape_opt():
    """ConstantOfShape(Shape(x)) -> constant tensor with known shape."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "ConstantOfShape",
                ["shape1"],
                ["filled"],
                value=helper.make_tensor("val", TensorProto.FLOAT, [1], [5.0]),
            ),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "filled",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_constant_of_shape_opt.onnx",
    )


def gather_shape_chain():
    """Chained Shape->Gather where first Gather's result feeds second Gather's index.

    Shape(x) -> Gather(const 1) -> first_dim (== 1 for x.shape=[3,1,4])
    Shape(y) -> Gather(first_dim) -> second_dim (== dim_1 of y)

    The second Gather uses the first's output as its index. Without proper
    value_store propagation, the second Gather can never be folded because
    its index input has value_source=Constant but value_store=None.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node(
                "Constant",
                [],
                ["idx1"],
                value=helper.make_tensor("idx1_val", TensorProto.INT64, [], [1]),
            ),
            # Gather dim 1 from x's shape: x.shape=[3,1,4] -> 1
            helper.make_node("Gather", ["shape_x", "idx1"], ["dim_from_x"], axis=0),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            # Use the gathered value (1) as index into y's shape: y.shape=[5,6,7] -> 6
            helper.make_node("Gather", ["shape_y", "dim_from_x"], ["dim_from_y"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 1, 4]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[5, 6, 7]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "dim_from_y",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_gather_shape_chain.onnx",
    )


if __name__ == "__main__":
    print("Generating simplify test models:")
    shape_folding()
    gather_on_shape()
    slice_on_shape()
    concat_shapes()
    reshape_from_shape()
    binary_ops_on_shape()
    cast_shape()
    where_on_shapes()
    expand_from_shape()
    constant_of_shape_opt()
    gather_shape_chain()
    print("Done.")
