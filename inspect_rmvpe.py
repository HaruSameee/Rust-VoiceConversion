#!/usr/bin/env python
import onnx
import onnxruntime as ort
import numpy as np


def inspect(path: str):
    # 1a. Static shape from ONNX graph
    model = onnx.load(path)
    inp = model.graph.input[0]
    out = model.graph.output[0]
    in_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"[static]  input  name='{inp.name}' shape={in_shape}")
    print(f"[static]  output name='{out.name}' shape={out_shape}")

    # 1b. Runtime shape with dummy input
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 16000), dtype=np.float32)
    result = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f"[runtime] output shape={result[0].shape}  dtype={result[0].dtype}")
    frames = result[0].shape[1] if result[0].ndim >= 2 else result[0].shape[0]
    print(f"[runtime] output frames={frames}")

    # 1c. List all nodes that contain literal value 50 or 49
    print("\n[nodes containing 49 or 50]")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                if 49 in attr.ints or 50 in attr.ints:
                    print(
                        f"  node={node.name} op={node.op_type} attr={attr.name} vals={list(attr.ints)}"
                    )
            if attr.type == onnx.AttributeProto.INT:
                if attr.i in (49, 50):
                    print(
                        f"  node={node.name} op={node.op_type} attr={attr.name} val={attr.i}"
                    )

    # 1d. Identify Resize / Interpolate nodes
    print("\n[Resize / Interpolate nodes]")
    for node in model.graph.node:
        if node.op_type in ("Resize", "Upsample", "Interpolate"):
            print(
                f"  name={node.name} inputs={list(node.input)} outputs={list(node.output)}"
            )


if __name__ == "__main__":
    import sys

    inspect(sys.argv[1])
