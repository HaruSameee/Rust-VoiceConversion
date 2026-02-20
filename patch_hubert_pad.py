#!/usr/bin/env python
"""
Insert constant-zero tail padding into HuBERT ONNX input.

This script keeps host-side contract [1, 16000] unchanged and adds internal
Pad([0,0,0,80]) so the model effectively runs on [1, 16080].
"""

import argparse

import numpy as np
import onnx
from onnx import checker, numpy_helper, shape_inference


def load_and_inspect(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    graph = model.graph
    inp = graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"[inspect] input name='{inp.name}' shape={shape}")
    if len(shape) != 2 or shape[1] != 16000:
        raise AssertionError(
            f"Expected rank-2 input with time axis 16000, got {shape}. Wrong model?"
        )
    if shape[0] not in (0, 1):
        raise AssertionError(
            f"Expected batch dim 0/1 for strict host contract, got {shape[0]}"
        )
    print(f"[inspect] opset={model.opset_import[0].version}")
    return model


def insert_pad_node(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    original_input_name = graph.input[0].name
    padded_name = original_input_name + "_padded80"

    # ONNX Pad pads format for rank=2 is:
    # [begin_dim0, begin_dim1, end_dim0, end_dim1]
    pad_values = np.array([0, 0, 0, 80], dtype=np.int64)
    pad_tensor = numpy_helper.from_array(pad_values, name="pad_amounts_val")

    pad_const_node = onnx.helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=["pad_amounts"],
        value=pad_tensor,
        name="Constant_pad_amounts",
    )

    fill_tensor = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="pad_fill_val"
    )
    fill_const_node = onnx.helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=["pad_fill_value"],
        value=fill_tensor,
        name="Constant_pad_fill",
    )

    pad_node = onnx.helper.make_node(
        op_type="Pad",
        inputs=[original_input_name, "pad_amounts", "pad_fill_value"],
        outputs=[padded_name],
        mode="constant",
        name="Pad_audio_80",
    )

    # Rewire existing nodes to consume the padded tensor.
    for node in graph.node:
        for i, node_input in enumerate(node.input):
            if node_input == original_input_name:
                node.input[i] = padded_name

    # Prepend in dependency order.
    graph.node.insert(0, pad_node)
    graph.node.insert(0, fill_const_node)
    graph.node.insert(0, pad_const_node)

    # Clear stale shape metadata and let ORT/shape-inference rebuild.
    del graph.value_info[:]

    return model


def validate(model: onnx.ModelProto, output_path: str) -> onnx.ModelProto:
    print("[validate] running shape inference...")
    model = shape_inference.infer_shapes(model)

    print("[validate] running checker...")
    checker.check_model(model)
    print("[validate] checker PASSED")

    onnx.save(model, output_path)
    print(f"[validate] saved -> {output_path}")
    return model


def verify_with_ort(original_path: str, patched_path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed, skipping")
        return

    dummy = np.zeros((1, 16000), dtype=np.float32)

    sess_orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    out_orig = sess_orig.run(None, {sess_orig.get_inputs()[0].name: dummy})
    frames_orig = out_orig[0].shape[1]
    print(f"[verify] original  output frames = {frames_orig}  (expected 49)")

    sess_patch = ort.InferenceSession(patched_path, providers=["CPUExecutionProvider"])
    out_patch = sess_patch.run(None, {sess_patch.get_inputs()[0].name: dummy})
    frames_patch = out_patch[0].shape[1]
    print(f"[verify] patched   output frames = {frames_patch}  (expected 50)")

    assert frames_orig == 49, f"Original should produce 49 frames, got {frames_orig}"
    assert frames_patch == 50, f"Patched should produce 50 frames, got {frames_patch}"
    print("[verify] PASSED: 49 -> 50 frame conversion confirmed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to hubert_strict.onnx")
    parser.add_argument("--output", required=True, help="Path to write hubert_pad80.onnx")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ORT inference to confirm 50-frame output",
    )
    args = parser.parse_args()

    model = load_and_inspect(args.input)
    model = insert_pad_node(model)
    validate(model, args.output)

    if args.verify:
        verify_with_ort(args.input, args.output)

    print("[done] Rust IoBinding contract unchanged: host supplies [1,16000]")
    print("[done] ONNX internally pads to [1,16080] -> HuBERT outputs 50 frames")
    print("[done] Remove Catmull-Rom interpolation from Rust inference pipeline")


if __name__ == "__main__":
    main()
