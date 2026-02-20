#!/usr/bin/env python
"""
Patch RMVPE ONNX to enforce decoder frame contract (50 frames).
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import onnx
from onnx import checker, numpy_helper, shape_inference


def runtime_frame_count(arr: np.ndarray) -> int:
    if arr.ndim >= 2:
        return int(arr.shape[1])
    return int(arr.shape[0])


def inspect_model(path: str) -> Tuple[onnx.ModelProto, int, Tuple[int, ...]]:
    model = onnx.load(path)
    inp = model.graph.input[0]
    out = model.graph.output[0]
    in_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"[static]  input  name='{inp.name}' shape={in_shape}")
    print(f"[static]  output name='{out.name}' shape={out_shape}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for frame-count inspection"
        ) from exc

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 16000), dtype=np.float32)
    output = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    frames = runtime_frame_count(output)
    print(f"[runtime] output shape={output.shape} dtype={output.dtype}")
    print(f"[runtime] output frames={frames}")

    print("\n[nodes containing 49 or 50]")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS and (
                49 in attr.ints or 50 in attr.ints
            ):
                print(
                    f"  node={node.name} op={node.op_type} attr={attr.name} vals={list(attr.ints)}"
                )
            if attr.type == onnx.AttributeProto.INT and attr.i in (49, 50):
                print(
                    f"  node={node.name} op={node.op_type} attr={attr.name} val={attr.i}"
                )

    print("\n[Resize / Interpolate nodes]")
    for node in model.graph.node:
        if node.op_type in ("Resize", "Upsample", "Interpolate"):
            print(
                f"  name={node.name} inputs={list(node.input)} outputs={list(node.output)}"
            )

    return model, frames, tuple(output.shape)


def detect_case(frames: int) -> str:
    if frames == 50:
        return "A"
    if frames == 100:
        return "B"
    if frames == 49:
        return "C"
    return "D"


def patch_case_a(model: onnx.ModelProto) -> Optional[onnx.ModelProto]:
    print("[case A] RMVPE already outputs 50 frames. No ONNX change required.")
    return None


def append_resize_to_output(
    model: onnx.ModelProto, runtime_output_shape: Tuple[int, ...], target_frames: int = 50
) -> onnx.ModelProto:
    graph = model.graph
    original_output_name = graph.output[0].name
    resized_name = original_output_name + f"_resized{target_frames}"
    output_rank = len(runtime_output_shape)
    if output_rank < 2:
        raise RuntimeError(
            f"Unsupported RMVPE output rank for Resize patch: rank={output_rank}"
        )

    # Build sizes tensor [B, 50, C] (or [B,50] for rank-2).
    sizes = list(runtime_output_shape)
    sizes[0] = 1
    sizes[1] = target_frames
    sizes = np.array(sizes, dtype=np.int64)

    sizes_tensor = numpy_helper.from_array(sizes, name="rmvpe_resize_sizes")
    sizes_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["rmvpe_resize_sizes_out"],
        value=sizes_tensor,
        name="Constant_rmvpe_sizes",
    )
    roi_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["rmvpe_roi"],
        value=numpy_helper.from_array(np.array([], dtype=np.float32), name="rmvpe_roi_val"),
        name="Constant_rmvpe_roi",
    )
    scales_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["rmvpe_scales"],
        value=numpy_helper.from_array(
            np.array([], dtype=np.float32), name="rmvpe_scales_val"
        ),
        name="Constant_rmvpe_scales",
    )
    resize_node = onnx.helper.make_node(
        "Resize",
        inputs=[
            original_output_name,
            "rmvpe_roi",
            "rmvpe_scales",
            "rmvpe_resize_sizes_out",
        ],
        outputs=[resized_name],
        coordinate_transformation_mode="half_pixel",
        mode="linear",
        name=f"Resize_rmvpe_to_{target_frames}",
    )

    graph.node.extend([roi_const, scales_const, sizes_const, resize_node])
    graph.output[0].name = resized_name
    del graph.value_info[:]
    return model


def patch_case_b(
    model: onnx.ModelProto, runtime_output_shape: Tuple[int, ...]
) -> onnx.ModelProto:
    print("[case B] RMVPE outputs 100 frames. Inserting output Resize 100->50.")
    return append_resize_to_output(model, runtime_output_shape, target_frames=50)


def patch_case_c(model: onnx.ModelProto, pad_samples: int) -> onnx.ModelProto:
    print(
        f"[case C] RMVPE outputs 49 frames. Inserting input Pad with pad_samples={pad_samples}."
    )
    graph = model.graph
    original_input_name = graph.input[0].name
    padded_name = original_input_name + f"_padded{pad_samples}"
    pad_values = np.array([0, 0, 0, pad_samples], dtype=np.int64)
    pad_tensor = numpy_helper.from_array(pad_values, name="rmvpe_pad_amounts_val")
    pad_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["rmvpe_pad_amounts"],
        value=pad_tensor,
        name="Constant_rmvpe_pad_amounts",
    )
    fill_tensor = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="rmvpe_pad_fill_val"
    )
    fill_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["rmvpe_pad_fill_value"],
        value=fill_tensor,
        name="Constant_rmvpe_pad_fill",
    )
    pad_node = onnx.helper.make_node(
        "Pad",
        inputs=[original_input_name, "rmvpe_pad_amounts", "rmvpe_pad_fill_value"],
        outputs=[padded_name],
        mode="constant",
        name=f"Pad_rmvpe_input_{pad_samples}",
    )
    for node in graph.node:
        for i, node_input in enumerate(node.input):
            if node_input == original_input_name:
                node.input[i] = padded_name
    graph.node.insert(0, pad_node)
    graph.node.insert(0, fill_const_node)
    graph.node.insert(0, pad_const_node)
    del graph.value_info[:]
    return model


def patch_case_d(
    model: onnx.ModelProto, runtime_output_shape: Tuple[int, ...], frames: int
) -> onnx.ModelProto:
    print(
        f"[case D] RMVPE outputs unexpected frame count N={frames}. "
        "Applying output Resize to 50 as architecture-neutral fix."
    )
    return append_resize_to_output(model, runtime_output_shape, target_frames=50)


def validate_and_save(model: onnx.ModelProto, output_path: str) -> onnx.ModelProto:
    print("[validate] running shape inference...")
    inferred = shape_inference.infer_shapes(model)
    print("[validate] running checker...")
    checker.check_model(inferred)
    print("[validate] checker PASSED")
    onnx.save(inferred, output_path)
    print(f"[validate] saved -> {output_path}")
    return inferred


def verify(original_path: str, patched_path: str):
    import onnxruntime as ort

    sess_orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    sess_patch = ort.InferenceSession(patched_path, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 16000), dtype=np.float32)

    out_orig = sess_orig.run(None, {sess_orig.get_inputs()[0].name: dummy})[0]
    out_patch = sess_patch.run(None, {sess_patch.get_inputs()[0].name: dummy})[0]

    frames_orig = runtime_frame_count(out_orig)
    frames_patch = runtime_frame_count(out_patch)

    print(f"[verify] original  frames={frames_orig}")
    print(f"[verify] patched   frames={frames_patch} (expected 50)")
    assert frames_patch == 50, f"FAILED: expected 50, got {frames_patch}"
    print("[verify] PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to rmvpe_strict.onnx")
    parser.add_argument("--output", required=True, help="Path to patched ONNX")
    parser.add_argument("--verify", action="store_true", help="Run ORT verification")
    args = parser.parse_args()

    model, frames, runtime_shape = inspect_model(args.input)
    case = detect_case(frames)
    print(f"\n[decision] case={case}")

    patched_model: Optional[onnx.ModelProto]
    if case == "A":
        patched_model = patch_case_a(model)
        if patched_model is None:
            # Keep workflow simple: write a checked copy for downstream use.
            patched_model = model
    elif case == "B":
        patched_model = patch_case_b(model, runtime_shape)
    elif case == "C":
        # Same strategy as HuBERT stride fix: append 80 samples on time-axis.
        patched_model = patch_case_c(model, pad_samples=80)
    else:
        patched_model = patch_case_d(model, runtime_shape, frames)

    validate_and_save(patched_model, args.output)
    if args.verify:
        verify(args.input, args.output)

    print("[done] IoBinding host input contract remains [1,16000]")


if __name__ == "__main__":
    main()
