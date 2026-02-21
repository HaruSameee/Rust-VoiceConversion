"""
Export a trained RVC generator checkpoint (.pt/.pth) to ONNX with dynamic axes.

This script targets common RVC model classes from `infer_pack.models` and exports
an inference wrapper with dynamic time axis support so runtime can pass variable
frame lengths (and therefore variable waveform lengths).

Usage:
  python scripts/export_dynamic_rvc_onnx.py \
    --checkpoint scripts/rv_e250_s36750.pth \
    --output model/model_dynamic.onnx \
    --rvc-root <path-to-rvc-python-repo> \
    --verify
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except Exception:
    ort = None


DEFAULT_OPSET = 17
DEFAULT_FRAMES = 200
DEFAULT_BATCH = 1
DEFAULT_RND_CHANNELS = 192
DEFAULT_PHONE_DIM_V1 = 256
DEFAULT_PHONE_DIM_V2 = 768


def add_rvc_root_to_path(rvc_root: Optional[Path]) -> None:
    if rvc_root is None:
        return
    root = str(rvc_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, Mapping):
        raise RuntimeError(f"checkpoint is not dict-like: {path}")
    return ckpt


def unwrap_state_dict(ckpt: Mapping[str, Any]) -> MutableMapping[str, torch.Tensor]:
    if "weight" in ckpt and isinstance(ckpt["weight"], MutableMapping):
        return ckpt["weight"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], MutableMapping):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], MutableMapping):
        return ckpt["model"]
    # Plain state_dict fallback
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return dict(ckpt)  # shallow copy for mutation safety
    raise RuntimeError(
        "could not locate state_dict in checkpoint. Expected one of: "
        "`weight`, `state_dict`, `model`, or plain tensor map."
    )


def infer_version(ckpt: Mapping[str, Any]) -> str:
    v = ckpt.get("version", "v2")
    return str(v).lower()


def infer_has_f0(ckpt: Mapping[str, Any]) -> bool:
    raw = ckpt.get("f0", 1)
    try:
        return bool(int(raw))
    except Exception:
        return True


def infer_phone_dim(version: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    return DEFAULT_PHONE_DIM_V2 if "v2" in version else DEFAULT_PHONE_DIM_V1


def update_checkpoint_config_with_speakers(
    ckpt: Mapping[str, Any], state: Mapping[str, torch.Tensor]
) -> Optional[list]:
    cfg = ckpt.get("config")
    if not isinstance(cfg, (list, tuple)):
        return None
    cfg_list = list(cfg)
    emb = state.get("emb_g.weight")
    if emb is not None and emb.ndim >= 1 and len(cfg_list) >= 3:
        # RVC convention: cfg[-3] == n_speakers
        cfg_list[-3] = int(emb.shape[0])
    return cfg_list


def pick_model_class(
    module: Any,
    version: str,
    has_f0: bool,
    explicit_class: Optional[str],
) -> type[nn.Module]:
    if explicit_class:
        if not hasattr(module, explicit_class):
            raise RuntimeError(f"class `{explicit_class}` not found in module `{module.__name__}`")
        return getattr(module, explicit_class)

    preferred = (
        "SynthesizerTrnMs768NSFsid" if "v2" in version else "SynthesizerTrnMs256NSFsid"
    )
    candidates = []
    if has_f0:
        candidates.extend(
            [
                preferred,
                "SynthesizerTrnMs768NSFsid",
                "SynthesizerTrnMs256NSFsid",
                "SynthesizerTrnMsNSFsid",
            ]
        )
    else:
        candidates.extend(
            [
                f"{preferred}_nono",
                "SynthesizerTrnMs768NSFsid_nono",
                "SynthesizerTrnMs256NSFsid_nono",
                "SynthesizerTrnMsNSFsidNono",
                "SynthesizerTrnMsNSFsid_nono",
            ]
        )
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    raise RuntimeError(
        "could not find a known RVC synthesizer class. "
        "Pass --class-name explicitly."
    )


def instantiate_model(
    cls: type[nn.Module], cfg: Optional[list], device: torch.device
) -> nn.Module:
    errors: list[str] = []
    constructors = []
    if cfg is not None:
        constructors.append((cfg, {"is_half": False}))
        constructors.append((cfg, {}))
    constructors.append(([], {}))

    for args, kwargs in constructors:
        try:
            model = cls(*args, **kwargs)
            model.to(device)
            model.eval()
            return model
        except Exception as exc:  # keep trying plausible constructor variants
            errors.append(f"{cls.__name__}(*{len(args)} args, kwargs={kwargs}): {exc}")
    raise RuntimeError("failed to instantiate model:\n  - " + "\n  - ".join(errors))


def normalize_state_keys(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        name = k
        if name.startswith("module."):
            name = name[len("module.") :]
        out[name] = v
    return out


class DynamicRvcWrapper(nn.Module):
    def __init__(self, model: nn.Module, has_f0: bool):
        super().__init__()
        self.model = model
        self.has_f0 = bool(has_f0)

    def _build_infer_args(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        rnd: torch.Tensor,
    ) -> list[Any]:
        sig = inspect.signature(self.model.infer)
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        args: list[Any] = []
        for p in params:
            n = p.name.lower()
            if "phone" in n or "feat" in n:
                args.append(phone)
            elif "length" in n:
                args.append(phone_lengths)
            elif n == "pitch" or "pitch_" in n:
                args.append(pitch)
            elif "pitchf" in n or "nsff0" in n or n == "f0":
                args.append(pitchf)
            elif n in ("sid", "ds") or "speaker" in n or "spk" in n:
                args.append(sid)
            elif "rnd" in n or "noise" in n or n == "z":
                args.append(rnd)
            elif p.default is not inspect._empty:
                # optional arg; keep default by skipping
                continue
            else:
                raise RuntimeError(
                    f"cannot map required infer() parameter `{p.name}`. "
                    "Pass a compatible class via --class-name."
                )
        return args

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        rnd: torch.Tensor,
    ) -> torch.Tensor:
        args = self._build_infer_args(phone, phone_lengths, pitch, pitchf, sid, rnd)
        out = self.model.infer(*args)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"model.infer returned non-tensor output: {type(out)}")
        # Expected waveform shape [B, 1, T] for RVC decoder ONNX.
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out


def build_dummy_inputs(
    batch: int,
    frames: int,
    phone_dim: int,
    rnd_channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    phone = torch.zeros((batch, frames, phone_dim), dtype=torch.float32, device=device)
    phone_lengths = torch.full((batch,), frames, dtype=torch.long, device=device)
    pitch = torch.full((batch, frames), 128, dtype=torch.long, device=device)
    pitchf = torch.full((batch, frames), 110.0, dtype=torch.float32, device=device)
    sid = torch.zeros((batch,), dtype=torch.long, device=device)
    rnd = torch.zeros((batch, rnd_channels, frames), dtype=torch.float32, device=device)
    return phone, phone_lengths, pitch, pitchf, sid, rnd


def export_dynamic_onnx(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    output: Path,
    opset: int,
) -> None:
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"]
    output_names = ["audio"]
    dynamic_axes = {
        "phone": {0: "batch", 1: "n_frames"},
        "phone_lengths": {0: "batch"},
        "pitch": {0: "batch", 1: "n_frames"},
        "pitchf": {0: "batch", 1: "n_frames"},
        "sid": {0: "batch"},
        "rnd": {0: "batch", 2: "n_frames"},
        "audio": {0: "batch", 2: "n_samples"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )


def verify_dynamic_runtime(output: Path, batch: int, phone_dim: int, rnd_channels: int) -> None:
    if ort is None:
        print("[warn] onnxruntime not installed; skipping runtime dynamic verification.")
        return
    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])

    def run_frames(frames: int) -> int:
        import numpy as np

        feed = {
            "phone": np.zeros((batch, frames, phone_dim), dtype=np.float32),
            "phone_lengths": np.full((batch,), frames, dtype=np.int64),
            "pitch": np.full((batch, frames), 128, dtype=np.int64),
            "pitchf": np.full((batch, frames), 110.0, dtype=np.float32),
            "sid": np.zeros((batch,), dtype=np.int64),
            "rnd": np.zeros((batch, rnd_channels, frames), dtype=np.float32),
        }
        out = sess.run(["audio"], feed)[0]
        return int(out.shape[-1])

    y1 = run_frames(100)
    y2 = run_frames(200)
    print(f"[verify] frames=100 -> audio_len={y1}")
    print(f"[verify] frames=200 -> audio_len={y2}")
    if y2 <= y1:
        raise RuntimeError(
            "dynamic verification failed: output length did not grow with input frames"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export dynamic-axis RVC generator ONNX from checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to trained RVC .pt/.pth")
    p.add_argument("--output", required=True, help="Path to output ONNX file")
    p.add_argument(
        "--rvc-root",
        default=".",
        help="Path containing infer_pack/models.py (added to sys.path)",
    )
    p.add_argument(
        "--module",
        default="infer_pack.models",
        help="Python module containing RVC model classes (default: infer_pack.models)",
    )
    p.add_argument(
        "--class-name",
        default=None,
        help="Explicit class name (auto-detected if omitted)",
    )
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Dummy export frame length")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Dummy export batch size")
    p.add_argument(
        "--rnd-channels",
        type=int,
        default=DEFAULT_RND_CHANNELS,
        help="Noise latent channels (default: 192)",
    )
    p.add_argument("--phone-dim", type=int, default=None, help="Override phone feature dim")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset (default: 17)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Export device")
    p.add_argument("--verify", action="store_true", help="Run ORT dynamic-shape sanity check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    rvc_root = Path(args.rvc_root).resolve() if args.rvc_root else None
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    add_rvc_root_to_path(rvc_root)
    module = importlib.import_module(args.module)
    ckpt = load_checkpoint(checkpoint)
    state = normalize_state_keys(unwrap_state_dict(ckpt))
    version = infer_version(ckpt)
    has_f0 = infer_has_f0(ckpt)
    phone_dim = infer_phone_dim(version, args.phone_dim)
    cfg = update_checkpoint_config_with_speakers(ckpt, state)
    cls = pick_model_class(module, version, has_f0, args.class_name)

    device = torch.device(args.device)
    model = instantiate_model(cls, cfg, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    wrapper = DynamicRvcWrapper(model, has_f0).to(device).eval()
    dummy_inputs = build_dummy_inputs(
        batch=max(1, args.batch),
        frames=max(2, args.frames),
        phone_dim=phone_dim,
        rnd_channels=max(1, args.rnd_channels),
        device=device,
    )
    with torch.no_grad():
        _ = wrapper(*dummy_inputs)

    export_dynamic_onnx(
        wrapper,
        dummy_inputs,
        output=output,
        opset=max(11, args.opset),
    )
    print(f"[ok] exported dynamic ONNX: {output}")
    print(
        f"[meta] version={version} has_f0={has_f0} phone_dim={phone_dim} "
        f"rnd_channels={args.rnd_channels} opset={max(11, args.opset)}"
    )

    if args.verify:
        verify_dynamic_runtime(
            output=output,
            batch=max(1, args.batch),
            phone_dim=phone_dim,
            rnd_channels=max(1, args.rnd_channels),
        )
        print("[ok] dynamic runtime verification passed")


if __name__ == "__main__":
    main()
