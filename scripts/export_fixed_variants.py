"""
NOTE: このスクリプトで生成した固定長ONNXは品質面の問題により現在は非推奨です。
      本番環境では hubert_pad80.onnx / rmvpe_strict.onnx (dynamic axes) を使用してください。
      将来の最適化実験用に保持しています。
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

# BASE_DIR sys.path setup (same intent as existing exporter scripts)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _force_torch_load_weights_only_false() -> None:
    # torch>=2.6 defaults to weights_only=True. Keep legacy checkpoint behavior.
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def _add_scripts_to_path(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _patch_fairseq_pad_to_multiple_for_fixed_export() -> None:
    utils_module = importlib.import_module("fairseq.models.wav2vec.utils")
    wav2vec2_module = importlib.import_module("fairseq.models.wav2vec.wav2vec2")
    original = getattr(utils_module, "pad_to_multiple", None)
    if original is None or getattr(utils_module, "_rust_vc_fixed_export_patched", False):
        return

    def patched_pad_to_multiple(x, multiple, dim=-1, value=0):
        if x is None:
            return None, 0
        tsz = x.size(dim)
        if isinstance(tsz, torch.Tensor):
            tsz = int(tsz.item())
        else:
            tsz = int(tsz)
        m = tsz / multiple
        remainder = math.ceil(m) * multiple - tsz
        if remainder == 0:
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

    utils_module.pad_to_multiple = patched_pad_to_multiple
    wav2vec2_module.pad_to_multiple = patched_pad_to_multiple
    utils_module._rust_vc_fixed_export_patched = True


class HubertWrapper(torch.nn.Module):
    def __init__(self, hubert_model: torch.nn.Module):
        super().__init__()
        self.hubert = hubert_model

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Output is [B, T, C] where C=768 for HuBERT base.
        return self.hubert(
            source=source,
            padding_mask=None,
            mask=False,
            features_only=True,
        )["x"]


class RmvpeMelToF0(torch.nn.Module):
    def __init__(self, rmvpe_model: torch.nn.Module, threshold: float = 0.03):
        super().__init__()
        self.rmvpe_model = rmvpe_model
        self.threshold = float(threshold)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 128, n_frames]
        n_frames = mel.shape[-1]
        padded_frames = 32 * ((n_frames - 1) // 32 + 1)
        if padded_frames > n_frames:
            mel = F.pad(mel, (0, padded_frames - n_frames), mode="reflect")
        salience = self.rmvpe_model(mel).float()[:, :n_frames, :]  # [B, frames, 360]
        conf, idx = torch.max(salience, dim=2)
        cents = 1997.3794 + idx.float() * 20.0
        f0 = 10.0 * torch.pow(torch.tensor(2.0, device=mel.device), cents / 1200.0)
        return torch.where(conf > self.threshold, f0, torch.zeros_like(f0))


VARIANTS = [
    {"block_size": 12000, "window_16k": 8000, "hop_16k": 4000, "n_frames": 64},
    {"block_size": 24000, "window_16k": 16000, "hop_16k": 8000, "n_frames": 128},
    {"block_size": 48000, "window_16k": 32000, "hop_16k": 16000, "n_frames": 224},
]


def export_hubert_fixed(
    pt_path: Path,
    out_path: Path,
    window_16k: int,
    opset: int,
    device: str,
) -> None:
    _patch_fairseq_pad_to_multiple_for_fixed_export()
    from fairseq import checkpoint_utils

    print(f"[hubert] loading {pt_path}")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(pt_path)])
    model = models[0].to(device).eval()
    wrapped_model = HubertWrapper(model).eval()
    dummy = torch.zeros(1, window_16k, dtype=torch.float32, device=device)

    # DirectML compatibility is better with the legacy exporter path.
    hubert_opset = max(opset, 18)
    torch.onnx.export(
        wrapped_model,
        (dummy,),
        str(out_path),
        input_names=["source"],
        output_names=["features"],
        # Fixed shape export (no dynamic_axes)
        opset_version=hubert_opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[hubert] fixed window={window_16k} -> {out_path}")


def export_rmvpe_fixed(
    pt_path: Path,
    out_path: Path,
    n_frames: int,
    opset: int,
    device: str,
) -> None:
    try:
        from rmvpe.rmvpe import RMVPE
    except ImportError:
        import rmvpe as rmvpe_module

        RMVPE = rmvpe_module.RMVPE

    print(f"[rmvpe] loading {pt_path}")
    model = RMVPE(str(pt_path), is_half=False, device=device)
    model.model.eval()
    wrapper = RmvpeMelToF0(model.model, threshold=0.03).eval()
    dummy = torch.zeros(1, 128, n_frames, dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["mel"],
        output_names=["f0"],
        # Fixed shape export (no dynamic_axes)
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[rmvpe] fixed n_frames={n_frames} -> {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    _force_torch_load_weights_only_false()

    ap = argparse.ArgumentParser(
        description="Export fixed-shape HuBERT/RMVPE ONNX variants for Rust-VC"
    )
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--hubert", type=Path, default=Path("scripts/hubert_base.pt"))
    ap.add_argument("--rmvpe", type=Path, default=Path("scripts/rmvpe.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("model"))
    ap.add_argument("--hubert-opset", type=int, default=17)
    ap.add_argument("--rmvpe-opset", type=int, default=18)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    _add_scripts_to_path(project_root)

    hubert_pt = (project_root / args.hubert).resolve()
    rmvpe_pt = (project_root / args.rmvpe).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hubert_pt.exists():
        raise FileNotFoundError(f"HuBERT checkpoint not found: {hubert_pt}")
    if not rmvpe_pt.exists():
        raise FileNotFoundError(f"RMVPE checkpoint not found: {rmvpe_pt}")

    for variant in VARIANTS:
        bs = variant["block_size"]
        window_16k = variant["window_16k"]
        n_frames = variant["n_frames"]
        print(f"\n=== block_size={bs} ===")

        hubert_out = out_dir / f"hubert_b{bs}.onnx"
        rmvpe_out = out_dir / f"rmvpe_b{bs}.onnx"

        export_hubert_fixed(
            hubert_pt,
            hubert_out,
            window_16k=window_16k,
            opset=args.hubert_opset,
            device=args.device,
        )
        export_rmvpe_fixed(
            rmvpe_pt,
            rmvpe_out,
            n_frames=n_frames,
            opset=args.rmvpe_opset,
            device=args.device,
        )

    print("\n✅ 全バリアント出力完了")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\nエラー: 固定長ONNXの出力に失敗しました: {err}")
        traceback.print_exc()
        sys.exit(1)
