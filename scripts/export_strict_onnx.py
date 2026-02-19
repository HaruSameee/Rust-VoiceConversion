#!/usr/bin/env python3
"""
Strict ONNX exporter for RVC front-end models (HuBERT / RMVPE).

Design goals:
- Engine-centric interface: model graph keeps raw audio I/O.
- Keep model-natural frame outputs (no forced Resize in ONNX graph).
- Rust runtime enforces downstream frame contract (e.g. target L=50).
- Opset >= 17 export by default.
- Built-in ONNXRuntime parity check against PyTorch wrapper output.

Usage examples:
  python scripts/export_strict_onnx.py \
    --kind hubert \
    --checkpoint ./hubert_base.pt \
    --output ./hubert_strict.onnx

  python scripts/export_strict_onnx.py \
    --kind rmvpe \
    --checkpoint ./rmvpe.pt \
    --output ./rmvpe_strict.onnx
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
except Exception:
    onnx = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


DEFAULT_INPUT_SAMPLES = 16_000
DEFAULT_HUBERT_FRAMES = 50
DEFAULT_RMVPE_FRAMES = 50
DEFAULT_OPSET = 17
RMVPE_SR = 16_000
RMVPE_N_FFT = 1_024
RMVPE_HOP = 160
RMVPE_WIN = 1_024
RMVPE_MELS = 128
RMVPE_CLASSES = 360


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class HubertStrictWrapper(nn.Module):
    """HuBERT wrapper with raw waveform input and natural output length."""

    def __init__(
        self,
        hubert_model: nn.Module,
        output_layer: int,
    ):
        super().__init__()
        self.hubert = hubert_model
        self.output_layer = int(output_layer)

    def _extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hasattr(self.hubert, "extract_features"):
            out = self.hubert.extract_features(
                source=source,
                padding_mask=padding_mask,
                output_layer=self.output_layer,
            )
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        out = self.hubert(
            source=source,
            padding_mask=padding_mask,
            features_only=True,
            output_layer=self.output_layer,
        )
        if isinstance(out, Mapping) and "x" in out:
            return out["x"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        source = audio.float()
        padding_mask = None
        feats = self._extract_features(source, padding_mask)
        if feats.dim() != 3:
            raise RuntimeError(f"HuBERT output must be [B,L,C], got {tuple(feats.shape)}")
        return feats


# ----------------------------- RMVPE core (minimal) -----------------------------


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] | None,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = int(n_encoders)
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_blocks,
                    momentum=momentum,
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            c, x = self.layers[i](x)
            concat_tensors.append(c)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_inters = int(n_inters)
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for _ in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = int(n_blocks)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: Tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = int(n_decoders)
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        concat_tensors = list(concat_tensors)
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            128,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel,
            en_de_layers,
            kernel_size,
            n_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: Tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * RMVPE_MELS, 256, n_gru),
                nn.Linear(512, RMVPE_CLASSES),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * RMVPE_MELS, RMVPE_CLASSES),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)  # [B,1,T,128]
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x  # [B,T,360]


def hz_to_mel_htk(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))


def mel_to_hz_htk(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filter_htk(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fmax = float(fmax if fmax is not None else (sr / 2.0))
    mel_min = hz_to_mel_htk(np.array([fmin], dtype=np.float64))[0]
    mel_max = hz_to_mel_htk(np.array([fmax], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz_htk(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int64)

    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])
        left = max(0, min(left, n_freq - 1))
        center = max(0, min(center, n_freq - 1))
        right = max(0, min(right, n_freq - 1))
        if center <= left:
            center = min(left + 1, n_freq - 1)
        if right <= center:
            right = min(center + 1, n_freq)
        if center > left:
            fb[m - 1, left:center] = (
                np.arange(left, center, dtype=np.float32) - float(left)
            ) / float(center - left)
        if right > center:
            fb[m - 1, center:right] = (
                float(right) - np.arange(center, right, dtype=np.float32)
            ) / float(right - center)
    return fb


class MelSpectrogramFixed(nn.Module):
    """RMVPE-compatible log-mel front-end with fixed params."""

    def __init__(
        self,
        n_mels: int = RMVPE_MELS,
        sampling_rate: int = RMVPE_SR,
        n_fft: int = RMVPE_N_FFT,
        win_length: int = RMVPE_WIN,
        hop_length: int = RMVPE_HOP,
        mel_fmin: float = 30.0,
        mel_fmax: float = 8000.0,
        clamp_min: float = 1e-5,
    ):
        super().__init__()
        mel_basis = mel_filter_htk(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("hann_window", torch.hann_window(win_length))
        n_freq = (n_fft // 2) + 1
        n = torch.arange(n_fft, dtype=torch.float32).view(1, n_fft)
        k = torch.arange(n_freq, dtype=torch.float32).view(n_freq, 1)
        phase = (2.0 * math.pi / float(n_fft)) * (k * n)
        # Real-valued RFFT basis so ONNX export avoids complex dtype ops.
        self.register_buffer("rfft_cos", torch.cos(phase).transpose(0, 1).contiguous())
        self.register_buffer("rfft_sin", -torch.sin(phase).transpose(0, 1).contiguous())
        frame_kernel = torch.eye(win_length, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("frame_kernel", frame_kernel)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.clamp_min = float(clamp_min)
        self.pad = int(n_fft // 2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.float()
        x = F.pad(x.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        frames = F.conv1d(x, self.frame_kernel, stride=self.hop_length).transpose(1, 2)
        frames = frames * self.hann_window.view(1, 1, self.win_length)
        if self.n_fft > self.win_length:
            frames = F.pad(frames, (0, self.n_fft - self.win_length))

        real = torch.matmul(frames, self.rfft_cos)
        imag = torch.matmul(frames, self.rfft_sin)
        mag = torch.sqrt(real.pow(2) + imag.pow(2))

        mel = torch.matmul(mag, self.mel_basis.transpose(0, 1)).transpose(1, 2)
        return torch.log(torch.clamp(mel, min=self.clamp_min))


class RmvpeStrictWrapper(nn.Module):
    """RMVPE wrapper that accepts raw waveform and returns natural output length."""

    def __init__(self, rmvpe_core: nn.Module):
        super().__init__()
        self.mel = MelSpectrogramFixed()
        self.core = rmvpe_core

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.float()
        mel = self.mel(x)  # [B,128,T]
        n_frames = mel.size(-1)
        pad_right = (32 - (n_frames % 32)) % 32
        mel = F.pad(mel, (0, pad_right), mode="reflect")
        salience = self.core(mel)[:, :n_frames]  # [B,T,360]
        return salience


# ----------------------------- build / load helpers -----------------------------


def unwrap_state_dict(obj: object) -> MutableMapping[str, torch.Tensor]:
    if isinstance(obj, OrderedDict):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], (dict, OrderedDict)):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], (dict, OrderedDict)):
            return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # already a plain state_dict
    raise RuntimeError(
        "unsupported checkpoint format for RMVPE. "
        "expected state_dict or dict containing `state_dict`/`model`."
    )


def load_rmvpe_wrapper(checkpoint_path: Path) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    model = E2E(4, 1, (2, 2))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"RMVPE checkpoint missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        raise RuntimeError(
            f"RMVPE checkpoint has unexpected keys ({len(unexpected)}): {unexpected[:8]}"
        )
    model.eval()
    return RmvpeStrictWrapper(model).eval()


def load_hubert_wrapper(
    checkpoint_path: Path,
    output_layer: int,
) -> nn.Module:
    try:
        from fairseq import checkpoint_utils
        from fairseq.models.wav2vec import utils as wav2vec_utils
        from fairseq.models.wav2vec import wav2vec2 as wav2vec2_mod
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "fairseq is required for HuBERT export. Install it in your python environment."
        ) from e

    def _pad_to_multiple_export_safe(
        x: torch.Tensor,
        multiple: int,
        dim: int = -1,
        value: float = 0.0,
    ):
        tsz = int(x.size(dim))
        remainder = (int(multiple) - (tsz % int(multiple))) % int(multiple)
        if remainder == 0:
            return x, 0
        pad_offset = (0,) * ((-1 - dim) * 2)
        return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

    wav2vec_utils.pad_to_multiple = _pad_to_multiple_export_safe
    wav2vec2_mod.pad_to_multiple = _pad_to_multiple_export_safe

    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat
    try:
        models, _saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint_path)],
            arg_overrides={"data": str(checkpoint_path.parent)},
        )
    finally:
        torch.load = original_torch_load

    if not models:
        raise RuntimeError("fairseq returned no HuBERT model instances")
    hubert = models[0].eval()

    return HubertStrictWrapper(
        hubert_model=hubert,
        output_layer=output_layer,
    ).eval()


def build_model(args: argparse.Namespace) -> Tuple[nn.Module, str]:
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    kind = args.kind.lower()
    if kind == "hubert":
        return (
            load_hubert_wrapper(
                checkpoint_path=ckpt,
                output_layer=args.hubert_layer,
            ),
            "phone",
        )
    if kind == "rmvpe":
        return (load_rmvpe_wrapper(ckpt), "salience")
    raise ValueError(f"unsupported --kind: {args.kind}")


def export_onnx(
    model: nn.Module,
    output_name: str,
    output_path: Path,
    input_samples: int,
    batch: int,
    opset: int,
    dynamic_time: bool,
    verbose: bool,
) -> torch.Tensor:
    dummy = torch.randn(batch, input_samples, dtype=torch.float32)
    dynamic_axes: Dict[str, Dict[int, str]] = {
        "audio": {0: "batch"},
        output_name: {0: "batch"},
    }
    if dynamic_time:
        dynamic_axes["audio"][1] = "samples"
        dynamic_axes[output_name][1] = "frames"

    with torch.inference_mode():
        torch_out = model(dummy)
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["audio"],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            verbose=verbose,
        )
    return torch_out


def verify_opset(onnx_path: Path, min_opset: int) -> None:
    if onnx is None:
        print("[warn] onnx package not installed; skip opset verification")
        return
    model = onnx.load(str(onnx_path))
    versions = [imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")]
    if not versions:
        raise RuntimeError("cannot find ai.onnx opset import in exported model")
    current = max(versions)
    if current < min_opset:
        raise RuntimeError(f"exported opset {current} < required {min_opset}")


def verify_parity(
    onnx_path: Path,
    output_name: str,
    audio: torch.Tensor,
    torch_output: torch.Tensor,
    atol: float,
) -> None:
    if ort is None:
        raise RuntimeError("onnxruntime is required for parity verification")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    out = sess.run([output_name], {input_name: audio.cpu().numpy().astype(np.float32)})[0]

    ref = torch_output.detach().cpu().numpy()
    if out.shape != ref.shape:
        raise RuntimeError(f"shape mismatch: ONNX {out.shape} vs Torch {ref.shape}")
    max_abs = float(np.max(np.abs(out - ref)))
    print(f"[verify] max_abs_error={max_abs:.8e} (atol={atol:.1e})")
    if max_abs > atol:
        raise RuntimeError(f"parity check failed: {max_abs:.8e} > {atol:.1e}")


def _reflect_left_pad_indices(pad: int, src_len: int) -> np.ndarray:
    if src_len <= 1:
        return np.zeros((pad,), dtype=np.int64)
    period = 2 * (src_len - 1)
    dists = np.arange(pad, 0, -1, dtype=np.int64)
    m = dists % period
    m[m == 0] = period
    left = np.where(m <= (src_len - 1), m, period - m)
    return left.astype(np.int64, copy=False)


def fit_waveform_to_len_rust_sync(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    """Mirror Rust `fit_waveform_to_len`: tail-crop, or left reflect-pad."""
    if audio.dim() != 2:
        raise ValueError(f"audio must be [B,T], got {tuple(audio.shape)}")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    bsz, t = audio.shape
    if t == target_len:
        return audio
    if t > target_len:
        start = t - target_len
        return audio[:, start : start + target_len]
    if t == 0:
        return torch.zeros((bsz, target_len), dtype=audio.dtype)
    if t == 1:
        return audio.repeat(1, target_len)

    pad = target_len - t
    src = audio.detach().cpu().numpy()
    left_idx = _reflect_left_pad_indices(pad, t)
    out = np.zeros((bsz, target_len), dtype=np.float32)
    out[:, :pad] = src[:, left_idx]
    out[:, pad:] = src
    return torch.from_numpy(out)


def normalize_peak_rust_sync(
    audio: torch.Tensor,
    target_peak: float = 0.95,
    min_peak: float = 1.0e-6,
) -> torch.Tensor:
    """Mirror Rust `normalize_peak`: per-batch peak gain with clamp [0.25, 32]."""
    if audio.dim() != 2:
        raise ValueError(f"audio must be [B,T], got {tuple(audio.shape)}")
    x = audio.detach().cpu().numpy().astype(np.float32, copy=True)
    abs_max = np.max(np.abs(x), axis=1)
    gains = np.ones_like(abs_max, dtype=np.float32)
    active = abs_max > max(min_peak, 1.0e-12)
    gains[active] = np.clip(target_peak / abs_max[active], 0.25, 32.0)
    x *= gains[:, None]
    return torch.from_numpy(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export strict RVC feature models to ONNX.")
    p.add_argument("--kind", choices=["hubert", "rmvpe"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to hubert_base.pt or rmvpe.pt")
    p.add_argument("--output", required=True, help="Output ONNX path")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version (>=17)")
    p.add_argument(
        "--input-samples",
        type=int,
        default=DEFAULT_INPUT_SAMPLES,
        help="Fixed input waveform length for export (default: 16000)",
    )
    p.add_argument("--batch", type=int, default=1, help="Dummy export batch size")
    p.add_argument("--hubert-layer", type=int, default=12, help="HuBERT output layer index")
    p.add_argument(
        "--hubert-frames",
        type=int,
        default=DEFAULT_HUBERT_FRAMES,
        help="Target HuBERT frame count expected by Rust-side post-shape logic",
    )
    p.add_argument(
        "--rmvpe-frames",
        type=int,
        default=DEFAULT_RMVPE_FRAMES,
        help="Target RMVPE frame count expected by Rust-side post-shape logic",
    )
    p.add_argument(
        "--dynamic-time",
        action="store_true",
        help="Allow dynamic time axis. Keep disabled for strict fixed-shape interface.",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Parity tolerance between PyTorch and ONNX outputs",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNXRuntime parity verification step.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset < 17:
        raise ValueError("--opset must be >= 17")
    if args.input_samples <= 0:
        raise ValueError("--input-samples must be positive")
    if onnx is None:
        raise RuntimeError(
            "onnx package is required for export in this environment. "
            "Install with: pip install onnx"
        )
    if not args.skip_verify and ort is None:
        raise RuntimeError(
            "onnxruntime is required for parity verification. "
            "Install with: pip install onnxruntime"
        )

    seed_everything(args.seed)
    model, output_name = build_model(args)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch_out = export_onnx(
        model=model,
        output_name=output_name,
        output_path=output_path,
        input_samples=args.input_samples,
        batch=args.batch,
        opset=args.opset,
        dynamic_time=args.dynamic_time,
        verbose=args.verbose,
    )

    # Recompute on a deterministic probe for parity test.
    # Rust-sync mode:
    # 1) Start from a longer waveform chunk.
    # 2) Fit to strict input length with tail-crop / reflect-left-pad.
    # 3) Apply the same peak normalization policy as Rust.
    verify_probe_len = args.input_samples + max(1, args.input_samples // 4)
    probe_raw = torch.randn(args.batch, verify_probe_len, dtype=torch.float32)
    probe = fit_waveform_to_len_rust_sync(probe_raw, args.input_samples)
    probe = normalize_peak_rust_sync(probe)
    with torch.inference_mode():
        probe_out = model(probe)

    verify_opset(output_path, args.opset)
    if args.skip_verify:
        print("[warn] parity verification skipped (--skip-verify).")
    else:
        verify_parity(
            onnx_path=output_path,
            output_name=output_name,
            audio=probe,
            torch_output=probe_out,
            atol=args.atol,
        )

    if args.kind == "hubert" and args.input_samples == 16_000 and not args.dynamic_time:
        if probe_out.shape[1] <= 0:
            raise RuntimeError(
                f"invalid HuBERT output frame count: L={probe_out.shape[1]}"
            )
        print(
            "[contract] HuBERT T=16000 -> natural L="
            f"{probe_out.shape[1]} (Rust enforces target={args.hubert_frames})."
        )
    if args.kind == "rmvpe" and args.input_samples == 16_000 and not args.dynamic_time:
        if probe_out.shape[1] <= 0:
            raise RuntimeError(
                f"invalid RMVPE output frame count: L={probe_out.shape[1]}"
            )
        print(
            "[contract] RMVPE T=16000 -> natural L="
            f"{probe_out.shape[1]} (Rust enforces target={args.rmvpe_frames})."
        )

    print(f"[ok] exported {args.kind} ONNX: {output_path}")


if __name__ == "__main__":
    main()
