"""
Export HuBERT, RMVPE, and RVC Generator with dynamic axes.
Standalone Version: Depends ONLY on local infer_pack/ and rmvpe/ folders.
"""
import argparse
import sys
import os
import torch
import numpy as np

#  PyTorch 2.6 セキュリティ制限解除パッチ 
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

# scriptsフォルダ自身をPythonパスの先頭に追加
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ──────────────────────────────────────────
# 1. HuBERT
# ──────────────────────────────────────────
class HubertWrapper(torch.nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert
    def forward(self, x):
        return self.hubert(source=x, padding_mask=None, mask=False, features_only=True)["x"]

def export_hubert(pt_path: str, out_path: str):
    print(f"\n[hubert] loading {pt_path}")
    from fairseq import checkpoint_utils
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([pt_path])
    model = models[0].eval()
    wrapped_model = HubertWrapper(model)

    dummy = torch.zeros(1, 16000)
    torch.onnx.export(
        wrapped_model, (dummy,), out_path,
        input_names=["source"], output_names=["features"],
        dynamic_axes={"source": {0: "batch", 1: "n_samples"}, "features": {0: "batch", 1: "n_frames"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"[hubert] -> {out_path}")

# ──────────────────────────────────────────
# 2. RMVPE 
# ──────────────────────────────────────────
def export_rmvpe(pt_path: str, out_path: str):
    print(f"\n[rmvpe] loading {pt_path}")
    try:
        from rmvpe.rmvpe import RMVPE
    except ImportError:
        import rmvpe as rmvpe_module
        RMVPE = rmvpe_module.RMVPE

    model = RMVPE(pt_path, is_half=False, device="cpu")
    model.model.eval()

    # n_mels=128, n_frames=128 (32の倍数)
    dummy = torch.zeros(1, 128, 128)

    torch.onnx.export(
        model.model,
        (dummy,),
        out_path,
        input_names=["mel"],
        output_names=["f0"],
        dynamic_axes={
            "mel": {0: "batch", 2: "n_frames"}, # 軸2がフレーム
            "f0":  {0: "batch", 1: "n_frames"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,  
    )
    print(f"[rmvpe] -> {out_path}")
# ──────────────────────────────────────────
# 3. Generator
# ──────────────────────────────────────────
def resolve_generator_class(sd: dict):
    import sys
    import infer_pack
    sys.modules["lib"] = type("DummyLib", (), {})()
    sys.modules["lib.infer_pack"] = infer_pack
    from infer_pack import models as rvc_models

    if "enc_p.emb_phone.weight" in sd:
        hidden_dim = sd["enc_p.emb_phone.weight"].shape[1]
    else:
        hidden_dim = 256 # 安全のためのフォールバック

    version = "v2" if hidden_dim == 768 else "v1"
    
    # ピッチ(F0)の有無は、ピッチ埋め込み層(emb_pitch)の存在で確定させる
    has_f0 = any("emb_pitch" in k for k in sd.keys())

    name = {
        ("v1", True):  "SynthesizerTrnMs256NSFsid",
        ("v1", False): "SynthesizerTrnMs256NSFsid_nono",
        ("v2", True):  "SynthesizerTrnMs768NSFsid",
        ("v2", False): "SynthesizerTrnMs768NSFsid_nono",
    }.get((version, has_f0), "SynthesizerTrnMs768NSFsid")
    
    print(f"[generator] auto-selected class: {name} (Dim: {hidden_dim}, F0: {has_f0})")
    return getattr(rvc_models, name), version

def export_generator(pth_path: str, out_path: str, sr: int = 48000):
    print(f"\n[generator] loading {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    sd = ckpt.get("weight", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    Cls, version = resolve_generator_class(sd)
    
    cfg = ckpt.get("config", None)
    if cfg is None: raise RuntimeError("checkpointにconfigが見つかりません。")

    model = Cls(*cfg, is_half=False)
    model.load_state_dict(sd, strict=False)
    model.eval().remove_weight_norm()

    n_frames = 100
    hidden_dim = 768 if version == "v2" else 256
    
    # テンソルの準備
    phone   = torch.zeros(1, n_frames, hidden_dim)
    phone_l = torch.LongTensor([n_frames])
    pitch   = torch.zeros(1, n_frames, dtype=torch.long)
    pitchf  = torch.zeros(1, n_frames)
    sid     = torch.zeros(1, dtype=torch.long)
    rnd     = torch.zeros(1, 192, n_frames)
    ds      = torch.zeros(1, n_frames) 

    # 💥 ここがポイント！ argsを空のタプル () にして、kwargsで名前を明示して渡す！
    dummy_args = ()
    dummy_kwargs = {
        "phone": phone,
        "phone_lengths": phone_l,
        "pitch": pitch,
        "pitchf": pitchf,
        "sid": sid,
        "ds": ds,     
        "rnd": rnd
    }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # kwargsを使ってエクスポート
        torch.onnx.export(
            model,
            args=dummy_args,
            kwargs=dummy_kwargs,
            f=out_path,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "ds", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone":  {0: "batch", 1: "n_frames"},
                "pitch":  {0: "batch", 1: "n_frames"},
                "pitchf": {0: "batch", 1: "n_frames"},
                "ds":     {0: "batch", 1: "n_frames"},
                "rnd":    {0: "batch", 2: "n_frames"}, 
                "audio":  {0: "batch", 2: "n_samples"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False, 
        )
    print(f"[generator] -> {out_path}")

# ──────────────────────────────────────────
# main
# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubert",    default="scripts/hubert_base.pt")
    ap.add_argument("--rmvpe",     default="scripts/rmvpe.pt")
    ap.add_argument("--generator", default="scripts/rv_e250_s36750.pth")
    ap.add_argument("--out-dir",   default="model/dynamic")
    ap.add_argument("--sr",        type=int, default=48000)
    ap.add_argument("--verify",    action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    export_hubert(args.hubert, os.path.join(args.out_dir, "hubert_dynamic.onnx"))
    export_rmvpe(args.rmvpe, os.path.join(args.out_dir, "rmvpe_dynamic.onnx"))
    export_generator(args.generator, os.path.join(args.out_dir, "model_dynamic.onnx"), sr=args.sr)

    if args.verify:
        import onnxruntime as ort
        print("\n[verify] checking model_dynamic.onnx with variable length...")
        sess = ort.InferenceSession(os.path.join(args.out_dir, "model_dynamic.onnx"), providers=["CPUExecutionProvider"])
        hidden_dim = sess.get_inputs()[0].shape[2] 
        for n_frames in [50, 100, 200]:
            out = sess.run(None, {
                "phone":         np.zeros((1, n_frames, hidden_dim), dtype=np.float32),
                "phone_lengths": np.array([n_frames], dtype=np.int64),
                "pitch":         np.zeros((1, n_frames), dtype=np.int64),
                "pitchf":        np.zeros((1, n_frames), dtype=np.float32),
                "sid":           np.zeros((1,), dtype=np.int64),
                "rnd":           np.zeros((1, 192, n_frames), dtype=np.float32),
            })
            print(f"  n_frames={n_frames} -> audio.shape={out[0].shape}")
        print("\n✨ [verify] ALL OK")

if __name__ == "__main__":
    main()