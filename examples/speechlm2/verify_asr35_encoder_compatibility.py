#!/usr/bin/env python
"""
Phase-1 G0/G1 pre-check for replacing the VoiceChat perception encoder with
``nvidia/nemotron-3.5-asr-streaming-0.6b``.

Run INSIDE the NIM inference container (see voicechat_realtime_instructions/
deploy-asr35-zh.md) so the nemo package used for the check is exactly the one
the runtime uses. The script only depends on the installed nemo package plus
torch / safetensors / omegaconf — it never imports code from the Speech repo.

Modes
-----
G0 (model sanity):
    --asr <hf-name-or-.nemo> [--save-nemo out.nemo] [--probe-wav zh.wav]
    Downloads / restores the ASR 3.5 model, asserts d_model == 1024 and a
    causal att_context_size, optionally transcribes a probe WAV, optionally
    saves a local .nemo.

G1 (compatibility pre-check, requires --voicechat-checkpoint-dir):
    1. Key alignment: ASR ``encoder.*`` / ``preprocessor.*`` tensors vs the
       checkpoint's ``stt_model.perception.encoder.*`` /
       ``stt_model.perception.preprocessor.*`` tensors (presence + shape).
    2. Latent similarity (requires --audio-dir with en_*.wav / zh_*.wav):
       rebuild the OLD encoder from the checkpoint, run the same audio through
       both encoders, report per-file / per-language cosine similarity.

Interpretation
--------------
    english mean cosine > 0.9  -> latent geometry preserved, zero-shot worth
                                  trying directly (proceed to G2).
    english mean cosine < 0.7  -> latent drift is large; expect scenario B/C
                                  (adapter/projection tuning), still proceed
                                  if you want the empirical data point.
"""

import argparse
import json
import os
import sys

import torch


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_asr(asr_arg: str, save_nemo: str = None):
    from nemo.collections.asr.models import ASRModel

    if asr_arg.endswith(".nemo") and os.path.isfile(asr_arg):
        print(f"[load] restoring ASR model from {asr_arg}")
        asr = ASRModel.restore_from(asr_arg, map_location="cpu")
    else:
        print(f"[load] downloading ASR model {asr_arg} (needs network; run with HF_HUB_OFFLINE=0)")
        asr = ASRModel.from_pretrained(asr_arg)

    if save_nemo:
        os.makedirs(os.path.dirname(os.path.abspath(save_nemo)) or ".", exist_ok=True)
        asr.save_to(save_nemo)
        print(f"[load] saved local .nemo to {save_nemo}")
    return asr


def hard_assertions(asr, force: bool = False) -> dict:
    """Hard requirements from the phase-1 plan. Fail-fast unless --force."""
    enc_cfg = asr.cfg.encoder
    d_model = enc_cfg.get("d_model", None)
    att = list(enc_cfg.get("att_context_size", []) or [])
    print(f"[assert] encoder d_model     = {d_model}")
    print(f"[assert] att_context_size    = {att}")

    failures = []
    if d_model != 1024:
        failures.append(f"d_model={d_model} != 1024 (modality_adapter / RNNT joint expect 1024)")
    if not att or len(att) < 2 or att[1] != 0:
        failures.append(f"att_context_size={att} is not causal (right context must be 0)")
    if list(att) != [56, 0]:
        print(f"[assert] WARNING: att_context_size {att} != [56, 0] (plan expects 80ms streaming [56,0])")

    if failures:
        for f in failures:
            eprint(f"[assert] FAIL: {f}")
        if not force:
            eprint("[assert] hard requirement violated — aborting (use --force to continue anyway)")
            sys.exit(2)
    else:
        print("[assert] OK: d_model=1024, causal att_context_size")
    return {"d_model": d_model, "att_context_size": att}


def transcribe_probe(asr, wav_path: str, device: str):
    try:
        asr.eval()
        asr.to(device)
        out = asr.transcribe([wav_path])
        hyps = out[0] if isinstance(out, (list, tuple)) else out
        text = None
        if isinstance(hyps, (list, tuple)) and hyps:
            h = hyps[0]
            text = getattr(h, "text", None) or (h.get("text") if isinstance(h, dict) else str(h))
        elif isinstance(hyps, dict):
            text = hyps.get("text")
        print(f"[probe] {os.path.basename(wav_path)} -> {text!r}")
        return text
    except Exception as e:  # transcription API varies across nemo versions
        eprint(f"[probe] WARNING: transcribe failed ({e}); G0 transcript check skipped")
        return None


# ---------------------------------------------------------------------------
# Checkpoint side
# ---------------------------------------------------------------------------


def load_full_state_dict(ckpt_dir: str) -> dict:
    from safetensors.torch import load_file

    main = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(main):
        return load_file(main)
    # defensive: sharded checkpoints
    import glob

    shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no model.safetensors (or shards) found in {ckpt_dir}")
    sd = {}
    for s in shards:
        print(f"[ckpt] loading shard {os.path.basename(s)}")
        sd.update(load_file(s))
    return sd


def build_old_perception(ckpt_dir: str, device: str):
    """Rebuild the OLD encoder (preprocessor + encoder) from the checkpoint,
    mirroring the container's load_perception() logic."""
    from nemo.collections.speechlm2.modules import AudioPerceptionModule
    from omegaconf import OmegaConf, open_dict
    from safetensors.torch import load_file

    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path, "r") as f:
        full_cfg = json.load(f)
    per = OmegaConf.create(full_cfg["model"]["stt"]["model"]["perception"])
    with open_dict(per):
        per.output_dim = 4480  # same hardcode as container load_perception()

    module = AudioPerceptionModule(per)
    prefix = "stt_model.perception."
    sd_path = os.path.join(ckpt_dir, "model.safetensors")
    raw = load_file(sd_path) if os.path.isfile(sd_path) else load_full_state_dict(ckpt_dir)
    sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    n_enc_missing = sum(1 for k in missing if k.startswith(("encoder.", "preprocessor.")))
    print(f"[old-encoder] perception keys loaded={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
    if n_enc_missing:
        eprint(f"[old-encoder] FAIL: {n_enc_missing} encoder/preprocessor keys missing from checkpoint")
        sys.exit(2)
    module.eval().to(device)
    return module.preprocessor, module.encoder


# ---------------------------------------------------------------------------
# Key alignment (structure level)
# ---------------------------------------------------------------------------


def key_alignment(asr, vc_sd: dict) -> dict:
    asr_sd = asr.state_dict()
    prefixes = ("encoder.", "preprocessor.")
    new_keys = {k: tuple(v.shape) for k, v in asr_sd.items() if k.startswith(prefixes)}
    vc_prefix = "stt_model.perception."
    old_keys = {
        k[len(vc_prefix):]: tuple(v.shape)
        for k, v in vc_sd.items()
        if k.startswith(tuple(vc_prefix + p for p in prefixes))
    }

    exact, shape_diff, new_only, old_only = [], [], [], []
    for k, shape in sorted(new_keys.items()):
        if k in old_keys:
            if old_keys[k] == shape:
                exact.append(k)
            else:
                shape_diff.append((k, old_keys[k], shape))
        else:
            new_only.append(k)
    for k in sorted(old_keys):
        if k not in new_keys:
            old_only.append(k)

    print("\n===== key alignment (ASR encoder.* / preprocessor.* vs checkpoint) =====")
    print(f"  exact match      : {len(exact)}")
    print(f"  shape mismatch   : {len(shape_diff)}")
    print(f"  new-model-only   : {len(new_only)}   (fine: runtime builds module from NEW config)")
    print(f"  checkpoint-only  : {len(old_only)}   (dropped at G2)")
    for k, old_s, new_s in shape_diff[:10]:
        print(f"    MISMATCH {k}: checkpoint{old_s} vs asr{new_s}")
    for k in new_only[:10]:
        print(f"    NEW-ONLY {k} {new_keys[k]}")
    for k in old_only[:10]:
        print(f"    OLD-ONLY {k} {old_keys[k]}")

    if not exact:
        eprint("[align] FAIL: zero overlapping encoder keys — wrong model or wrong prefixes")
        sys.exit(2)
    return {
        "exact": len(exact),
        "shape_mismatch": len(shape_diff),
        "new_only": new_only,
        "old_only": old_only,
        "shape_mismatch_detail": [(k, list(a), list(b)) for k, a, b in shape_diff[:50]],
    }


# ---------------------------------------------------------------------------
# Latent comparison
# ---------------------------------------------------------------------------


def load_wav_16k(path: str) -> torch.Tensor:
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav
    except ImportError:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        t = torch.from_numpy(data).unsqueeze(0)
        if sr != 16000:
            eprint(f"[audio] WARNING: {path} is {sr}Hz and torchaudio unavailable — cannot resample")
        return t


@torch.no_grad()
def encode_side(preprocessor, encoder, wav: torch.Tensor, device: str):
    wav = wav.to(device)
    lens = torch.tensor([wav.shape[-1]], device=device)
    spec, spec_len = preprocessor(input_signal=wav, length=lens)
    emb, emb_len = encoder(audio_signal=spec, length=spec_len)
    return emb[0].float().cpu(), int(emb_len[0])


def latent_comparison(asr, ckpt_dir: str, audio_dir: str, device: str) -> dict:
    old_pre, old_enc = build_old_perception(ckpt_dir, device)
    new_pre, new_enc = asr.preprocessor.eval().to(device), asr.encoder.eval().to(device)

    wavs = sorted(
        f for f in os.listdir(audio_dir) if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3"))
    )
    if not wavs:
        eprint(f"[latent] WARNING: no audio files in {audio_dir} — latent comparison skipped")
        return {}

    per_file, per_lang = [], {}
    print("\n===== latent similarity (old encoder vs ASR 3.5 encoder, frame-wise cosine) =====")
    for fname in wavs:
        wav = load_wav_16k(os.path.join(audio_dir, fname))
        old_emb, old_len = encode_side(old_pre, old_enc, wav, device)
        new_emb, new_len = encode_side(new_pre, new_enc, wav, device)
        T = min(old_len, new_len)
        if abs(old_len - new_len) > 1:
            print(f"  WARNING {fname}: frame counts differ (old={old_len}, new={new_len})")
        a, b = old_emb[:T], new_emb[:T]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        lang = "zh" if "zh" in fname.lower() else ("en" if "en" in fname.lower() else "other")
        rec = {
            "file": fname,
            "lang": lang,
            "frames": int(T),
            "cos_mean": round(float(cos.mean()), 4),
            "cos_std": round(float(cos.std()), 4),
            "old_mean_abs": round(float(a.abs().mean()), 4),
            "new_mean_abs": round(float(b.abs().mean()), 4),
        }
        per_file.append(rec)
        per_lang.setdefault(lang, []).append(rec["cos_mean"])
        print(
            f"  {fname:<28} [{lang}] frames={T:<5} cos={rec['cos_mean']:.4f}"
            f"±{rec['cos_std']:.4f}  |old|={rec['old_mean_abs']:.3f} |new|={rec['new_mean_abs']:.3f}"
        )

    summary = {lang: round(sum(v) / len(v), 4) for lang, v in per_lang.items()}
    print(f"\n  per-language mean cosine: {summary}")
    if "en" in summary:
        if summary["en"] > 0.9:
            print("  -> EN latent geometry preserved: zero-shot replacement is worth trying directly (G2)")
        elif summary["en"] < 0.7:
            print("  -> EN latent drifted a lot: expect scenario B/C (adapter/projection tuning)")
        else:
            print("  -> EN latent partially preserved: outcome uncertain — run G2/G4 for the empirical answer")
    return {"per_file": per_file, "per_language_mean": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asr", required=True, help="HF model name (nvidia/nemotron-3.5-asr-streaming-0.6b) or local .nemo path")
    p.add_argument("--save-nemo", default=None, help="G0: save the downloaded model to this .nemo path")
    p.add_argument("--probe-wav", default=None, help="G0: transcribe this WAV to sanity-check Chinese ASR")
    p.add_argument("--voicechat-checkpoint-dir", default=None, help="G1: official VoiceChat checkpoint dir (config.json + model.safetensors)")
    p.add_argument("--audio-dir", default=None, help="G1: dir with en_*.wav / zh_*.wav for latent comparison")
    p.add_argument("--report", default=None, help="write JSON report to this path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="continue even if hard assertions fail")
    args = p.parse_args()

    print(f"[env] torch={torch.__version__} device={args.device}")
    import nemo

    print(f"[env] nemo={getattr(nemo, '__version__', 'unknown')}")

    asr = load_asr(args.asr, args.save_nemo)
    facts = hard_assertions(asr, force=args.force)
    report = {"asr": args.asr, "encoder_facts": facts}

    if args.probe_wav:
        report["probe"] = transcribe_probe(asr, args.probe_wav, args.device)

    if args.voicechat_checkpoint_dir:
        vc_sd = load_full_state_dict(args.voicechat_checkpoint_dir)
        report["key_alignment"] = key_alignment(asr, vc_sd)
        del vc_sd
        if args.audio_dir:
            report["latent"] = latent_comparison(asr, args.voicechat_checkpoint_dir, args.audio_dir, args.device)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[report] written to {args.report}")
    print("[done] G0/G1 pre-check complete")


if __name__ == "__main__":
    main()
