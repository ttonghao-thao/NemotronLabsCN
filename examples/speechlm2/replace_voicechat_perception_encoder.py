#!/usr/bin/env python
"""
Phase-1 G2: solidify the ASR 3.5 perception encoder into a VoiceChat checkpoint.

Produces a self-contained Chinese-capable checkpoint directory that drops into
the official deployment flow unchanged (deploy_s2s_model.sh -> run_s2s_server.sh):

    input : official NVIDIA-NemotronLabs-VoiceChat-11B checkpoint dir
            (config.json + model.safetensors [+ rnnt_tokenizer/])
            + local nemotron-3.5-asr-streaming-0.6b .nemo
    output: NVIDIA-NemotronLabs-VoiceChat-11B-ZH style checkpoint dir

What it does
------------
1. Weights (model.safetensors):
   - DROP all stt_model.perception.encoder.* / stt_model.perception.preprocessor.*
   - ADD   asr encoder.*        -> stt_model.perception.encoder.*
           asr preprocessor.*   -> stt_model.perception.preprocessor.*
   Untouched: stt_model.perception.modality_adapter.* / proj.*, rnnt_decoder/joint,
   llm, tts_model, embed_tokens, function head.
2. config.json:
   - model.stt.model.perception.encoder    <- ASR 3.5 cfg.encoder   (att_context_size=[56,0])
   - model.stt.model.perception.preprocessor <- ASR 3.5 cfg.preprocessor
   Untouched: modality_adapter / output_dim / _rnnt_merge_info / pretrained_asr / everything else.
   (The container's deploy step later forces pretrained_asr=None / pretrained_weights=False and
   rebuilds the perception module from this config — see checkpoint_utils/import_utils.py.)

Runtime compatibility contract (asserted before writing):
   - encoder d_model == 1024, att_context_size causal (right context 0)

Run INSIDE the NIM inference container (see voicechat_realtime_instructions/
deploy-asr35-zh.md). Depends only on installed nemo + torch + safetensors + omegaconf.
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


PERCEPTION_PREFIX = "stt_model.perception."
REPLACE_PREFIXES = ("encoder.", "preprocessor.")
VC_REPLACE_PREFIXES = tuple(PERCEPTION_PREFIX + p for p in REPLACE_PREFIXES)

ASR35_MODEL_MODULE = "nemo.collections.asr.models.rnnt_bpe_models_prompt"
ASR35_MODEL_CLASS = "EncDecRNNTBPEModelWithPrompt"


def ensure_asr35_model_class():
    """Make ASR 3.5 .nemo checkpoints restorable on nemo builds that lack
    ``nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt``
    (e.g. the NIM image's nemo 2.4.0rc0): register a thin subclass of
    EncDecRNNTBPEModel under that exact module path. G2 only touches cfg /
    encoder.* / preprocessor.*, which the base class serves fine; extra
    prompt/streaming keys in cfg are ignored. No-op on newer nemo that ships
    the real class. (Kept in sync with verify_asr35_encoder_compatibility.py —
    both scripts are standalone by design.)
    """
    import importlib
    import sys
    import types

    try:
        importlib.import_module(ASR35_MODEL_MODULE)  # real class already present
        return
    except ModuleNotFoundError:
        pass

    import nemo.collections.asr.models as models_pkg
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    class _WithPrompt(EncDecRNNTBPEModel):
        """Stand-in for the ASR 3.5 prompt-RNNT model on older nemo builds.

        The real model carries a language-prompt conditioning stack
        (prompt_kernel.*, fed by cfg's prompt_dictionary / num_prompts) that
        EncDecRNNTBPEModel does not define. G2 only uses cfg and encoder.* /
        preprocessor.* — and the VoiceChat runtime never consumes
        prompt_kernel — so those tensors are tolerated as unexpected keys and
        dropped instead of failing the strict restore. Anything else missing
        or unexpected still raises. (Kept in sync with
        verify_asr35_encoder_compatibility.py.)
        """

        def load_state_dict(self, state_dict, strict=True, assign=False):
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            missing = list(result.missing_keys)
            tolerated = sorted(k for k in result.unexpected_keys if k.startswith("prompt_kernel."))
            bad_unexpected = sorted(k for k in result.unexpected_keys if not k.startswith("prompt_kernel."))
            if strict and (missing or bad_unexpected):
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(self).__name__}:\n"
                    f"\tMissing key(s): {missing}\n\tUnexpected key(s): {bad_unexpected}"
                )
            if tolerated:
                print(
                    f"[shim] dropped {len(tolerated)} prompt_kernel.* tensors "
                    f"(language-prompt conditioning, unused by G0-G2): {tolerated}"
                )
            return result

    _WithPrompt.__name__ = ASR35_MODEL_CLASS
    _WithPrompt.__qualname__ = ASR35_MODEL_CLASS
    _WithPrompt.__module__ = ASR35_MODEL_MODULE

    shim = types.ModuleType(ASR35_MODEL_MODULE)
    shim.EncDecRNNTBPEModelWithPrompt = _WithPrompt
    sys.modules[ASR35_MODEL_MODULE] = shim
    setattr(models_pkg, "rnnt_bpe_models_prompt", shim)
    print(
        f"[shim] nemo lacks {ASR35_MODEL_MODULE}; registered "
        f"{ASR35_MODEL_CLASS} as an EncDecRNNTBPEModel subclass"
    )


def dump_nemo_targets(nemo_path: str):
    """Diagnostic for a still-failing restore: list every _target_ inside the
    .nemo config so the output names exactly which other classes need shimming."""
    import tarfile

    print(f"[diag] _target_ entries inside {nemo_path}:")
    try:
        with tarfile.open(nemo_path, "r:*") as tar:
            for m in tar.getmembers():
                if not m.name.endswith((".yaml", ".json")):
                    continue
                data = tar.extractfile(m).read().decode("utf-8", "replace")
                for line in data.splitlines():
                    s = line.strip()
                    if "_target_" in s or s.startswith("target:"):
                        print(f"[diag]   {m.name}: {s}")
    except Exception as e:  # diagnostic must never mask the original error
        print(f"[diag] could not read archive ({e})")


def cfg_to_dict(cfg) -> dict:
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return OmegaConf.to_container(cfg, resolve=False)


def load_full_state_dict(ckpt_dir: str) -> dict:
    from safetensors.torch import load_file

    main = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(main):
        return load_file(main)
    import glob

    shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no model.safetensors (or shards) found in {ckpt_dir}")
    sd = {}
    for s in shards:
        print(f"  loading shard {os.path.basename(s)}")
        sd.update(load_file(s))
    return sd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voicechat-checkpoint-dir", required=True, help="official VoiceChat checkpoint dir (config.json + model.safetensors)")
    p.add_argument("--asr-nemo", required=True, help="local nemotron-3.5-asr-streaming-0.6b .nemo produced by G0")
    p.add_argument("--output-dir", required=True, help="output checkpoint dir (e.g. zh-checkpoint/)")
    p.add_argument("--overwrite", action="store_true", help="replace output dir if it exists")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    p.add_argument("--force", action="store_true", help="continue past hard-assertion failures")
    args = p.parse_args()

    t0 = time.time()
    ckpt_dir = args.voicechat_checkpoint_dir
    for f in ("model.safetensors", "config.json"):
        if not os.path.isfile(os.path.join(ckpt_dir, f)):
            eprint(f"ERROR: {f} not found in {ckpt_dir}")
            sys.exit(2)
    if not os.path.isfile(args.asr_nemo):
        eprint(f"ERROR: {args.asr_nemo} not found (run G0 first)")
        sys.exit(2)

    if os.path.exists(args.output_dir):
        if args.overwrite:
            print(f"[prep] removing existing output dir {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            eprint(f"ERROR: {args.output_dir} exists — use --overwrite")
            sys.exit(2)

    # ------------------------------------------------------------------ ASR 3.5
    print(f"[1/5] restoring ASR 3.5 from {args.asr_nemo}")
    from nemo.collections.asr.models import ASRModel

    ensure_asr35_model_class()
    try:
        asr = ASRModel.restore_from(args.asr_nemo, map_location="cpu").eval()
    except Exception:
        dump_nemo_targets(args.asr_nemo)
        raise

    enc_cfg = asr.cfg.encoder
    d_model = enc_cfg.get("d_model", None)
    att = list(enc_cfg.get("att_context_size", []) or [])
    print(f"      encoder d_model={d_model} att_context_size={att}")
    failures = []
    if d_model != 1024:
        failures.append(f"d_model={d_model} != 1024")
    if not att or len(att) < 2 or att[1] != 0:
        failures.append(f"att_context_size={att} not causal")
    if failures:
        for f in failures:
            eprint(f"      FAIL: {f}")
        if not args.force:
            sys.exit(2)

    asr_sd = asr.state_dict()
    new_tensors = {k: v for k, v in asr_sd.items() if k.startswith(REPLACE_PREFIXES)}
    n_enc = sum(1 for k in new_tensors if k.startswith("encoder."))
    n_pre = sum(1 for k in new_tensors if k.startswith("preprocessor."))
    print(f"      ASR tensors to inject: encoder={n_enc} preprocessor={n_pre}")
    if n_enc == 0:
        eprint("ERROR: no encoder.* tensors in ASR state dict — wrong checkpoint?")
        sys.exit(2)
    skipped = [k for k in asr_sd if not k.startswith(REPLACE_PREFIXES)]
    print(f"      ASR tensors skipped (RNNT/lang-ID/etc.): {len(skipped)}")

    # ------------------------------------------------------------- checkpoint
    print(f"[2/5] loading VoiceChat checkpoint from {ckpt_dir}")
    vc_sd = load_full_state_dict(ckpt_dir)
    print(f"      total tensors: {len(vc_sd)}")

    old_replace_keys = [k for k in vc_sd if k.startswith(VC_REPLACE_PREFIXES)]
    kept_keys = [k for k in vc_sd if not k.startswith(VC_REPLACE_PREFIXES)]
    kept_adapter = sum(1 for k in kept_keys if k.startswith(PERCEPTION_PREFIX + "modality_adapter.") or k.startswith(PERCEPTION_PREFIX + "proj."))
    print(f"      dropping old encoder/preprocessor tensors: {len(old_replace_keys)}")
    print(f"      keeping (incl. modality_adapter/proj={kept_adapter}): {len(kept_keys)}")

    # information-only diff vs OLD checkpoint keys (runtime module follows the NEW config)
    old_norm = {k[len(PERCEPTION_PREFIX):]: tuple(v.shape) for k, v in vc_sd.items() if k.startswith(VC_REPLACE_PREFIXES)}
    exact = sum(1 for k, s in new_tensors.items() if k in old_norm and tuple(old_norm[k]) == tuple(s.shape))
    shape_diff = [(k, old_norm[k], tuple(s.shape)) for k, s in new_tensors.items() if k in old_norm and tuple(old_norm[k]) != tuple(s.shape)]
    new_only = [k for k in new_tensors if k not in old_norm]
    print(f"      alignment info: exact={exact} shape_mismatch={len(shape_diff)} new_only={len(new_only)}")
    for k, a, b in shape_diff[:10]:
        print(f"        MISMATCH {k}: old{a} vs new{b}")
    if shape_diff and not args.force:
        eprint("ERROR: shape mismatches vs old encoder keys — the architectures differ more than expected. Use --force to proceed anyway.")
        sys.exit(2)

    out_sd = {k: vc_sd[k] for k in kept_keys}
    for k, v in new_tensors.items():
        out_sd[PERCEPTION_PREFIX + k] = v.contiguous()
    n_params = sum(v.numel() for v in new_tensors.values())
    print(f"      output tensors: {len(out_sd)} (injected {len(new_tensors)} tensors, {n_params:,} params)")

    del vc_sd

    # ------------------------------------------------------------------ config
    print("[3/5] patching config.json perception.encoder / perception.preprocessor")
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path, "r") as f:
        full_cfg = json.load(f)
    per = full_cfg["model"]["stt"]["model"]["perception"]
    for section in REPLACE_PREFIXES:
        name = section.rstrip(".")
        per[name] = cfg_to_dict(getattr(asr.cfg, name))
        print(f"      perception.{name} <- ASR cfg (keys={len(per[name])})")
    assert per.get("modality_adapter"), "modality_adapter config must stay untouched"
    if "output_dim" not in per:
        per["output_dim"] = 4480
    # _rnnt_merge_info and everything else intentionally untouched
    if "_rnnt_merge_info" in full_cfg:
        print("      _rnnt_merge_info preserved (old RNNT stays)")

    # ------------------------------------------------------------------- write
    if args.dry_run:
        print("\n[dry-run] no files written. Summary above is the G2 report.")
        return

    print(f"[4/5] writing output checkpoint to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    from safetensors.torch import save_file

    save_file(out_sd, os.path.join(args.output_dir, "model.safetensors"))
    del out_sd
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(full_cfg, f, indent=2, ensure_ascii=False)
    for name in os.listdir(ckpt_dir):  # rnnt_tokenizer/ and any other siblings
        if name in ("model.safetensors", "config.json"):
            continue
        src, dst = os.path.join(ckpt_dir, name), os.path.join(args.output_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    # --------------------------------------------------------------- verify
    print("[5/5] verifying output")
    out_safetensors = os.path.join(args.output_dir, "model.safetensors")
    size_gb = os.path.getsize(out_safetensors) / 1024**3
    with open(os.path.join(args.output_dir, "config.json"), "r") as f:
        check_cfg = json.load(f)
    att_out = check_cfg["model"]["stt"]["model"]["perception"]["encoder"].get("att_context_size")
    ok_merge = "_rnnt_merge_info" in check_cfg if "_rnnt_merge_info" in full_cfg else True
    print(f"      model.safetensors: {size_gb:.2f} GB")
    print(f"      config att_context_size: {att_out}   rnnt_merge_info kept: {ok_merge}")
    if not ok_merge or list(att_out or [])[1:2] != [0]:
        eprint("ERROR: output verification failed")
        sys.exit(2)

    print(f"\n[done] ZH checkpoint ready at {args.output_dir} ({time.time() - t0:.0f}s)")
    print("       next: deploy_s2s_model.sh with this dir mounted at /checkpoint (see deploy-asr35-zh.md)")


if __name__ == "__main__":
    main()
