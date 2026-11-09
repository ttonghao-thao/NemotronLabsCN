# Phase 1 — Chinese Support via ASR 3.5 Encoder Replacement (ZH Deployment Guide)

This guide replaces the English perception encoder (`nemotron-speech-streaming-en-0.6b`)
inside the official VoiceChat 11B checkpoint with `nvidia/nemotron-3.5-asr-streaming-0.6b`,
keeping everything else (modality adapter, projection, old RNNT, Nano 9B, EAR-TTS,
function head) unchanged.

**Your official deployment flow is used unchanged** — the only difference is *which
checkpoint directory* is mounted. All new code (two scripts) runs inside the NIM
inference container via mounts, so the nemo package used for the check is exactly
the one the runtime uses.

Scope & acceptance (phase 1): verify 中文语音 → Nano 9B → 正确理解并回答.
Chinese RNNT transcription / turn-taking quality is **not** a phase-1 acceptance
criterion (the old English RNNT stays in place).

## Server directory layout

```bash
mkdir -p ~/voicechat-zh/{scripts,models,audio,zh-checkpoint,model-repo}
# The NIM container runs as a non-root user (uid 1000): grant read on
# everything and write on the output dirs, otherwise every step dies with
# `[Errno 13] Permission denied` on the mounted files (common when the
# .nemo was downloaded/transferred as root or with 0600 perms).
chmod -R a+rX ~/voicechat-zh
chmod -R a+rwX ~/voicechat-zh/{zh-checkpoint,model-repo}
```

```
~/voicechat-zh/
├── scripts/         # the two scripts from this repo
├── hf-checkpoint/   # official EN checkpoint (hf download)
├── models/          # pre-downloaded ASR 3.5 (bare .nemo or HF snapshot dir)
├── audio/           # en_*.wav / zh_*.wav 16kHz mono         (G1 input)
├── zh-checkpoint/   # ZH checkpoint                           (G2 output)
└── model-repo/      # Triton model repository                 (G3 output)
```

Disk: keep ≥ 80 GB free (EN checkpoint + ZH checkpoint + model-repo + vLLM dirs).

Get the scripts and the EN checkpoint:

```bash
git clone -b nemotron-labs-voicechat https://github.com/ttonghao-thao/NemotronLabsCN.git /tmp/repo
cp /tmp/repo/examples/speechlm2/verify_asr35_encoder_compatibility.py \
   /tmp/repo/examples/speechlm2/replace_voicechat_perception_encoder.py \
   ~/voicechat-zh/scripts/

pip install -U "huggingface_hub[cli]"
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --local-dir ~/voicechat-zh/hf-checkpoint
```

ASR 3.5 (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is assumed to be already
downloaded into `models/` (e.g. `hf download nvidia/nemotron-3.5-asr-streaming-0.6b
--local-dir ~/voicechat-zh/models/nemotron-3.5-asr-streaming-0.6b`, run wherever
network is available). G0/G1/G2 run fully offline against this local copy.

Prepare at least a few 16 kHz mono WAVs in `audio/`, named with language tags
(e.g. `en_1.wav`, `en_2.wav`, `zh_1.wav`, `zh_2.wav`) — the language tag drives
the per-language latent report.

All `docker run` commands below share this base (adjust GPU index as needed):

```bash
--runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
-v ~/voicechat-zh:/work -w /work \
--entrypoint bash \
nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

## G0 — sanity-check the local ASR 3.5 (inside container, offline)

ASR 3.5 is **already downloaded into `models/`** — G0/G1 never touch the
network (keep the image default `HF_HUB_OFFLINE=1`). `--asr` accepts either
the bare `.nemo` file or the snapshot directory containing it (the script
finds the `.nemo` inside automatically):

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh:/work -w /work \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python scripts/verify_asr35_encoder_compatibility.py \
     --asr models/nemotron-3.5-asr-streaming-0.6b.nemo \
     --probe-wav audio/zh_1.wav"
```

Pass: prints `d_model = 1024` and reports a causal streaming mode among the
model's `att_context_size` modes. ASR 3.5 ships `[[56, 3], [56, 0], [56, 6],
[56, 13]]` — four streaming *modes* (not per-layer contexts); G0 reports the
causal `[56, 0]` mode and G2 pins it into the ZH checkpoint. A garbled or
missing `--probe-wav` transcript is a known shim caveat (no language
prompting); the authoritative check is G4.

If your local copy is an HF snapshot dir, pass the directory instead
(`--asr models/nemotron-3.5-asr-streaming-0.6b`) — for G1/G2 note the actual
`.nemo` path it resolves to and use that in `--asr-nemo`.

## G1 — compatibility pre-check (inside container)

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh:/work -w /work \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python scripts/verify_asr35_encoder_compatibility.py \
     --asr models/nemotron-3.5-asr-streaming-0.6b.nemo \
     --voicechat-checkpoint-dir hf-checkpoint \
     --audio-dir audio \
     --report compatibility_report.json"
```

Pass: key alignment shows no shape mismatches (nonzero `new_only` is fine).
Decision hint from the latent report: EN mean cosine > 0.9 → zero-shot likely
works; < 0.7 → expect adapter/projection tuning later (still proceed — the
empirical answer comes from G4).

## G2 — solidify the ZH checkpoint (inside container)

Dry-run first, then write:

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh:/work -w /work \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python scripts/replace_voicechat_perception_encoder.py \
     --voicechat-checkpoint-dir hf-checkpoint \
     --asr-nemo models/nemotron-3.5-asr-streaming-0.6b.nemo \
     --output-dir zh-checkpoint --dry-run"
# then the same command without --dry-run (add --overwrite when rerunning)
```

Pass: report shows encoder+preprocessor tensors injected (shape mismatches = 0),
`modality_adapter`/`proj` kept, `_rnnt_merge_info` preserved, output
`att_context_size = [56, 0]`.

## G3 — generate the Triton model repo (official flow, ZH checkpoint mounted)

Identical to [generate-model-repo.md](generate-model-repo.md), only the mounted
checkpoint differs:

```bash
docker run -it --rm \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -v ~/voicechat-zh/zh-checkpoint:/checkpoint \
  -v ~/voicechat-zh/model-repo:/data/models \
  -e NEMO_CHECKPOINT_PATH=/checkpoint \
  --entrypoint /s2s/deploy_s2s_model.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Pass: `Triton model repository generation completed successfully!`
This step fully instantiates `NemotronVoiceChat` from the patched config inside
the container, so a successful conversion is itself the structural
compatibility check. Failure here → compare encoder config keys and re-run G2
with a field mapping.

### G3 on an air-gapped host (offline tokenizer/config bundle)

Everything in G3 is local except two `from_pretrained` calls against
`nvidia/NVIDIA-Nemotron-Nano-9B-v2` (deploy tokenizer bundling in
`checkpoint_utils/import_utils.py`, and `AutoConfig`/`AutoTokenizer` in
`/s2s/scripts/convert_nemotronllm_checkpoint.py`).

**Recommended: patched files + a plain LLM files dir.** This repo carries
patched copies of both files at their in-image paths
(`s2s/scripts/convert_nemotronllm_checkpoint.py` and
`opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py`).
When `NEMO_LLM_LOCAL_DIR` points at a plain directory holding the LLM's
`config.json` + tokenizer files (a modelscope or HF snapshot copy works
as-is — no hub-cache layout involved), both steps load from it; when unset,
behavior is identical to the image originals. No HF cache, no env juggling,
no image changes.

Stage the LLM files once (server):

```bash
mkdir -p ~/voicechat-zh/nano9b
cp -a /root/.cache/modelscope/models/nv-community--NVIDIA-Nemotron-Nano-9B-v2/snapshots/master/. \
      ~/voicechat-zh/nano9b/
chmod -R a+rX ~/voicechat-zh/nano9b
# sanity: must match the G3 log ("Saved embeddings (vocab_size=131072, hidden_size=4480)")
grep -E '"(vocab_size|hidden_size)"' ~/voicechat-zh/nano9b/config.json
```

Sync the patched files from the repo (same as the scripts/ sync) and add
these flags to the G3 `docker run`:

```bash
  -v ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py:/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py:ro \
  -v ~/voicechat-zh/s2s/scripts/convert_nemotronllm_checkpoint.py:/s2s/scripts/convert_nemotronllm_checkpoint.py:ro \
  -v ~/voicechat-zh/nano9b:/nano9b:ro \
  -e NEMO_LLM_LOCAL_DIR=/nano9b
```

Watch for `NEMO_LLM_LOCAL_DIR set - …` log lines in both the deploy
tokenizer step and the nano-v2-vllm conversion.

**Alternative (no patched files): pre-seeded HF hub cache.** Seed a real hub
cache on a networked machine (`hf download … --exclude "*.safetensors"`, tar
only the `models--*` dir) or map a modelscope snapshot into the layout
(cache dir named after the HF repo id; `refs/main` = snapshot dir name via
`printf`, no trailing newline; real files in `snapshots/<name>/`). Mount it
with `-v ~/voicechat-zh/hf-cache/hub:/hf-hub -e HF_HUB_CACHE=/hf-hub
-e HF_HOME=/tmp/hf-home -e TRANSFORMERS_OFFLINE=1` — never point `HF_HOME`
at the read-only mount (the token path is `$HF_HOME/token`; a
`PermissionError` there means a stray unreadable token file or an
unsearchable mount root). Verify as the container user with
`sudo -u '#1000' cat …/refs/main`.

If `AutoConfig(trust_remote_code=True)` later asks for remote code, copy the
repo's `*.py` files into the staged dir and rerun. G3 is idempotent —
steps already written to the model repo are skipped on rerun. Serving (G4)
needs none of this: the runtime rebinds the tokenizer to the bundled
`model-repo/.../tokenizer/` at load time (`load_utils.py`), so the serving
container stays fully offline.

## G4 — serve & test (official flow, no code mounts)

```bash
docker run -it --rm --name=nemotron-labs-voicechat-zh \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -v ~/voicechat-zh/model-repo:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Note: the serving container mounts **only the model repo** — no code mounts,
no image changes.

Readiness poll:

```bash
curl 'http://localhost:9000/v1/realtime/health'
```

Client test (from [deploy.md](deploy.md)):

```bash
docker cp nemotron-labs-voicechat-zh:/s2s/nemotron-voicechat-client.py .
pip install websockets soundfile numpy
python nemotron-voicechat-client.py --input-file ~/voicechat-zh/audio/zh_1.wav --no-playback
```

Fixed test set: `你好，你是谁？` / `请简单介绍一下北京。` / `帮我查一下产品知识。`
plus an English regression clip (e.g. the EN wav used in G1).

| Outcome | Meaning | Next |
|---|---|---|
| Chinese understood & answered, EN intact | zero-shot works (scenario A) | done — ZH checkpoint is the deliverable |
| Partial / unstable Chinese | scenario B | freeze encoder/LLM/TTS, fine-tune adapter+projection on Chinese speech |
| Chinese not understood | scenario C | redo Speech→LLM alignment |

Known phase-1 caveats: the RNNT side-channel transcript (`output_asr_text`) is
expected to be garbage on Chinese — the old English RNNT consumes the new
encoder's raw output. Turn-taking (barge-in/EOU) may misfire on Chinese; treat
oddities there as phase-2 work (RNNT replacement), not encoder failure.

## Troubleshooting

**G0/G2 restore dies with `No module named
'nemo.collections.asr.models.rnnt_bpe_models_prompt'` (followed by a confusing
`Can't instantiate abstract class ASRModel ...`)** — the ASR 3.5 `.nemo` targets
a model class newer than the NIM image's nemo (2.4.0rc0). Both scripts handle
this automatically: before `restore_from` they register a drop-in subclass of
`EncDecRNNTBPEModel` under that exact module path (a no-op on newer nemo that
ships the real class). If restore *still* fails, the scripts print every
`_target_` inside the `.nemo` (`[diag]` lines) — send that output back so the
shim can be extended to the next missing class.

**`Unexpected key(s) in state_dict: "prompt_kernel.*"`** — the real ASR 3.5
class carries a language-prompt conditioning stack (fed by the
`prompt_dictionary` / `num_prompts` config; `zh-CN` maps to id 4) that the
shim's base class does not define. The shim tolerates and drops exactly those
tensors on restore: G0–G2 and the VoiceChat runtime never consume
`prompt_kernel` — only `encoder.*` / `preprocessor.*` cross into the ZH
checkpoint at G2. Any other missing/unexpected key still fails the restore.
Consequence: the `--probe-wav` transcript runs the encoder *without* language
prompting, so treat a wrong-language or garbled probe transcript as a known
shim limitation, not a model defect — the authoritative check is G4.

**`att_context_size [[56, 3], [56, 0], [56, 6], [56, 13]]` flagged as "not
causal"** — a list-of-lists is NeMo's multi-**mode** streaming config, not
per-layer contexts: modes are sampled during training (per
`att_context_probs`) and inference uses the first entry unless switched.
`[56, 0]` causal is one of the trained modes. The VoiceChat runtime
re-encodes the whole audio buffer and consumes the frontier encoder frame
immediately (`audio_idx = audio_chunk_idx`, `MODEL_STEPS_PER_CALL=2`) — that
frame has no audio ahead of it, so it always runs with right context 0
whatever the configured mode (the old EN encoder itself ran `[70, 1]` under
the same regime). G2 therefore pins `[56, 0]`, which is in-distribution and
matches runtime behavior exactly; G1's latent comparison pins the same mode.

**Probe transcript skipped (`[probe] WARNING: transcribe failed`)** — the shim
class supports cfg / encoder / preprocessor access, but full streaming
transcription may not work on the older nemo. G0's hard assertions
(d_model=1024, causal att_context_size) and the G1 report are unaffected;
transcription comes back at G4 with the full runtime.

## Rollback

Serve the English stack again at any time by pointing the serving container at a
model repo generated from `hf-checkpoint` (or the NGC prebuilt) — the ZH
checkpoint is a separate directory and nothing in the flow is destructive.
