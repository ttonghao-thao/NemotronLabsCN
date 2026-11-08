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

Pass: prints `d_model = 1024`, causal `att_context_size = [56, 0]`, and a
readable Chinese transcript of the probe WAV.

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

## Rollback

Serve the English stack again at any time by pointing the serving container at a
model repo generated from `hf-checkpoint` (or the NGC prebuilt) — the ZH
checkpoint is a separate directory and nothing in the flow is destructive.
