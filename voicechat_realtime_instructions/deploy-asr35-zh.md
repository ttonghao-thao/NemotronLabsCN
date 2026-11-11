# Phase 1 — ZH VoiceChat via ASR 3.5 Encoder Replacement (Complete Operations Guide)

This flow replaces the English perception encoder
(`nemotron-speech-streaming-en-0.6b`) inside the official VoiceChat 11B
checkpoint with `nvidia/nemotron-3.5-asr-streaming-0.6b`, keeps everything
else frozen (modality adapter, projection, old EN RNNT, Nano 9B, EAR-TTS,
function head), and serves/tests the result.

**Phase-1 verdict (recorded, see Step 6): scenario C — zero-shot is NOT
viable.** The flow below remains the complete, verified pipeline: it is the
diagnostic + rebuild loop that phase-2 (adapter/projection fine-tuning)
iterates on, and every command has been executed end-to-end on the target
2×L20 host.

Operating principles: the NIM image is **never modified** — all custom code
(the G0–G2 scripts, three patched deploy files for offline G3, one patched
`load_utils.py` for dual-GPU serving) is mounted over the image copies at
run time. The whole flow runs **fully offline** (no HF access at any step).
Scope & acceptance (phase 1): 中文语音 → Nano 9B → 正确理解并回答; RNNT
side-channel transcription quality is explicitly **not** an acceptance
criterion (the old EN RNNT stays in place).

## Step 0 — one-time server setup

```bash
mkdir -p ~/voicechat-zh/{scripts,models,audio,zh-checkpoint,model-repo,nano9b}
# The NIM container runs as a non-root user (uid 1000): grant read on
# everything and write on the output dirs, otherwise every step dies with
# `[Errno 13] Permission denied` on the mounted files (common when files
# were downloaded/transferred as root or with 0600 perms).
chmod -R a+rX ~/voicechat-zh
chmod -R a+rwX ~/voicechat-zh/{zh-checkpoint,model-repo}
```

```
~/voicechat-zh/
├── scripts/         # G0-G2 scripts from this repo
├── s2s/             # patched deploy scripts (Step 4 mounts two of them)
├── opt/             # patched checkpoint_utils/{import_utils,load_utils}.py
├── nano9b/          # LLM config + tokenizer files, plain snapshot copy
├── hf-checkpoint/   # official EN checkpoint (hf download)
├── models/          # pre-downloaded ASR 3.5 (bare .nemo or snapshot dir)
├── audio/           # en_*.wav / zh_*.wav 16kHz mono  (G1 input / test set)
├── zh-checkpoint/   # ZH checkpoint                    (Step 3 output)
├── model-repo/      # Triton model repository          (Step 4 output)
└── client-out/      # test-client results              (Step 6 output)
```

Disk: ≥ 80 GB free. Sync the repo files (note: `s2s/` and `opt/` are
SIBLINGS of `model-repo/` — never copy anything into `model-repo/` itself;
Triton polls every subdirectory of the repo path as a model, and a stray
dir there aborts serving startup with `failed to load all models`):

```bash
git clone -b nemotron-labs-voicechat https://github.com/ttonghao-thao/NemotronLabsCN.git /tmp/repo
cp /tmp/repo/examples/speechlm2/verify_asr35_encoder_compatibility.py \
   /tmp/repo/examples/speechlm2/replace_voicechat_perception_encoder.py \
   ~/voicechat-zh/scripts/
cp -a /tmp/repo/s2s ~/voicechat-zh/
mkdir -p ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils
cp /tmp/repo/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py \
   /tmp/repo/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py \
   ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/
chmod -R a+rX ~/voicechat-zh/s2s ~/voicechat-zh/opt
```

Inputs to stage (all offline afterwards):

```bash
# EN official checkpoint (G1/G2 input, and the Step-6 differential baseline)
pip install -U "huggingface_hub[cli]"
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --local-dir ~/voicechat-zh/hf-checkpoint

# ASR 3.5 (G0-G2 input)
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
   --local-dir ~/voicechat-zh/models/nemotron-3.5-asr-streaming-0.6b

# LLM config + tokenizer files for offline G3 (a few MB, no weights).
# Any plain snapshot copy works (HF or modelscope mirror) — no hub-cache layout:
#   modelscope download nv-community/NVIDIA-Nemotron-Nano-9B-v2 --exclude "*.safetensors"
cp -a /root/.cache/modelscope/models/nv-community--NVIDIA-Nemotron-Nano-9B-v2/snapshots/master/. \
      ~/voicechat-zh/nano9b/
chmod -R a+rX ~/voicechat-zh/nano9b
# sanity: must match the Step-4 log line
# "Saved embeddings (vocab_size=131072, hidden_size=4480)"
grep -E '"(vocab_size|hidden_size)"' ~/voicechat-zh/nano9b/config.json
```

Prepare test WAVs in `audio/` named with language tags (`en_*.wav`,
`zh_*.wav`, 16 kHz mono) — the tag drives G1's per-language latent report
and the Step-6 test set.

All G0–G2 `docker run` commands share this base:

```bash
--runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
-v ~/voicechat-zh:/work -w /work \
--entrypoint bash \
nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

## Step 1 — G0: sanity-check the local ASR 3.5

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh:/work -w /work \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python scripts/verify_asr35_encoder_compatibility.py \
     --asr models/nemotron-3.5-asr-streaming-0.6b.nemo \
     --probe-wav audio/zh_1.wav"
```

Pass: `d_model = 1024` and a causal streaming mode among the
`att_context_size` modes. ASR 3.5 ships `[[56, 3], [56, 0], [56, 6],
[56, 13]]` — four streaming *modes* (not per-layer contexts); G0 reports
the causal `[56, 0]` mode and G2 pins it into the ZH checkpoint (the
runtime consumes the frontier encoder frame with no audio ahead of it, so
causal is the only in-distribution choice; the old EN encoder itself ran
`[70, 1]`). If your ASR 3.5 copy is a snapshot dir, pass the dir as
`--asr` — the script resolves the `.nemo` inside.

A garbled/missing `--probe-wav` transcript is a known shim limitation (no
language prompting on nemo 2.4.0rc0), not a model defect.

## Step 2 — G1: compatibility pre-check

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

Pass: key alignment `shape_mismatch = 0` (nonzero `new_only` is fine).
The latent report predicts the G4 outcome: EN mean cosine > 0.9 →
zero-shot likely works; < 0.7 → expect adapter/projection tuning.

**Reference values from this deployment (recorded):** key alignment
`exact=638, shape_mismatch=0`. The latent cosines this run printed
(EN 0.083 / ZH 0.0027) are **invalid** — the script sliced the encoder's
`[B,D,T]` output along the channel axis instead of the frame axis;
[asr35-zh-next-steps.md](asr35-zh-next-steps.md) item 2 defines the
corrected computation. Do not cite the cosine numbers; the verdict rests on
the Step-6 differential alone.

## Step 3 — G2: solidify the ZH checkpoint

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh:/work -w /work \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python scripts/replace_voicechat_perception_encoder.py \
     --voicechat-checkpoint-dir hf-checkpoint \
     --asr-nemo models/nemotron-3.5-asr-streaming-0.6b.nemo \
     --output-dir zh-checkpoint --dry-run"
# review the report, then rerun without --dry-run (add --overwrite when rerunning)
```

Pass: encoder+preprocessor tensors injected (shape mismatches = 0),
`modality_adapter`/`proj` kept, `_rnnt_merge_info` preserved, output
`att_context_size = [56, 0]` (G2 pins the causal mode and drops
`att_context_probs`).

## Step 4 — G3: generate the Triton model repo (offline)

Same official entrypoint as [generate-model-repo.md](generate-model-repo.md)
(`/s2s/deploy_s2s_model.sh`). Three spots in the deploy path load the LLM's
config/tokenizer from `nvidia/NVIDIA-Nemotron-Nano-9B-v2` over the network;
this repo's patched copies load from a local dir instead when
`NEMO_LLM_LOCAL_DIR` is set (unset → behavior identical to stock):
`checkpoint_utils/import_utils.py` (tokenizer bundling + the
`NemotronVoiceChat` build in the TTS-init-inputs step),
`s2s/scripts/convert_nemotronllm_checkpoint.py`, and
`s2s/scripts/convert_eartts_checkpoint.py`. The patched `import_utils.py`
also pins the generated `config.pbtxt` instance_group to `gpus: [0]` —
without `gpus`, Triton creates `count` instances **per visible GPU**, i.e.
two full model copies and a guaranteed OOM on a 2-GPU box.

```bash
docker run -it --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v ~/voicechat-zh/zh-checkpoint:/checkpoint \
  -v ~/voicechat-zh/model-repo:/data/models \
  -v ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py:/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py:ro \
  -v ~/voicechat-zh/s2s/scripts/convert_nemotronllm_checkpoint.py:/s2s/scripts/convert_nemotronllm_checkpoint.py:ro \
  -v ~/voicechat-zh/s2s/scripts/convert_eartts_checkpoint.py:/s2s/scripts/convert_eartts_checkpoint.py:ro \
  -v ~/voicechat-zh/nano9b:/llm/Nemotron-Nano-9B-v2:ro \
  -e NEMO_LLM_LOCAL_DIR=/llm/Nemotron-Nano-9B-v2 \
  -e NEMO_CHECKPOINT_PATH=/checkpoint \
  --entrypoint /s2s/deploy_s2s_model.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

The `nano9b` host dir is mounted at a path containing `Nemotron` **on
purpose**: `DuplexSTTModel` branches on `'Nemotron' in
cfg.pretrained_llm` in several places, and the patch points that field at
`NEMO_LLM_LOCAL_DIR` — a path without the substring silently skips the
Nemotron-specific model handling.

Pass: `NEMO_LLM_LOCAL_DIR set - …` log lines at the tokenizer step and both
conversions, then `Triton model repository generation completed
successfully!`. This step fully instantiates `NemotronVoiceChat` from the
patched config, so a successful conversion is itself the structural
compatibility check; failure → compare encoder config keys and re-run G2
with a field mapping.

Notes: G3 is idempotent (completed steps are skipped on rerun). The model
repo top level must contain **only** `nemotron-voicechat/` — that is all
G3 creates, and Step 5 verifies it before starting. The patched files are
modified copies of the image's originals: when bumping the NIM image tag,
re-diff them against the new image and re-apply the hunks if upstream
changed. Serving needs none of the G3 mounts — the runtime rebinds the
tokenizer to the bundled `model-repo/.../tokenizer/` at load time
(`load_utils.py`).

## Step 5 — G4: serve (dual-GPU: LLM on GPU 0, TTS on GPU 1)

The serving container mounts the model repo plus one patched file
(`load_utils.py`); the image itself is untouched. Pre-flight check first —
the model repo top level must show **only** `nemotron-voicechat/` (Triton
polls every subdirectory of the repo path as a model; a stray directory
aborts startup with `failed to load all models` even though the model
loaded fine):

```bash
ls ~/voicechat-zh/model-repo/   # must show only: nemotron-voicechat
```

How the placement works, and what each env var does:

- The patched `load_utils.py` wraps `load_tts_model_in_vllm(...)` with a
  temporary `CUDA_VISIBLE_DEVICES` swap driven by `TTS_CUDA_VISIBLE_DEVICES`
  — vLLM `EngineCore` subprocesses inherit the environment at creation
  time, so the TTS engine lands on physical GPU 1 while the LLM (created
  first) stays on GPU 0; the variable is restored immediately so the rest
  of the Triton stub still sees both GPUs.
- `DEFAULT_INSTRUCTIONS` is part of the command, not an optional extra:
  the test client sends no instructions of its own, and the built-in
  tool-calling default explicitly mandates "conversations in English"
  (`audio_server.py:DEFAULT_INSTRUCTIONS_FC`), which biases Chinese input
  toward English replies. Resolution order: client instructions >
  `DEFAULT_INSTRUCTIONS` > built-in defaults. (`prompt_template.jinja`
  needs no change — language-neutral plumbing.)
- Verified memory layout: GPU 0 ≈ 350 M main + 3.3 G stub + ~31 G LLM
  engine; GPU 1 ≈ 350 M main + 3.8 G TTS engine.

| Variable | Value | Notes |
|---|---|---|
| `TTS_CUDA_VISIBLE_DEVICES` | `1` | activates the load_utils patch; TTS lands on physical GPU 1 |
| `LLM_GPU_MEM_UTIL` | `0.65` | stock default 0.45. 0.85 failed vLLM startup precheck with both stubs resident (37.49 G free < 37.84 G needed); raise only while watching nvidia-smi |
| `TTS_GPU_MEM_UTIL` | `0.4` | stock default; a cap, not a reservation (measured ≈3.8 G) |

```bash
docker run -d --name nemotron-labs-voicechat-zh \
  --runtime=nvidia --gpus all --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -e DEFAULT_INSTRUCTIONS="你是一名语音助手。始终用用户所说的语言回答:用户说中文就用中文,说英文就用英文。口语化、对话式、简洁自然,先向用户问好。" \
  -e TTS_CUDA_VISIBLE_DEVICES=1 \
  -e LLM_GPU_MEM_UTIL=0.65 \
  -e TTS_GPU_MEM_UTIL=0.4 \
  -v ~/voicechat-zh/model-repo:/data/models \
  -v ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py:/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Single-GPU fallback: drop `TTS_CUDA_VISIBLE_DEVICES` and the
`load_utils.py` mount, use `--gpus '"device=0"'` and `-it --rm` if you
prefer. Ops notes: `-d` not `-it` (a closed terminal SIGHUPs the
container). Before a restart, check `nvidia-smi` for orphaned
`VLLM::EngineCore` processes and kill them. Startup is slow (Triton
health-wait up to 1200 s + warmup); `[TRITON]`-prefixed log lines come
from `run_s2s_server.sh`'s sed and are normal. `load_utils`'s logger has
no INFO handler in the backend path — verify engine placement with
`nvidia-smi`, not logs. One instance per box.

Readiness (terminal 2):

```bash
curl 'http://localhost:9000/v1/realtime/health'
```

## Step 6 — test & verdict

Run the test client in a second container — the image already ships the
client and all its deps (websockets / soundfile / numpy); nothing is
installed on the host. `--network=host` reaches the server's published
port; audio in and results out via mounts:

```bash
mkdir -p ~/voicechat-zh/client-out
docker run -it --rm --network=host \
  -v ~/voicechat-zh/audio:/audio:ro \
  -v ~/voicechat-zh/client-out:/out -w /out \
  --entrypoint bash \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest -c \
  "python /s2s/nemotron-voicechat-client.py --server ws://localhost:9000 \
     --input-file /audio/zh_1.wav --no-playback"
```

Repeat per clip. Results land on the host:

```bash
cat ~/voicechat-zh/client-out/agent_text.txt   # the LLM's reply — the acceptance signal
cat ~/voicechat-zh/client-out/user_text.txt    # RNNT side-channel ASR
```

Fixed test set: `你好，你是谁？` / `请简单介绍一下北京。` / `帮我查一下产品知识。`
plus an English regression clip. If the ws handshake fails, retry with
`--server wss://localhost:9000`. File input streams the clip plus ~20 s of
trailing silence, so each run takes clip-length + ~20 s.

**Differential test (mandatory before any verdict)**: serve the EN stock
checkpoint's model repo with the identical Step-5 config and run the same
clips. This isolates checkpoint issues from serving issues — it is what
cleared the serving stack in phase 1.

| Outcome | Meaning | Next |
|---|---|---|
| Chinese understood & answered, EN intact | scenario A | done — ZH checkpoint is the deliverable |
| Partial / unstable Chinese | scenario B | freeze encoder/LLM/TTS, fine-tune adapter+projection on Chinese speech |
| Chinese not understood | scenario C | redo Speech→LLM alignment |

### Recorded phase-1 verdict: scenario C — zero-shot is not viable

The load-bearing evidence is the **G4 differential**: the EN stock
checkpoint greets/transcribes/responds under the identical serving config,
while the ZH checkpoint streams pure silence with zero text events — not
even the proactive greeting, which depends only on the LLM path
(`proj(adapter(new_encoder))`) reading *silence* frames, not on the RNNT.

**G1 latent numbers: retracted.** The original script sliced the encoder's
`[B,D,T]` output along the channel axis instead of frames, so the printed
cosines (EN 0.083 / ZH 0.0027) are meaningless. The causal story
("orthogonal representation spaces", "training is mandatory") is on hold
until the corrected comparison and the frame-level EN/ZH trace in
[asr35-zh-next-steps.md](asr35-zh-next-steps.md) /
[asr35-g4-runbook-zh.md](asr35-g4-runbook-zh.md) say otherwise; the
empirical verdict above stands on the differential alone.

Conclusion: swapping only the perception encoder did not work at runtime —
the frozen downstream could not act on the swapped encoder's output. Phase-1
deliverables: the diagnostic chain (G0–G4 + differential methodology), the
offline G3 / dual-GPU serving infrastructure, and the follow-up plan with
corrected metrics.

Phase-2 (scenario-C row, gated on the corrected diagnostics): fine-tune
`modality_adapter` + `projection`, everything else frozen, starting from
the ZH checkpoint, on mixed ZH/EN speech with paired responses — entry
points are this repo's `examples/speechlm2/` scripts
(`s2s_duplex_train.py`, `salm_train.py`). Side task: replace the RNNT side
channel with ASR 3.5's own decoder/joint + tokenizer (the 15 tensors G2
skips).

Known caveats regardless of scenario: the RNNT side-channel transcript
(`user_text.txt`) degrades with the new encoder (old EN RNNT reads the raw
new-encoder output); turn-taking oddities are phase-2 RNNT work, not
encoder failure.

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
prompting — a garbled probe transcript is a known shim limitation, not a
model defect.

**`att_context_size [[56, 3], [56, 0], [56, 6], [56, 13]]` flagged as "not
causal"** — a list-of-lists is NeMo's multi-**mode** streaming config, not
per-layer contexts: modes are sampled during training (per
`att_context_probs`) and inference uses the first entry unless switched.
`[56, 0]` causal is one of the trained modes. The VoiceChat runtime
re-encodes the whole audio buffer and consumes the frontier encoder frame
immediately (`audio_idx = audio_chunk_idx`, `MODEL_STEPS_PER_CALL=2`) — that
frame has no audio ahead of it, so it always runs with right context 0
whatever the configured mode. G2 pins `[56, 0]`; G1's latent comparison pins
the same mode.

**Probe transcript skipped (`[probe] WARNING: transcribe failed`)** — the shim
class supports cfg / encoder / preprocessor access, but full streaming
transcription may not work on the older nemo. G0's hard assertions and the
G1 report are unaffected; transcription comes back at Step 6 with the full
runtime.

**G3 fails loading the LLM tokenizer (`Network is unreachable`,
`PermissionError at …/token`, or `TypeError: PosixPath / NoneType`)** — one of
the three patched deploy files isn't mounted, or `-e NEMO_LLM_LOCAL_DIR` is
missing, so the stock image code tried the HF hub. Check the Step-4 command
has all three `-v …` file mounts plus `NEMO_LLM_LOCAL_DIR`, and that
`~/voicechat-zh/nano9b/config.json` exists (`vocab_size=131072,
hidden_size=4480`). You should see `NEMO_LLM_LOCAL_DIR set - …` in the logs
before the failing step.

## Rollback

Serve the English stack again at any time by pointing the Step-5 serving
container at a model repo generated from `hf-checkpoint` (or the NGC
prebuilt) — the ZH checkpoint is a separate directory and nothing in the
flow is destructive.
