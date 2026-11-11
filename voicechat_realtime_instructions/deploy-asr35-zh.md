# Phase 1 — Chinese Support via ASR 3.5 Encoder Replacement (ZH Deployment Guide)

This guide replaces the English perception encoder (`nemotron-speech-streaming-en-0.6b`)
inside the official VoiceChat 11B checkpoint with `nvidia/nemotron-3.5-asr-streaming-0.6b`,
keeping everything else (modality adapter, projection, old RNNT, Nano 9B, EAR-TTS,
function head) unchanged.

**The official deployment flow is used unchanged** — the only difference is *which
checkpoint directory* is mounted. All custom code (the G0–G2 scripts, three
patched deploy files for offline G3, and a patched `load_utils.py` for
dual-GPU serving) runs inside the NIM inference container via
mounts; the image itself is never modified, so the nemo package used for every
check is exactly the one the runtime uses. The whole flow G0→G4 runs fully
offline (no HF access at any step).

Scope & acceptance (phase 1): verify 中文语音 → Nano 9B → 正确理解并回答.
Chinese RNNT transcription / turn-taking quality is **not** a phase-1 acceptance
criterion (the old English RNNT stays in place).

## Server directory layout

```bash
mkdir -p ~/voicechat-zh/{scripts,models,audio,zh-checkpoint,model-repo,nano9b}
# The NIM container runs as a non-root user (uid 1000): grant read on
# everything and write on the output dirs, otherwise every step dies with
# `[Errno 13] Permission denied` on the mounted files (common when the
# .nemo was downloaded/transferred as root or with 0600 perms).
chmod -R a+rX ~/voicechat-zh
chmod -R a+rwX ~/voicechat-zh/{zh-checkpoint,model-repo}
```

```
~/voicechat-zh/
├── scripts/         # G0-G2 scripts from this repo
├── s2s/             # patched deploy scripts (G3 mounts two of them over the image copies)
├── opt/             # patched checkpoint_utils/import_utils.py (G3 mounts it)
├── nano9b/          # LLM config + tokenizer files, plain snapshot copy (G3 input)
├── hf-checkpoint/   # official EN checkpoint (hf download)
├── models/          # pre-downloaded ASR 3.5 (bare .nemo or snapshot dir)
├── audio/           # en_*.wav / zh_*.wav 16kHz mono         (G1 input)
├── zh-checkpoint/   # ZH checkpoint                           (G2 output)
└── model-repo/      # Triton model repository                 (G3 output)
```

Disk: keep ≥ 80 GB free (EN checkpoint + ZH checkpoint + model-repo + vLLM dirs).

Get the repo files and the EN checkpoint:

```bash
git clone -b nemotron-labs-voicechat https://github.com/ttonghao-thao/NemotronLabsCN.git /tmp/repo
# G0-G2 scripts
cp /tmp/repo/examples/speechlm2/verify_asr35_encoder_compatibility.py \
   /tmp/repo/examples/speechlm2/replace_voicechat_perception_encoder.py \
   ~/voicechat-zh/scripts/
# patched deploy files for offline G3 (mounted over the image copies, see G3)
cp -a /tmp/repo/s2s ~/voicechat-zh/
mkdir -p ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils
cp /tmp/repo/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py \
   ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/
chmod -R a+rX ~/voicechat-zh/s2s ~/voicechat-zh/opt

pip install -U "huggingface_hub[cli]"
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --local-dir ~/voicechat-zh/hf-checkpoint
```

Stage the LLM files for offline G3 — `nvidia/NVIDIA-Nemotron-Nano-9B-v2`'s
`config.json` + tokenizer files only (a few MB, weights not needed). Any plain
snapshot copy works as-is (HF or modelscope mirror); no HF hub cache layout
is involved:

```bash
# from a machine that has the snapshot (e.g. modelscope):
#   modelscope download nv-community/NVIDIA-Nemotron-Nano-9B-v2 --exclude "*.safetensors"
cp -a /root/.cache/modelscope/models/nv-community--NVIDIA-Nemotron-Nano-9B-v2/snapshots/master/. \
      ~/voicechat-zh/nano9b/
chmod -R a+rX ~/voicechat-zh/nano9b
# sanity: must match the G3 log line "Saved embeddings (vocab_size=131072, hidden_size=4480)"
grep -E '"(vocab_size|hidden_size)"' ~/voicechat-zh/nano9b/config.json
```

ASR 3.5 (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is assumed to be already
downloaded into `models/` (e.g. `hf download nvidia/nemotron-3.5-asr-streaming-0.6b
--local-dir ~/voicechat-zh/models/nemotron-3.5-asr-streaming-0.6b`, run wherever
network is available). G0/G1/G2 run fully offline against this local copy.

Prepare at least a few 16 kHz mono WAVs in `audio/`, named with language tags
(e.g. `en_1.wav`, `en_2.wav`, `zh_1.wav`, `zh_2.wav`) — the language tag drives
the per-language latent report.

All G0–G2 `docker run` commands share this base (adjust GPU index as needed):

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

## G3 — generate the Triton model repo (offline, patched deploy files)

Same official entrypoint as [generate-model-repo.md](generate-model-repo.md)
(`/s2s/deploy_s2s_model.sh`), with the ZH checkpoint mounted. Three spots in
the deploy path load the LLM's config/tokenizer from
`nvidia/NVIDIA-Nemotron-Nano-9B-v2` over the network; this repo carries
patched copies of exactly those three files that load from a local directory
instead when `NEMO_LLM_LOCAL_DIR` is set (unset, they behave identically to
the image originals):

- `opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py`
  — deploy tokenizer bundling **and** the `NemotronVoiceChat` build in the
  TTS-init-inputs step (rewrites `pretrained_llm` / `pretrained_lm_name`)
- `s2s/scripts/convert_nemotronllm_checkpoint.py` — `AutoConfig` /
  `AutoTokenizer` for the nano-v2-vllm conversion
- `s2s/scripts/convert_eartts_checkpoint.py` — `DuplexEARTTS`'s
  `AutoTokenizer` (`cfg.model.pretrained_lm_name`)

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
purpose**: `DuplexSTTModel` branches on `'Nemotron' in cfg.pretrained_llm` in
several places, and the patch points that field at `NEMO_LLM_LOCAL_DIR` — a
path without the substring would silently skip the Nemotron-specific model
handling.

Pass: `NEMO_LLM_LOCAL_DIR set - …` log lines appear at the tokenizer step and
both conversions, and the run ends with
`Triton model repository generation completed successfully!`. This step fully
instantiates `NemotronVoiceChat` from the patched config inside the container,
so a successful conversion is itself the structural compatibility check.
Failure here → compare encoder config keys and re-run G2 with a field mapping.

Notes:

- G3 is idempotent — steps already written to the model repo are skipped on
  rerun.
- The patched files are modified copies of the image's originals. When bumping
  the NIM image tag, re-diff them against the new image and re-apply the
  `NEMO_LLM_LOCAL_DIR` hunks if upstream changed.
- If `AutoConfig(trust_remote_code=True)` asks for remote code, copy the
  repo's `*.py` files into `nano9b/` and rerun.
- G4 serving mounts **only the model repo** and needs none of this: the
  runtime rebinds the tokenizer to the bundled `model-repo/.../tokenizer/`
  at load time (`load_utils.py`), so the serving container stays fully
  offline.

## G4 — serve & test (official flow, no code mounts)

G4 runs entirely on the server, in three steps: start the serving container
(terminal 1) → wait for readiness → run the test client in a second
container (terminal 2). The serving container mounts **only the model repo**
— no code mounts, no image changes.

**Step 1 — start the server** (terminal 1). `DEFAULT_INSTRUCTIONS` is part
of this command, not an optional extra: the test client sends no
instructions of its own, and the server's built-in tool-calling default
explicitly mandates "conversations in English"
(`audio_server.py:DEFAULT_INSTRUCTIONS_FC`), which would bias Chinese input
toward English replies and pollute the acceptance result. Resolution order:
client instructions > `DEFAULT_INSTRUCTIONS` > built-in defaults.
(`prompt_template.jinja` needs no change — language-neutral plumbing.)

```bash
docker run -it --rm --name=nemotron-labs-voicechat-zh \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -e DEFAULT_INSTRUCTIONS="你是一名语音助手。始终用用户所说的语言回答:用户说中文就用中文,说英文就用英文。口语化、对话式、简洁自然,先向用户问好。" \
  -v ~/voicechat-zh/model-repo:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

**Step 2 — wait for readiness** (terminal 2):

```bash
curl 'http://localhost:9000/v1/realtime/health'
```

**Step 3 — run the test client in a second container** (terminal 2). The
image already ships the client and all its deps (websockets / soundfile /
numpy); nothing is installed on the host. `--network=host` reaches the
server's published port via localhost; audio goes in and results come out
through mounts:

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

Repeat for each test clip. Results land on the host after each run:

```bash
cat ~/voicechat-zh/client-out/agent_text.txt   # the LLM's reply — the acceptance signal
cat ~/voicechat-zh/client-out/user_text.txt    # RNNT side-channel ASR (garbage on ZH is expected, phase-1 caveat)
```

If the ws handshake fails, retry with `--server wss://localhost:9000`
(depends on how the server terminates TLS). File input streams the clip plus
~20 s of trailing silence, so each run takes clip-length + ~20 s.

### G4 dual-GPU variant (LLM on GPU 0, TTS on GPU 1)

On a 2-GPU box, replace the Step-1 server command with the one at the end of
this subsection. Two changes make it work, plus three env vars.

**Change 1 — patch `load_utils.py` (this repo:
`opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py`).
The TTS vLLM engine is created after the LLM engine; vLLM `EngineCore`
subprocesses inherit the environment at creation time, so the patch
temporarily swaps `CUDA_VISIBLE_DEVICES` to `TTS_CUDA_VISIBLE_DEVICES`
around `load_tts_model_in_vllm(...)` and restores it right after (the rest
of the Triton stub keeps seeing both GPUs). Unset, behavior is stock.
Mount it over the image copy:

```bash
cp /tmp/repo/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py \
   ~/voicechat-zh/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/
```

**Change 2 — pin the Triton instance to GPU 0** (`gpus: [ 0 ]` in
`config.pbtxt`'s `instance_group`). This is the root cause of
duplicate-instance OOMs: for a `KIND_GPU` model without `gpus`, Triton
creates `count` instances **per visible GPU** — two GPUs × count 1 = two
full model copies (stub ≈3.2 G + LLM + TTS each), guaranteed OOM. With this
repo's patched `import_utils.py`, G3 generates the field automatically; for
a model repo generated before the patch, hand-edit
`model-repo/nemotron-voicechat/config.pbtxt`:

```
instance_group [ { count: 1  kind: KIND_GPU  gpus: [ 0 ] } ]
```

**Env vars** (verified values; memory layout: GPU 0 ≈ 350 M main + 3.3 G
stub + ~31 G LLM engine, GPU 1 ≈ 350 M main + 3.8 G TTS engine):

| Variable | Value | Notes |
|---|---|---|
| `TTS_CUDA_VISIBLE_DEVICES` | `1` | activates the load_utils patch; TTS lands on physical GPU 1 |
| `LLM_GPU_MEM_UTIL` | `0.65` | stock default 0.45. 0.85 failed vLLM startup precheck with both stubs resident (37.49 G free < 37.84 G needed); raise only while watching nvidia-smi |
| `TTS_GPU_MEM_UTIL` | `0.4` | stock default; it's a cap, not a reservation (measured ≈3.8 G) |

**Server command** (dual-GPU; note `-d`, `--gpus all`, and that
`DEFAULT_INSTRUCTIONS` is still required for the Chinese test):

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

Ops notes: `-d` not `-it` (a closed terminal SIGHUPs the container). Both
GPUs must be visible (`--gpus all`, or `--gpus '"device=0,1"'`). Before a
restart, check `nvidia-smi` for orphaned `VLLM::EngineCore` processes
holding memory and kill them first. Startup is slow (Triton health-wait up
to 1200 s + warmup client); `[TRITON]`-prefixed lines in `docker logs -f`
come from `run_s2s_server.sh`'s sed and are normal. `load_utils`'s logger
has no INFO handler in the backend path — verify engine placement with
`nvidia-smi`, not logs. One instance per box.

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

**G3 fails loading the LLM tokenizer (`Network is unreachable`,
`PermissionError at …/token`, or `TypeError: PosixPath / NoneType`)** — one of
the three patched deploy files isn't mounted, or `-e NEMO_LLM_LOCAL_DIR` is
missing, so the stock image code tried the HF hub. Check the G3 command has
all three `-v …import_utils.py / …convert_nemotronllm_checkpoint.py /
…convert_eartts_checkpoint.py` mounts plus `NEMO_LLM_LOCAL_DIR`, and that
`~/voicechat-zh/nano9b/config.json` exists (`vocab_size=131072,
hidden_size=4480`). You should see `NEMO_LLM_LOCAL_DIR set - …` in the logs
before the failing step.

## Rollback

Serve the English stack again at any time by pointing the serving container at a
model repo generated from `hf-checkpoint` (or the NGC prebuilt) — the ZH
checkpoint is a separate directory and nothing in the flow is destructive.
