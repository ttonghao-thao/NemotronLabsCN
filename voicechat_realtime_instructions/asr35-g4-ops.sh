#!/usr/bin/env bash
# Source this file from Bash on the GPU host. See asr35-g4-runbook-zh.md.

g4_init() {
BASE=/home/thao/voicechat-zh
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
read -rp '输入当前正常原版服务的容器名: ' STOCK
IMAGE=$(docker inspect -f '{{.Image}}' "$STOCK")
docker inspect -f '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' "$STOCK"
EN_REPO=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/data/models"}}{{.Source}}{{end}}{{end}}' "$STOCK")
read -erp '失败的 ZH 模型仓库绝对路径: ' ZH_REPO
test -f "$ZH_REPO/nemotron-voicechat/1/perception.safetensors"
if [[ -z "$EN_REPO" || ! -f "$EN_REPO/nemotron-voicechat/1/config.json" ]]; then
  read -erp '原版模型仓库在宿主机的绝对路径: ' EN_REPO
fi
test -f "$EN_REPO/nemotron-voicechat/1/config.json"
test "$(realpath "$EN_REPO")" != "$(realpath "$ZH_REPO")"
test -f "$BASE/zh-checkpoint/model.safetensors"
test -f "$BASE/hf-checkpoint/model.safetensors"
test -r "$BASE/scripts/verify_asr35_encoder_compatibility.py"
mkdir -p "$BASE/diag"
RUN=$(mktemp -d "$BASE/diag/g4-XXXXXXXX")
OUT_UID=$(docker run --rm --entrypoint id "$IMAGE" -u)
OUT_GID=$(docker run --rm --entrypoint id "$IMAGE" -g)
make_out() { mkdir "$1"; chown "$OUT_UID:$OUT_GID" "$1"; chmod 775 "$1"; }
docker inspect -f 'image={{.Image}} auto_remove={{.HostConfig.AutoRemove}} GPUs={{json .HostConfig.DeviceRequests}}' "$STOCK" | tee "$RUN/stock.txt"
printf 'EN=%s\nZH=%s\nRUN=%s\n' "$EN_REPO" "$ZH_REPO" "$RUN" | tee -a "$RUN/stock.txt"
STOCK_AUTO=$(docker inspect -f '{{.HostConfig.AutoRemove}}' "$STOCK")
printf 'stock=%s auto_remove=%s image=%s\n' "$STOCK" "$STOCK_AUTO" "$IMAGE" | tee -a "$RUN/stock.txt"
}

g4_audit() {
cat > "$RUN/audit.py" <<'PY'
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file
from nemo.collections.speechlm2.modules import AudioPerceptionModule

def config(root):
    return json.loads((Path(root) / "config.json").read_text())

def tensors(label, source, prefix, dest, starts=("",)):
    with safe_open(str(source), framework="pt", device="cpu") as a, safe_open(str(dest), framework="pt", device="cpu") as b:
        left = {k[len(prefix):] for k in a.keys() if k.startswith(prefix) and k[len(prefix):].startswith(starts)}
        right = {k for k in b.keys() if k.startswith(starts)}
        bad = sorted(left ^ right)
        for k in sorted(left & right):
            x, y = a.get_tensor(prefix + k), b.get_tensor(k)
            if x.dtype != y.dtype or not torch.equal(x, y):
                bad.append(k)
        ok = bool(left) and not bad
        print(label, "n=", len(left), "diff=", len(bad), bad[:8], flush=True)
        return ok

def tokenizer_files(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}

stock, live = Path("/en/1"), Path("/zh/1")
p2 = config("/g2")["model"]["stt"]["model"]["perception"]
stt = config(live)["model"]["stt"]["model"]
p3 = stt["perception"]
ok = (p2 == p3 and p3["encoder"]["att_context_size"] == [56, 0]
      and stt.get("pretrained_weights") is False and not stt.get("pretrained_asr"))
print("perception config equal=", p2 == p3, "context=", p3["encoder"]["att_context_size"],
      "pretrained_weights=", stt.get("pretrained_weights"), "pretrained_asr=", stt.get("pretrained_asr"))
ok &= tensors("G2 -> G3 perception", "/g2/model.safetensors", "stt_model.perception.", live/"perception.safetensors")
ok &= tensors("EN -> ZH RNNT", stock/"rnnt-asr.safetensors", "", live/"rnnt-asr.safetensors")
ok &= tensors("EN -> ZH projection", stock/"perception.safetensors", "", live/"perception.safetensors", ("proj.",))
old_tok = tokenizer_files(stock/"rnnt_tokenizer")
tok_ok = bool(old_tok) and old_tok == tokenizer_files(live/"rnnt_tokenizer")
print("RNNT tokenizer equal=", tok_ok)
ok &= tok_ok
cfg = OmegaConf.create(p3)
cfg.output_dim = 4480
module = AudioPerceptionModule(cfg)
keys = module.load_state_dict(load_file(str(live/"perception.safetensors")), strict=False)
print("missing=", keys.missing_keys, "unexpected=", keys.unexpected_keys)
ok &= not keys.missing_keys and not keys.unexpected_keys
print("AUDIT_PASS" if ok else "AUDIT_REVIEW")
raise SystemExit(0 if ok else 2)
PY
chmod 644 "$RUN/audit.py"
audit() {
  docker run --rm -w /tmp \
    -v "$BASE/zh-checkpoint:/g2:ro" \
    -v "$EN_REPO/nemotron-voicechat:/en:ro" \
    -v "$ZH_REPO/nemotron-voicechat:/zh:ro" \
    -v "$RUN/audit.py:/tmp/audit.py:ro" \
    --entrypoint python "$IMAGE" -u /tmp/audit.py 2>&1 | tee "$RUN/audit-$(date +%s).log"
}
audit
}

g4_rebuild() {
df -h "$BASE"
nvidia-smi
FRESH="$RUN/model-repo-fresh"
make_out "$FRESH"
IMPORT=opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/import_utils.py
for f in "$IMPORT" s2s/scripts/convert_nemotronllm_checkpoint.py s2s/scripts/convert_eartts_checkpoint.py nano9b/config.json; do test -f "$BASE/$f"; done
docker run --rm --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -v "$BASE/zh-checkpoint:/checkpoint:ro" -v "$FRESH:/data/models" \
  -v "$BASE/$IMPORT:/$IMPORT:ro" \
  -v "$BASE/s2s/scripts/convert_nemotronllm_checkpoint.py:/s2s/scripts/convert_nemotronllm_checkpoint.py:ro" \
  -v "$BASE/s2s/scripts/convert_eartts_checkpoint.py:/s2s/scripts/convert_eartts_checkpoint.py:ro" \
  -v "$BASE/nano9b:/llm/Nemotron-Nano-9B-v2:ro" \
  -e NEMO_CHECKPOINT_PATH=/checkpoint -e NEMO_LLM_LOCAL_DIR=/llm/Nemotron-Nano-9B-v2 \
  --entrypoint /s2s/deploy_s2s_model.sh "$IMAGE" 2>&1 | tee "$RUN/export.log"
ZH_REPO="$FRESH"
g4_audit
}

g4_g1() {
make_out "$RUN/g1"
python3 - "$BASE/scripts/verify_asr35_encoder_compatibility.py" "$RUN/g1/verify.py" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
old = "    return emb[0].float().cpu(), int(emb_len[0])"
new = """    n = int(emb_len[0])
    assert emb.ndim == 3 and emb.shape[1] == 1024 and 0 < n <= emb.shape[2], (emb.shape, n)
    return emb[0, :, :n].T.contiguous().float().cpu(), n"""
assert s.count(old) == 1, "服务器 G1 源码版本不同，请核对 encode_side"
s = s.replace(old, new)
anchor = "        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)"
assert s.count(anchor) == 1
s = s.replace(anchor, """        assert a.shape == b.shape == (T, 1024)
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), fname
""" + anchor)
Path(sys.argv[2]).write_text(s)
PY
chmod 644 "$RUN/g1/verify.py"
docker run --rm --shm-size=8GB -v "$BASE:/work:ro" -v "$RUN/g1:/out" -w /tmp \
  --entrypoint python "$IMAGE" -u /out/verify.py \
  --asr /work/models/nemotron-3.5-asr-streaming-0.6b.nemo \
  --voicechat-checkpoint-dir /work/hf-checkpoint --audio-dir /work/audio \
  --device cpu --report /out/compatibility-framewise.json 2>&1 | tee "$RUN/g1.log"
python3 - "$RUN/g1/compatibility-framewise.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(r["key_alignment"])
for row in r["latent"]["per_file"]:
    print(row["file"], row["frames"], row["cos_mean"], row["old_mean_abs"], row["new_mean_abs"])
print(r["latent"]["per_language_mean"])
PY
}

g4_trace() {
BACK=/opt/tritonserver/backends/nemotron-voicechat
docker cp "$STOCK:$BACK/model.py" "$RUN/model.original.py"
python3 - "$RUN/model.original.py" "$RUN/model.trace.py" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
def add(anchor, code):
    global s
    assert s.count(anchor) == 1, "backend 版本不匹配: " + anchor
    s = s.replace(anchor, code + "\n" + anchor)

add("            current_input_embeds = (", """            feat = encoded_audio[batch_idx, audio_idx].float()
            logger.log_info(f"G4TRACE proj seq={seq_id} step={seq_state.decoder_global_step} idx={audio_idx} rms={feat.square().mean().sqrt().item():.6g}")""")
add("            if seq_state.fn_call[\"state\"] == \"idle\":\n                seq_state.generated_text_tokens[:, seq_state.decoder_global_step]", """            native = int(result_batched[batch_idx]["token"])
            logger.log_info(f"G4TRACE token seq={seq_id} step={seq_state.decoder_global_step} native={native} pad={self.pad_id} bos={self.model.tokenizer.bos_id} eos={self.model.tokenizer.eos_id}")""")
add("                if not seq_state.rnnt_agent_speaking:\n                    seq_state.rnnt_rolling_density", """                raw_rms = f[i].float().square().mean().sqrt().item()
                pb_utils.Logger.log_info(f"G4TRACE rnnt seq={seq_id} step={seq_state.decoder_global_step} idx={frame_indices[i]} blank={int(_blank)} raw_rms={raw_rms:.6g}")""")
add("                if asr_emb is not None and self.model.rnnt_decoder is not None:\n                    self._rnnt_step_transcript_display(", """                    st = self._seq_mgr[seq_id]
                    emitted = int(st.generated_text_tokens[0, st.decoder_global_step])
                    logger.log_info(f"G4TRACE turn seq={seq_id} step={st.decoder_global_step} final={emitted} user={int(st.rnnt_user_speaking)} agent={int(st.rnnt_agent_speaking)}")""")
add("        encoded_audio, asr_emb = self.perception(audio_to_encode, buffer_len_tensor)", """        for i, seq_id in enumerate(seq_ids_batched):
            import zlib
            pcm = audio_to_encode[i, -self.options.num_samples_per_chunk * self.options.num_chunks_per_inference:].float()
            checksum = zlib.crc32(pcm.detach().cpu().contiguous().numpy().tobytes())
            logger.log_info(f"G4TRACE pcm seq={seq_id} step={self._seq_mgr[seq_id].decoder_global_step} n={pcm.numel()} crc={checksum} rms={pcm.square().mean().sqrt().item():.6g}")""")
compile(s, sys.argv[2], "exec")
Path(sys.argv[2]).write_text(s)
PY
chmod 644 "$RUN/model.trace.py"
}

g4_prepare_server() {
stock_env() { docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$STOCK" | sed -n "s/^$1=//p"; }
EXTRA=()
for k in DEFAULT_INSTRUCTIONS TTS_CUDA_VISIBLE_DEVICES LLM_GPU_MEM_UTIL TTS_GPU_MEM_UTIL; do
  val=$(stock_env "$k")
  if [[ -n "$val" ]]; then EXTRA+=(-e "$k=$val"); fi
done
LOAD_MOUNT=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/opt/tritonserver/backends/nemotron-voicechat/checkpoint_utils/load_utils.py"}}{{.Source}}{{end}}{{end}}' "$STOCK")
if [[ -n "$LOAD_MOUNT" ]]; then EXTRA+=(-v "$LOAD_MOUNT:$BACK/checkpoint_utils/load_utils.py:ro"); fi
docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$STOCK"
read -rp '按原服务填写 GPU 选择，双卡填 all，单卡 0 填 device=0: ' GPU_SPEC
# 如原版另有影响推理的环境参数或 backend 挂载，同样加进 EXTRA。
DIAG="vc-diag-$(basename "$RUN")"
start_diag() {
  local repo=$1
  G4_MODE=${2:-trace}
  local mounts=()
  test -r "$repo/nemotron-voicechat/1/config.json"
  case "$G4_MODE" in
    trace) test -r "$RUN/model.trace.py"; mounts+=(-v "$RUN/model.trace.py:$BACK/model.py:ro") ;;
    clean) mounts+=(-v "$RUN/model.original.py:$BACK/model.py:ro") ;;
    *) echo 'mode must be trace or clean' >&2; return 1 ;;
  esac
  docker run -d --name "$DIAG" --runtime=nvidia --gpus "$GPU_SPEC" --shm-size=8GB \
    -p 127.0.0.1:9100:9000 -e NIM_HTTP_API_PORT=9000 \
    "${EXTRA[@]}" -v "$repo:/data/models:ro" "${mounts[@]}" \
    --entrypoint /s2s/run_s2s_server.sh "$IMAGE"
  for ((i=0; i<150; i++)); do
    if curl -fsS http://localhost:9100/v1/realtime/health 2>/dev/null | python3 -c \
      'import json,sys; r=json.load(sys.stdin); sys.exit(0 if (r.get("status"),r.get("triton_status"))==("ok","ready") else 1)' 2>/dev/null; then
      echo READY; return 0
    fi
    [[ $(docker inspect -f '{{.State.Running}}' "$DIAG") == true ]] || break
    sleep 10
  done
  docker logs --tail 100 "$DIAG"
  return 1
}
run_case() {
  local label=$1 wav=$2 dir="$RUN/$1"
  if [[ -e "$dir" ]]; then echo "输出目录已存在，请换一个 label: $dir" >&2; return 1; fi
  make_out "$dir"
  local start sid seq
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  docker run --rm --network=host -v "$BASE/audio:/audio:ro" -v "$dir:/out" -w /out \
    --entrypoint python "$IMAGE" -u /s2s/nemotron-voicechat-client.py \
    --server ws://localhost:9100 --input-file "$wav" --no-playback --verbose 2>&1 | tee "$dir/client.log"
  test -s "$dir/received_audio.wav"
  docker logs --since "$start" "$DIAG" > "$dir/server.log" 2>&1
  if [[ "$G4_MODE" == clean ]]; then ls -lh "$dir"; return 0; fi
  sid=$(sed -n 's/^Session created: //p' "$dir/client.log" | head -1)
  test -n "$sid"
  seq=$(sed -n "/Client $sid:/s/.*sequence_id=\([0-9]*\).*/\1/p" "$dir/server.log" | head -1)
  test -n "$seq"
  grep -F "seq=$seq " "$dir/server.log" | grep -F G4TRACE > "$dir/frames.log" || { echo "该 sequence 无 G4TRACE，检查 backend 挂载与 server.log" >&2; return 1; }
  echo "$label: session=$sid seq=$seq trace_rows=$(wc -l < "$dir/frames.log")"
  ls -lh "$dir"
}
stop_diag() { docker stop "$DIAG" && docker rm "$DIAG"; }
}

g4_verify_six() {
for f in "$BASE"/audio/0[1-3]_*.wav; do
  test -f "$f"
  run_case "verify-$(basename "$f" .wav)" "/audio/$(basename "$f")"
done
}

g4_summary() {
python3 - "$RUN" <<'PY'
import collections, math, re, sys, wave
from array import array
from pathlib import Path

pcm_by_case = {}
for p in sorted(Path(sys.argv[1]).glob("*/frames.log")):
    counts = collections.Counter()
    first = {}
    pcm_by_case[p.parent.name] = {}
    for line in p.read_text().splitlines():
        kind = line.split("G4TRACE ", 1)[1].split()[0]
        d = dict(re.findall(r"(\w+)=([^\s\"]+)", line))
        if kind == "token":
            tag = next((k for k in ("pad","bos","eos") if d["native"] == d[k]), "other")
            event = "native_" + tag
        elif kind == "rnnt":
            event = "blank" if d["blank"] == "1" else "nonblank"
        else:
            event = kind
        counts[event] += 1
        first.setdefault(event, d["step"])
        if kind == "pcm":
            pcm_by_case[p.parent.name][int(d["step"])] = (int(d["n"]), int(d["crc"]))
        for k in ("rms", "raw_rms"):
            if k in d and not math.isfinite(float(d[k])):
                counts["nonfinite_" + kind] += 1
    with wave.open(str(p.parent/"received_audio.wav")) as wav:
        assert (wav.getsampwidth(), wav.getnchannels()) == (2, 1)
        x = array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little": x.byteswap()
        hop = max(1, round(0.08 * wav.getframerate()))
        rms = max((math.sqrt(sum(v*v for v in x[i:i+hop])/len(x[i:i+hop])) for i in range(0,len(x),hop)), default=0)
    print(p.parent.name, dict(counts), "first_steps=", first,
          "PCM16_peak=", max(map(abs,x),default=0), "loudest_80ms_RMS=", round(rms,2))
en = pcm_by_case.get("en-builtin", {})
zh = pcm_by_case.get("zh-builtin", {})
if en and zh:
    diff = sorted(k for k in en.keys() | zh.keys() if en.get(k) != zh.get(k))
    print("EN/ZH PCM mismatched steps:", len(diff), "first:", diff[:8])
PY
}

# Conditional one-batch export/reload check using the image's English warmup recording.
g4_projection_pilot() {
  PILOT="$RUN/pilot-repo"
  test ! -e "$PILOT"
  df -h "$RUN"
  du -sh "$ZH_REPO"
  mkdir "$PILOT"
  cp -a --reflink=auto "$ZH_REPO/." "$PILOT/"
  local version="$PILOT/nemotron-voicechat/1"
  test "$(realpath "$version")" = "$(realpath "$PILOT")/nemotron-voicechat/1"
  test ! -L "$version/perception.safetensors"
  # The copy may retain read-only owner permissions. Only its version directory needs to be writable.
  chown "$OUT_UID:$OUT_GID" "$version"
  chmod u+w "$version"
  docker run --rm -i -w /tmp -v "$EN_REPO/nemotron-voicechat/1:/en:ro" \
    -v "$ZH_REPO/nemotron-voicechat/1:/zh:ro" -v "$version:/pilot" \
    --entrypoint python "$IMAGE" -u - <<'PY' 2>&1 | tee "$RUN/pilot.log"
import json
import os
from pathlib import Path
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from nemo.collections.speechlm2.modules import AudioPerceptionModule

wav, sr = sf.read("/s2s/en-US_sample_extended.wav", dtype="float32", always_2d=True)
wave = torch.from_numpy(wav.mean(axis=1)).unsqueeze(0)
wave = torchaudio.functional.resample(wave, sr, 16000)
length = torch.tensor([wave.shape[-1]])
torch.manual_seed(0)

def forward(root, capture_input):
    cfg = json.loads((Path(root)/"config.json").read_text())["model"]["stt"]["model"]["perception"]
    cfg["output_dim"] = 4480
    model = AudioPerceptionModule(OmegaConf.create(cfg)).float().eval()
    model.load_state_dict(load_file(str(Path(root)/"perception.safetensors")), strict=True)
    assert isinstance(model.proj, torch.nn.Linear) and model.proj.bias is not None
    saved = []
    if capture_input:
        model.proj.register_forward_pre_hook(lambda m, args: saved.append(args[0].detach()))
    with torch.no_grad():
        out, n = model(input_signal=wave, input_signal_length=length)
    frames = int(n[0])
    feat = saved[0] if capture_input else out
    assert feat.ndim == 3 and 0 < frames <= feat.shape[1]
    return feat[0, :frames].clone()

target = forward("/en", False)
x = forward("/zh", True)
assert x.shape[0] == target.shape[0], (x.shape, target.shape)
assert torch.isfinite(x).all() and torch.isfinite(target).all()
path = Path("/pilot/perception.safetensors")
state = load_file(str(path))
w = state["proj.weight"].float().clone().requires_grad_()
b = state["proj.bias"].float().clone().requires_grad_()
def loss(weight, bias):
    return (F.linear(x, weight, bias) - target).square().sum(dim=-1).mean()
before = loss(w, b)
before.backward()
lr = 0.25 / (1 + x.square().sum(dim=-1).mean().item())
changed = []
for key, param in (("proj.weight", w), ("proj.bias", b)):
    assert torch.isfinite(param.grad).all()
    new = (param.detach() - lr * param.grad).to(state[key].dtype).contiguous()
    if not torch.equal(new, state[key]):
        changed.append(key)
    state[key] = new
after = loss(state["proj.weight"].float(), state["proj.bias"].float()).item()
assert changed and after < before.item(), (changed, before.item(), after)
temp = path.with_suffix(".pilot.tmp")
save_file(state, str(temp))
reloaded = load_file(str(temp))
assert state.keys() == reloaded.keys() and all(torch.equal(v, reloaded[k]) for k,v in state.items())
os.replace(temp, path)
print("PILOT_EXPORT_PASS", "changed=", changed, "loss_before=", before.item(), "loss_after=", after)
PY
}
