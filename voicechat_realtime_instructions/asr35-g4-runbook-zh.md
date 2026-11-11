# G4 静默故障操作手册

除第 0 步外，所有命令均在 GPU 服务器用 Bash 执行。按编号在同一个 shell 运行。已有的原版/替换版客户端 A/B 有效，本轮采集缺少的产物和逐帧数据。所有命令保留原 RNNT。

## 0　把随附脚本传到服务器

在本机的仓库根目录执行。输入平时登录 GPU 服务器的 SSH 地址，如 `thao@实际主机名`。若已在服务器上同步了此脚本，直接将它复制到下面的目标目录。

```bash
printf '服务器 SSH 地址: '
read -r DEST
scp voicechat_realtime_instructions/asr35-g4-ops.sh "$DEST:/home/thao/voicechat-zh/scripts/asr35-g4-ops.sh"
```

预期传输完成。服务器上运行 `bash -n /home/thao/voicechat-zh/scripts/asr35-g4-ops.sh` 应退出 0。

## 1　确认路径，建立本次输出目录

```bash
bash
set -euo pipefail
source /home/thao/voicechat-zh/scripts/asr35-g4-ops.sh
```

```bash
g4_init
```

预期两条模型仓库路径不同，`RUN` 为新目录。这里按现场 root 账号编写。若只有 checkpoint、没有 EN Triton 仓库，先用部署指南 Step 4 从原 checkpoint 导出到单独的空目录，再设置 `EN_REPO`。若宿主机只有一个曾被覆盖的模型目录，先找回失败时的 ZH 产物；新导出的结果应标记为新部署。

## 2　在线做 CPU 核对

```bash
g4_audit
```

如果 audit 失败使专用 Bash 退出，可以重新进入 Bash、source 脚本并运行 `g4_init`，再按 2A 操作。预期 `context=[56, 0]`，三组张量 `diff=0`，tokenizer 一致，`missing=[] unexpected=[]`，最后为 `AUDIT_PASS`。配置/张量不同先做 2A。仅有 missing/unexpected 时，在原版上检查同名键并查实际容器 NeMo 版本；记录异常键，勿以反复导出代替检查。Python traceback 或容器 OOM 表示核对未完成。

### 2A　仅当怀疑旧目录混装时重新导出

在空闲 GPU 或维护窗口运行，先检查磁盘空间和 GPU 显存。默认使用 GPU 0。保留原目录。以下路径沿用现场 G3 离线文件布局。

```bash
g4_rebuild
```

预期导出完成，重新核对通过。若仍出现同样差异，保留两份 audit 日志并停在此处检查 G2、原版来源及加载键。

## 3　修正 G1 的轴向，在 CPU 上重跑六条录音

使用脚本副本，保留原始报告。CPU 运行较慢，可占用大量主机内存；资源不足时在维护窗口单独运行。容器输出目录是本次新建的 `g1`。

```bash
g4_g1
```

预期原六条样本均有结果；使用同一批 checkpoint 时仍为 `exact=638`、`shape_mismatch=0`。新 cosine 没有预设目标值。若日志出现 `could not pin causal mode` 或 `cannot resample`，先处理该告警，再解释指标。

## 4　生成诊断版 backend 副本

从**当前原版服务容器**拷贝实际文件，在副本增加日志。原版和 ZH 后续挂载同一副本。若失败服务原先用过不同 backend 挂载，先记录该差异。命令假设 NIM backend 使用标准路径。

```bash
g4_trace
```

预期生成 `model.trace.py`，无异常。若报“backend 版本不匹配”，保留 `model.original.py`，停止此步骤，不把本地仓库的另一版本直接覆盖上去。日志会影响耗时，此轮只比较帧和输出内容。

## 5　顺序启动 EN、ZH 诊断服务，各跑同一条语音

在空闲测试机或维护窗口做。占用同一组 GPU 时，先记录原版启动命令，确认可以恢复。若原版为 `--rm`，停机后要用原启动命令重建；若为持久容器，可在结束时 `docker start "$STOCK"`。先复制第 4 步文件并读取下列配置，再停原服务。

```bash
g4_prepare_server
```

若安排了同机维护窗口，现在执行 `docker stop "$STOCK"` 释放 GPU。仅在原版恢复命令已留存时执行。先跑 EN；看到正确问候、`what is your name`、相关回答以及非空 `frames.log` 后再切 ZH。

```bash
start_diag "$EN_REPO"
run_case en-builtin /s2s/what_is_your_name.wav
cat "$RUN/en-builtin/agent_text.txt"
stop_diag
start_diag "$ZH_REPO"
run_case zh-builtin /s2s/what_is_your_name.wav
run_case zh-mandarin /audio/01_zh_hello_who_are_you.wav
```

若 EN 诊断版也沉默，先比较原服务与诊断服务的启动参数、挂载和 backend 来源；此时不要给 ZH 下结论。若 ZH 已恢复，切回同一镜像中的原始 backend，补测六条录音：

```bash
stop_diag
start_diag "$ZH_REPO" clean
g4_verify_six
```

结束后释放诊断服务并恢复原服务。持久容器使用以下命令；原容器为 `--rm` 时，第二行改为现场记录的原版启动命令。

```bash
stop_diag
docker start "$STOCK"
```

若专用 Bash 在诊断期间因报错退出，用另一个终端执行 `docker ps -a --filter name=vc-diag-`，确认本次诊断容器名，先保存其 `docker logs`，再停止该诊断容器并按原方式恢复服务。

待原服务启动后，执行 `curl -fsS http://localhost:9000/v1/realtime/health`，预期 `status=ok` 且 `triton_status=ready`。

## 6　汇总结果并选择下一步

```bash
g4_summary | tee "$RUN/summary.txt"
```

| 实测结果 | 立即操作 |
|---|---|
| EN 正常、ZH 的 `pcm` 正常，ZH 有 `nonfinite_proj` 或 `nonfinite_rnnt` | 在两份 `frames.log` 按 `step=` 对齐，查首次异常前后 2 个 step。先处理产生 NaN/Inf 的预处理、encoder 或投影。 |
| ZH 有声段 RNNT 全 blank，或有 nonblank 但 `turn` 中 `user/agent` 从未切换 | 对照 EN 同帧的 `rnnt` 和 `turn`。本阶段记录轮次阻塞，保留 RNNT。投影侧结果单独记录。 |
| 汇总显示 EN/ZH PCM mismatched steps 为 0，ZH 原始/投影值有限，但原版问候或回答时段内 ZH 的 `native` 持续为 PAD | 保存 `audit`、修正版 G1 和该时间窗日志，进入投影对齐小试。正式问答训练需要独立的中英成对语料、通过验证的权重回写入口；现有六条保留作验收。 |
| ZH 有正文 token，客户端却无文本 | 检查 `turn.final` 和 `server.log` 的发送错误，核对 Triton 解码文本及 WebSocket transcript 事件。`native_other` 包含其他特殊 token，不能单凭该计数认定有正文。 |
| ZH 有正确助手文本，但 `PCM16_peak=0` | 转查 EAR-TTS、codec 和服务端 PCM 编码。峰值大于零时试听实际回答，同时参考最响 80 ms RMS。 |
| 英文恢复，中文能逐题正确回答 | 用第 5 步 clean 模式复验六条，分别记录 `agent_text.txt` 的回答相关性和 WAV 听感；只问候不算通过。 |

对齐命令示例：查看 step 35–90（约 2.8–7.2 秒）内同一条英文录音的两份记录。时间以模型帧起点为准。

```bash
for label in en-builtin zh-builtin; do
  echo "$label"
  grep -E 'step=(3[5-9]|[4-8][0-9]|90) ' "$RUN/$label/frames.log" | head -n 24 || true
done
```

应留存 `stock.txt`、`audit-*.log`、`g1/compatibility-framewise.json`，以及每个 case 的 `client.log`、`server.log`、`frames.log`、WAV 和实际产生的 TXT。

## 7　仅在第 6 步指向投影失配时做单批次试验

在第 2 步通过、EN 探针正常、两版输入 PCM 对齐、ZH 特征有限且在关键时间窗持续输出 PAD 时执行。使用镜像内独立的英文 warmup 录音，更新一批次的投影权重，目标为原版在该录音上的投影输出。用前六步保留的 Bash 变量执行；`cp --reflink=auto` 在不支持 reflink 的文件系统会完整复制，先核对 `df -h "$RUN"`、`du -sh "$ZH_REPO"`。

```bash
g4_projection_pilot
```

预期 `PILOT_EXPORT_PASS`，有变化的键只涉及 `proj.weight`、`proj.bias`，`loss_after < loss_before`；输出为独立的 `$RUN/pilot-repo`。断言失败就保留 `pilot.log`，跳过下面的部署。在空闲 GPU 或下一次维护窗口：

```bash
start_diag "$PILOT" clean
run_case pilot-en /s2s/what_is_your_name.wav
run_case pilot-zh /audio/01_zh_hello_who_are_you.wav
stop_diag
```

预期两次会话正常执行并各自产生 WAV。记录是否有对应回答，按第 5 步恢复占用同一组 GPU 的原服务。此处仅验证投影可更新、可导出、可回放。英文单批次 loss 下降不计作中文问答训练或中文验收通过。
