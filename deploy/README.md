# AI学习台灯后端运行指南

接口以 `docs/ARCHITECTURE.md` 为准：第8节覆盖旧判题方案，第9节覆盖旧语音方案。

## Mac开发（无需GPU）

前置：Python 3.11、[uv](https://docs.astral.sh/uv/)、项目根目录 `.env`。
云端密钥仅填 `.env` 的 `CLOUD_LLM_API_KEY` / `HWR_API_KEY`；不要复制进脚本。

```bash
cd backend
uv sync
# 默认依赖不安装torch/funasr/任何模型。
CAMERA_BACKEND=mock AUTO_CAPTURE=false TTS_PROVIDER=mock \
  FIXTURE_TEMPLATE="$PWD/tests/fixtures/template.json" \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

真实模板未完成时，显式设置 `FIXTURE_TEMPLATE` 加载手工量取的数学前三题。
正式运行**去掉**该变量。无 `template.json` 的目录自动跳过，不伪造题库。
相机打不开会降级mock；`CAMERA_INDEX=-1`也禁用真实相机。
未安装语音依赖时health的`stt`/`vad`标红是预期，批改和文字对话不受影响。
`TTS_PROVIDER=mock`只产生静音PCM，验证音频协议，不代表真实语音已可用。

```bash
# 从项目根目录执行
uv run --project backend python tools/synth_capture.py \
  --template backend/tests/fixtures/template.json --mode mixed --seed 0 --count 3
uv run --project backend python tools/build_features.py --fixture backend/tests/fixtures/template.json
curl http://127.0.0.1:8000/api/health
curl --max-time 2 'http://127.0.0.1:8000/api/camera/stream?fps=20&width=1280' -o /tmp/lamp.mjpeg
# 无限流被max-time中止返回curl 28是正常的，检查Content-Type和收到的字节。
curl -X POST http://127.0.0.1:8000/api/debug/inject-image \
  -F 'grade=true' -F 'file=@data/mock_captures/fixture_math_xmyz_p01_mixed_001.jpg'
```

MJPEG最多2路，1280预览在线程内只编码一次、所有订阅者复用；其他width变体按帧缓存。
抓拍照片内存LRU保留20张，重启清空会话。`AUTO_CAPTURE=true`时mock循环“拿走1秒—放稳7秒—拿走1秒”。
注入照片之后mock固定显示注入图，方便调试；重启恢复循环。

### 可选本地语音

```bash
cd backend
uv sync --extra voice --extra kokoro
STT_PROVIDER=sensevoice STT_DEVICE=cpu TTS_PROVIDER=kokoro \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

首次使用才下载/加载模型，首次会更慢。Kokoro中文前处理依赖可能需要系统 `espeak-ng`。
依赖或模型加载失败会降级为文字；**Mac语音模型真实推理尚未验收**。
全双工需要Silero；无VAD可用PTT，但仍需要STT。浏览器麦克风仅在localhost或HTTPS可用。

## Linux + RTX4090

### 当前目标机的后台托管（2026-11-17）

当前服务由用户级 systemd 临时单元 `dahetao-study-lamp.service` 托管，不再依附 Claude 终端标签。
异常退出时等待10秒重启，10分钟内最多允许3次启动，避免无限重启。
子服务日志仍在 `deploy/logs/`，启动器日志为 `deploy/logs/service-manager.log`。

查看状态：

```bash
systemctl --user status dahetao-study-lamp.service --no-pager
```

修改代码后重启整套服务（不要直接杀子进程，否则可能触发自动拉起）：

```bash
systemctl --user restart dahetao-study-lamp.service
```

有意停服：

```bash
systemctl --user stop dahetao-study-lamp.service
```

`active` 只代表启动脚本在运行，必须等模型加载和 TTS 预热完成，再确认健康接口：

```bash
curl --noproxy '*' -sk https://172.17.114.10:8000/api/health
```

这是临时单元，**未配置开机自启**。目标机 `Linger=no`，不能承诺退出全部登录会话后继续运行。
如需跨系统重启或全部注销后常驻，需另行部署持久用户单元并明确配置登录生命周期。
下文直接运行 `start_4090.sh` / `nohup` 的方式仅用于未由 systemd 托管的环境；不要与当前单元同时运行两套。


1. 安装与驱动匹配的CUDA/PyTorch。vLLM分别放独立环境（不要与PaddleOCR解析器环境混装）。
2. 安装支持 `Qwen3_5ForConditionalGeneration`、`Qwen3ASRForConditionalGeneration` 的当前稳定vLLM。
   HWR最低要求沿用0.17，但ASR需额外确认本机版本原生支持`/v1/audio/transcriptions`和`prompt`偏置；
   不能只凭版本号假设可用。依据当前上游文档：
   https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_asr.py
3. 后端环境安装语音extra（不安装vLLM本身）。预下载TTS/Silero权重以降低首次等待。

```bash
cd backend
uv sync --extra voice --extra qwen-tts
cd ..
HWR_VLLM_BIN=/opt/venvs/hwr/bin/vllm \
ASR_VLLM_BIN=/opt/venvs/asr/bin/vllm \
  bash deploy/start_4090.sh
```

脚本严格顺序启动HWR(8001) → ASR(8003) → FastAPI(8000，单worker)。
两个vLLM的`gpu-memory-utilization`均为0.35，但KV缓存显式封顶为HWR 1.5GiB、ASR 1GiB；
不要把比例误当成最终显存占用，务必看`nvidia-smi`。后端默认开启`TTS_WARMUP=true`，
启动时完整合成一句短句（不广播），完成权重加载和CUDA graph捕获后才接受连接。
`/api/health`的`details.tts_warmed`必须为true；预热失败时后端仍提供文字/批改，但`ok`/`tts`标红。
Mac默认`TTS_WARMUP=false`，不因这个改动加载任何模型。
每个服务等待健康端点就绪，进程提前退出或15分钟超时会结束启动；任一服务退出清理其他子进程。
日志在`deploy/logs/`，不要开启shell `set -x`。脚本不会改 `.env`。
`HWR_MODEL_PATH`、`ASR_MODEL_PATH`、`TTS_MODEL`可指向离线模型目录。

ASR使用OpenAI兼容multipart `/v1/audio/transcriptions`，16kHz单声道WAV，`prompt`传当前题的题干/选项/tag。
每600ms累计识别partial，默认静音700ms后final（`VAD_END_SILENCE_MS`可设200–3000ms）。
final后1.2秒内、LLM首个delta前同一输入续说，会撤回临时轮次并合并，不显示“打断”。
已经开始出字/播音的真正插话仍走duck与barge-in。默认不强制language，以保留中英混读。
确认本机ASR服务接口：

```bash
curl http://127.0.0.1:8003/v1/audio/transcriptions \
  -F model=Qwen3-ASR-1.7B -F 'file=@sample.wav' \
  -F 'prompt=条件状语从句，主将从现，whether' -F response_format=json
```

TTS默认Qwen3-TTS-12Hz-1.7B-CustomVoice，`Vivian`中文女声。语气统一为自然平实、情绪克制，不再按关键词切换夸张表扬。
`TTS_SPEED=1.2` 默认在服务端使用系统 `libsonic` 保音调加速，浏览器保持1倍播放，避免音高升高或重复变速。
Linux需 `libsonic0`（当前目标机已有）；其他环境缺库时请安装对应库或显式设 `TTS_SPEED=1`，不静默忽略倍速。
文字和语音独立完成：`chat.message_done` 不等待 `tts.end`，朗读仍可被停止。缓存仅为文字且受 `CHAT_MAX_CHARS` 限制，音频仍逐段生成。
新增日志 `llm_queue_wait`、`llm_done`（请求ID/模型/字符数/结束原因）、`chat_text_done`、`tts_queue_wait`、`tts_segment_done`、`tts_done`，不记录密钥或正文。
`TTS_MODEL`可改0.6B，`TTS_PROVIDER=kokoro|mock`可轻量回退。
如使用单独部署的vLLM-Omni，设置`TTS_PROVIDER=vllm_omni`、`TTS_BASE_URL=http://127.0.0.1:8004/v1`，
适配器调用`/audio/speech`并要求24kHz PCM16LE流。该备选需服务端模型配置匹配，未做GPU验收。

### 国内网络、无sudo/gcc的目标机

`deploy/setup_models.sh`用清华PyPI镜像与ModelScope下载。ASR所在的**vLLM环境**必须装
`librosa soundfile av`（不是只装后端环境）；缺失时`/models`和`/health`仍返回200，
但真实WAV转写返回400 `Invalid or unsupported audio file`。修复依赖后须重启ASR进程。
TTS extra约束`transformers>=5.15.1,<5.16`，目标机实测5.15.1；勿无条件升级到5.16。
无系统gcc时脚本自动选择`deploy/bin/cc`（ziglang编译器包装）。
`sox: not found`警告不阻断当前CustomVoice流式路径；其他需要SoX的模式未经验证。

Mac源码同步到目标机，保留目标机独立配置和镜像锁文件：

```bash
rsync -az --exclude='.venv*' --exclude='models/' --exclude='deploy/logs/' \
  --exclude='deploy/certs/' --exclude='__pycache__/' --exclude='.env' \
  --exclude='backend/uv.lock' ./ Claude-114-10:dahetao/
ssh Claude-114-10
cd ~/dahetao
# 仅停止启动脚本；trap负责清理子进程。不要pkill -f模糊关键字。
kill $(pgrep -f '^bash deploy/start_4090.sh')
# 等旧组全部退出再启动；不要同时启动两套抢占端口/显存。
(setsid nohup bash deploy/start_4090.sh > deploy/logs/start_4090.log 2>&1 < /dev/null &)
```

证书存在`deploy/certs/cert.pem`与`key.pem`时自动HTTPS；内网访问
`https://172.17.114.10:8000/`，浏览器首次需接受自签证书。Mac开着代理时：

```bash
curl --noproxy '*' -sk https://172.17.114.10:8000/api/health
```

### WebSocket语音验收

素材只需生成一次，**只在4090运行**。生成程序会加载另一份TTS，须在后端TTS未加载时
（例如首次启动前）执行，避免占用双份模型显存。生成后进程退出释放显存。

```bash
HF_HUB_OFFLINE=1 backend/.venv/bin/python tools/synth_voice.py --output /tmp/lamp-voice
# 启动服务组、健康检查完成后：
backend/.venv/bin/python tools/voice_acceptance.py --fixtures /tmp/lamp-voice \
  --output /tmp/lamp-voice-results
# 支持--only voice|text|ack|barge|tail分别复测；默认全部测试（tail为合成结束后的播放缓存）。
```

客户端通过真实`wss://172.17.114.10:8000/ws`发送20ms/644字节PCM16帧，
自动重采样16kHz、前后加静音并模拟24kHz音频播放回执；验证事件顺序、轮次、
实际音频负载、打断后的旧帧丢弃、附和duck/恢复。事件与每轮WAV保存到输出目录。
不依赖模型包，复制素材后也可在Mac运行（只发请求，不加载模型）。会实际调用云端LLM。
测试期间不要另开麦克风/发起对话；它临时关闭自动抓拍，结束恢复原模式和选中题目，
不清空批改数据。`input_end_ms`从测试音频最后一帧发送算（旧版输出名为`acoustic_end_ms`），
它不是人工标注的言语终点：音频里的尾部呼吸/噪音可能晚于VAD判停，所以长句可出现负差值。
`vad_end_ms`从服务端`vad.state(speaking=false)`算，是本项目speech_end延迟的主基准，
不包含默认约700ms静音判停；无明显尾部噪音的短句可同时用input_end观察包含判停的体感延迟。
`--only slow`用带约900ms思考停顿的长问题验收合并，断言多次final最终只落一轮、无误打断；
它单独运行，不包含在默认五项里。
这些程序验收不替代真实扬声器/麦克风回声与浏览器AudioWorklet设备验收。

### 离线OCR

线上不启动PaddleOCR-VL。建模板时单独启动（建议先停线上组，避免显存超预算）：

```bash
OCR_VLLM_BIN=/opt/venvs/ocr/bin/vllm bash deploy/start_ocr.sh
# 用项目的tools/ocr_pages.py离线建库，完成后停止OCR服务。
uv run --project backend python tools/build_features.py
```

## 放卷自动识别与临时采样

见 [CAPTURE_SAMPLES.md](../docs/CAPTURE_SAMPLES.md)：工具栏自动识别状态、录像入口、同页答题区去重及离线回放步骤。
录像复用现有相机，不录音；开启 `RECORDING_ENABLED=true` 后由用户手动开始。不得为验收擅自录制真实桌面。

## 测试

```bash
cd backend
uv run --no-sync pytest -q -s
# 明确启用真实云端调用，会使用.env内现有模型与密钥。
RUN_CLOUD_TESTS=1 uv run --no-sync pytest tests/test_api.py::test_real_cloud_websocket -q -s --log-cli-level=INFO
```

视觉测试覆盖±15°旋转、光照、轻微模糊、题框投影<12px、空白识别；WS覆盖断线快照恢复、真实HWR/判题。
结构化日志包含capture/match/crop/hwr/first_verdict/all_verdicts/llm_ttft/tts_first_packet；
`page.grade_done.timings_ms`中的first_verdict/all_verdicts均为从抓拍开始累计耗时，不包含550ms稳帧窗口。
HWR为按题毫秒字典；all_explanations包含所有讲解完成时间。

目标机运行测试使用`--no-sync`或直接`backend/.venv/bin/python -m pytest`，避免自动同步时移除已安装的语音extra。
Qwen-ASR、本地GPU TTS及上述WS流程已在4090完成程序验收；Mac云端HWR性能不能代表4090。
真实麦克风/扬声器回声、浏览器AudioWorklet设备表现及SenseVoice回退仍须人工/对应环境验收。
