# AI学习台灯（AI学伴）系统架构与前后端接口契约

> 2026-11-17 更新：下文早期设计中的「本地秒判 + 模板解析兜底」已取消。当前所有题目走云端实时求解，不提供模板答案、解析或考点；新增 `uncertain`（待确认）状态，追问上下文限定当前页。最新提示词和数据边界见 [PROMPTS.md](PROMPTS.md)，具体实现以代码为准。

本文档为真实系统（非静态演示）的最终架构定型与前后端并行开发契约。视觉与交互严格对齐 `大屏学习看板/大屏学习看板.html` 原型，目标部署硬件为单机 Linux + 单卡 RTX 4090 (24GB)，支持 macOS 本地开发联调。

---

## 1. 草案评审与最终技术定型

### 1.1 进程拓扑与 vLLM 模型兼容性（已核实 vLLM >= 0.17.0 官方支持）
- **决策**：`1 个 FastAPI 主进程 (uvicorn, 单 worker 异步编排)` + `2 个独立 vLLM OpenAI-compatible 进程` + `云端 Qwen 27B API`。所有模型均通过 `.env` 的 `BASE_URL` / `MODEL` 解耦。
- **vLLM 兼容性核实与关键修正**：
  1. **`Qwen/Qwen3.5-2B`（本地手写识别 VLM）**：vLLM `>= 0.17.0` 原生支持（`Qwen3_5ForConditionalGeneration`，Gated DeltaNet 混合架构，默认非思考模式）。**关键坑点**：其 HuggingFace 默认上下文为 `262144` (262K)，直接启动会耗尽显存状态池；必须显式加 `--max-model-len 4096 --max-num-seqs 16 --mm-processor-kwargs '{"max_pixels":524288}'`，且请求侧固定传 `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`。
  2. **`PaddlePaddle/PaddleOCR-VL`（0.9B，离线建库 + 未知页兜底）**：vLLM 原生支持（`PaddleOCRVLForConditionalGeneration`），启动须加 `--trust-remote-code --no-enable-prefix-caching --mm-processor-cache-gb 0 --served-model-name PaddleOCR-VL-0.9B`。注意该模型是元素级识别 VLM，未知页兜底需在后端配合轻量版面检测 `PP-DocLayoutV2`（CPU/ONNX 运行，约 100MB，避免 Paddle 与 PyTorch 抢占 GPU 显存）切块后并发调 vLLM。
  3. **Fallback 方案**：若现场 CUDA/驱动与新版 vLLM 冲突，`Qwen3.5-2B` 与 `PaddleOCR-VL` 均支持 `SGLang` (`python -m sglang.launch_server`) OpenAI 兼容协议无缝替换；`PaddleOCR-VL` 亦可用官方镜像 `paddleocr genai_server --backend vllm`。

### 1.2 实时画面传输：MJPEG over HTTP（淘汰 WebRTC / WS-JPEG）
- **决策**：采集线程独占 USB 相机（`1920x1080 @ 30fps, FOURCC=MJPG`），仅保留最新帧；浏览器预览走 `GET /api/camera/stream`（`multipart/x-mixed-replace; boundary=frame`，缩放至 `1280x720 @ 20fps`，JPEG Q=75），前端直接挂载 `<img>`。
- **评审理由**：
  - **淘汰 WebRTC (`aiortc`)**：局域网/本机带宽充裕（720p20 约 1.8 MB/s），WebRTC 引入 ICE 协商、H.264 编码与抖动缓冲（反而增加 80-150ms），且跨 Mac/Linux 依赖脆弱。
  - **淘汰 WS-JPEG**：前端需在 JS 主线程做 Blob 解码绘制，与 KaTeX 公式流式渲染争抢主线程；若与音频共用 WS 还会造成 60KB 视频帧阻塞 20ms 语音包（Head-of-Line Blocking）。
  - **MJPEG 必做工程加固**：① Linux V4L2 必须设 `CAP_PROP_BUFFERSIZE=1` 并关闭连续自动对焦（防呼吸效应误触发运动检测），Mac `CAP_AVFOUNDATION` 用独立守护线程死循环 `read()` 抽干缓冲；② 前端切到批改定格图时必须置 `liveImg.src = ""` 释放连接，后端限制最多 2 路并发 MJPEG 流并在 `request.is_disconnected()` 时立即退出生成器，防止塞满浏览器同源连接池。

### 1.3 稳帧检测、页面匹配与手写裁剪（修正草案第 4、5 条）
- **稳帧检测状态机（`IDLE -> SETTLING -> CAPTURED -> LOCKED`）**：
  - 采集帧下采样至 `320x180` 灰度（15Hz 检测）：相邻帧平均绝对差 `motion_score < 2.5` 持续 **550ms**，且白纸连通域面积占比 `> 18%`、中心区域拉普拉斯方差（清晰度）达标时，从最近 3 帧中取最清晰的 `1920x1080` 原图定格。
  - **防抖重武装（Re-arm）**：进入 `LOCKED` 后忽略写字/手指微动；仅当大运动 `motion_score > 12.0` 持续 `>= 250ms`（翻页/换卷）或前端手动触发重拍时才重新武装。
- **页面匹配（CPU 实测预期 `15-22ms`，淘汰“先做标题行 OCR 初筛”）**：
  - **批评草案**：题库仅 4 份试卷（约 8-16 页模板），跑实时标题 OCR 初筛需 100ms+ 且怕纸张旋转遮挡，得不偿失。
  - **定型方案**：离线将 PDF 按 **200 DPI** 渲染为模板图（A4 为 `1654x2339 px`），预提取每页 **AKAZE 特征点（上限 2500 点，MLDB 二进制描述子）** 驻留内存。在线对 `1280px` 长边灰度图提 AKAZE（~8ms）-> `BFMatcher(NORM_HAMMING)` knnMatch 全库匹配（16 页共 ~4ms，Lowe ratio 0.78）-> Top-3 候选页跑 `cv2.findHomography(..., cv2.USAC_MAGSAC, 4.0)`（~4ms）。满足 `inliers >= 45` 且 `inlier_ratio >= 0.25` 且单应矩阵行列式合法即命中，直接用 `H` 将模板 bbox 投射回 `1920x1080` 原图像素空间。未命中则走 `PaddleOCR-VL` 未知页兜底。
- **手写识别裁剪（废弃“直接减背景模板抠墨迹”）**：
  - **批评草案**：纸张轻微拱起（3-5mm）会导致全局单应矩阵 `H` 存在 5-15px 局部偏差，叠加台灯照度渐变与阴影，`warped - clean_template` 会产生大量印刷体残影边缘，破坏笔迹。
  - **定型方案**：将捕获帧按 `H^-1` 透视校正回模板空间，对每个 `answer_area` 外扩 `+16px` 直接裁剪**自然灰度/彩色图像块**（保留括号、横线等印刷上下文，VLM 识别率远高于二值差分图）。空白检测仅对框内中心区做自适应阈值暗像素比对比模板基准（极空白 `< 0.3%` 直接判未作答），其余非空题块通过 `asyncio.gather` **单次并发批量发给本地 `Qwen3.5-2B`**（vLLM Continuous Batching 一次并行解码 3-8 张小图仅需 `160-220ms`，同时提示词要求过滤涂改划掉的废笔迹）。

### 1.4 判题快慢双通道与数学步骤诊断
- **快通道（本地秒判 + 云端并发）**：
  - **客观题（单选/英文填空）**：手写文本经大小写/标点归一化后与模板 `reference_answer` 精确匹配成功 -> **本地立即落判 `right`**（复用模板预置 `analysis`，无需等云端，首勾 `< 200ms`）；若不匹配且属明确选项/单词错误 -> 先推 `wrong` 落叉画圈，同时后台流式请求云端 **Qwen 27B（非思考模式）** 针对学生错答生成一两句纠错讲解；若手写置信度低或含歧义，交由 Qwen 27B 裁定。
  - **主观题（数学解方程/步骤题）与未作答题**：本地尝试用 `SymPy` 对相邻方程行做快速同解集校验（`< 15ms` 抓丢根/增根，如 `{0,3} -> {3}`），连同分步文本并发提交云端 Qwen 27B 生成结构化判定（含 `error_line_index`、`step_diagnosis`、`board_steps`）与流式讲解；空白题判为 `wrong` (`is_blank=true`) 并由 Qwen 27B 输出引导式解题提示。

### 1.5 语音对话链路（VAD / STT / TTS）与防回声自激打断
- **模型定型（中英混读友好、全本地驻留 4090、Mac 可跑）**：
  - **VAD**：**`Silero VAD v5`**（CPU ONNX Runtime，0 显存，单帧 32ms 耗时 `< 0.4ms`）。静音期起点阈值 `0.50`（持续 96ms），TTS 播报期起点阈值抬高至 `0.70`（持续 160ms，防扬声器残余回声误打断），句尾静音判定 **400ms**。
  - **STT**：**`FunASR SenseVoiceSmall` (`iic/SenseVoiceSmall`, 234M, FP16)**。显存仅 `0.6 GB`，4090 上 RTF `~0.006`（170x 实时，3-5 秒语音终审仅 **20-30ms**），中英混读（如九年级英语语法追问 `"为什么 if 从句不填 will rain"`）显著优于 `Paraformer-zh-streaming`（后者偏纯中文且每流占 2GB）。说话期间每 `400ms` 对当前语音缓冲跑一次滑动识别推送 `asr.partial` 实时字幕，`speech_end` 跑终审推送 `asr.final`。
  - **TTS**：首选 **`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`**（通过 **`faster-qwen3-tts`** 部署，BF16 + CUDA Graphs，显存 `~2.8 GB`，24kHz 输出）。实测 4090 上首包延迟（TTFA, `chunk_size=4`）仅 **130-156ms**，RTF `~0.18`（5.5x 实时），原生中英混读自然度远超 CosyVoice2，且在 Apple Silicon Mac 上自动切换 GGML Metal 后端运行。轻量备选（Fallback）：`Kokoro-82M-v1.1-zh`（显存 `0.5 GB`）或云端 OpenAI 兼容 `/v1/audio/speech`，通过 `TTS_PROVIDER` 切换。
- **数学符号播报清洗（TTS Normalizer）**：LLM 流式文本按标点切分（首分句满 6 字遇 `，。！？；：` 即送 TTS，后续满 15 字切句），送入 TTS 前经过确定性正则清洗（如 `x²` -> `x的平方`、`x₁` -> `x一`、`±` -> `正负`、`≠` -> `不等于`、过滤 `$` 与下划线 `____`），前端字幕仍展示原始公式文本。
- **Barge-in 打断闭环**：支持 `vad`（全双工打断）与 `ptt`（按住说话，演示现场强抗噪兜底）双模式。一旦触发打断，服务端递增 `turn_seq`、立即 `cancel()` 当前 LLM 与 TTS 协程，下发 `voice.interrupted`，前端清空 WebAudio 播放队列并丢弃旧 `turn_seq` 二进制音频帧。

### 1.6 前端技术栈（淘汰构建工具链，零构建原生模块化）
- **决策**：**Vanilla ES Modules (`<script type="module">`) + 本地化 `KaTeX` 静态资源**，由 FastAPI 直接托管静态目录，无 Node/Vite 构建步骤。
- **评审理由**：现有原型 `大屏学习看板.html` 已具备完整高水准三栏自适应布局、拖拽分栏、暗色模式、SVG 笔迹动画与缩放追光（Spotlight）逻辑。拆分为 7 个轻量 ES 模块并直接复用原型 CSS，开发最快且视觉 100% 还原。画面区改为双层对齐容器：底层为实时 `<img>` 或定格 `1920x1080` 抓拍图，顶层为同尺寸 `<svg viewBox="0 0 W H">` 像素级批改叠加层。

### 1.7 RTX 4090 (24GB = 24,564 MiB) 显存预算表

| 常驻进程 / 模块 | 模型与精度 | 关键显存控制参数 | 实占显存估算 |
| :--- | :--- | :--- | ---: |
| `vLLM #1` (`:8001`) | `Qwen/Qwen3.5-2B` (BF16) | `--gpu-memory-utilization 0.32 --max-model-len 4096` | ~7.8 GB |
| `vLLM #2` (`:8002`) | `PaddlePaddle/PaddleOCR-VL` (BF16) | `--gpu-memory-utilization 0.18 --max-model-len 8192` | ~4.4 GB |
| `FastAPI` 内嵌 STT | `FunASR SenseVoiceSmall` (FP16) | 单实例常驻，输入 `<= 30s` | ~0.6 GB |
| `FastAPI` 内嵌 TTS | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | `faster-qwen3-tts` BF16 + CUDA Graphs | ~2.8 GB |
| `FastAPI` CPU 组件 | `Silero VAD v5` + `PP-DocLayoutV2` | ONNX Runtime CPU (`intra_op_num_threads=4`) | 0.0 GB |
| 多进程 CUDA 上下文 | 3 个 CUDA 进程开销缓冲 | 按序启动：`vLLM #1 -> vLLM #2 -> FastAPI` | ~1.2 GB |
| **合计 / 安全余量** | **全部本地模型同时常驻** | **无需动态卸载换入** | **~16.8 GB（余 ~7.2 GB）** |

---

## 2. 端到端延迟预算表（RTX 4090 基准）

### 2.1 批改与首包播报链路（放稳纸张 -> 批改上屏 -> 开口总结/讲题）

| 阶段 | 耗时 (P50 / P90) | 累计时间（自定格触发 `t=0`） | 说明 |
| :--- | :--- | :--- | :--- |
| 前置：纸张放稳防抖窗口 | 550 ms | `-550 ms -> 0 ms` | 320x180 帧差 `< 2.5` + 清晰度门限达标 |
| 1. 原图抓取 + 内存快照编码 | 10 / 18 ms | 10 ms | 1920x1080 定格并推送 `capture.taken` |
| 2. AKAZE + USAC_MAGSAC 匹配投影 | 18 / 28 ms | **~28 ms（题号列表亮起）** | CPU 全库匹配 + `H` 投影题框，推 `page.matched` |
| 3. 答题区透视裁剪 + 空白检测 | 12 / 20 ms | 40 ms | 3-8 个答题框外扩 16px 并发编码 |
| 4. `Qwen3.5-2B` 手写批量识别 | 140 / 220 ms | 180 - 260 ms | 单选短框 ~120ms 先回，数学多行过程 ~220ms |
| 5. 本地规则比对 / SymPy 验算 | 2 / 15 ms | **~185 ms（首个客观题 ✓/✗ 上屏）** | 自物理停纸约 `735 ms` 亮起首个红勾/红叉 |
| 6. 云端 Qwen 27B 并发判题与解析 | 480 / 850 ms | **~680 - 950 ms（整页全部 ✓/✗ 完成）** | `enable_thinking=false`，主观题与错题并发流式返回 |
| 7. 首包语音出声（总结 / 首句讲解） | 170 / 600 ms | **~850 - 1350 ms（首包语音播放）** | 汇总播报无需等 LLM 约 `170ms`；LLM 首分句驱动约 `600ms` |

### 2.2 语音问答与打断链路（`speech-end` -> 首包 TTS 出声）

| 阶段 | 耗时 (P50 / P90) | 累计时间（自 `vad.speech_end` `t=0`） | 说明 |
| :--- | :--- | :--- | :--- |
| 前置：VAD 句尾静音确认窗 | 400 ms | `-400 ms -> 0 ms` | PTT 松开模式下此项为 `0 ms` |
| 1. `SenseVoiceSmall` 终审识别 | 25 / 40 ms | 25 ms | GPU FP16 整句识别，立即推 `asr.final` 上屏 |
| 2. 云端 Qwen 27B 首分句生成 | 420 / 650 ms | 445 ms | 非思考流式 TTFT ~310ms + 产出首短句（6-10 字）~110ms |
| 3. `faster-qwen3-tts` 首包合成 | 145 / 175 ms | 590 ms | 符号清洗 + CUDA Graph `chunk_size=4` 首包 24kHz PCM |
| 4. WS 下行 + WebAudio 缓冲调度 | 25 / 35 ms | **~615 ms（P90 ~900 ms）** | 含物理句尾静音窗合计约 `1.0 s` |
| **Barge-in 打断响应** | **175 / 220 ms** | 开口说话 -> 喇叭完全静音 | 160ms VAD 防误触确认 + 5ms 取消协程 + 10ms 清空缓冲 |

---

## 3. 模板 JSON 契约（`data/templates/{paper_id}/template.json`）

**坐标系铁律**：模板中所有 `bbox` / `mark_anchor` 均采用 `render_dpi: 200` 渲染出的模板页 PNG **像素坐标** `[x, y, w, h]`（左上角原点）。匹配成功后，直接用 `cv2.perspectiveTransform(pts, H)` 一次性映射至抓拍原图像素空间（`image_width x image_height`），严禁混用归一化比例或 PDF pt 单位。

```json
{
  "paper_id": "en_g9_unit1",
  "title": "九年级英语 · 课后练习",
  "subject": "en",
  "grade": "9",
  "source_pdf": "en_g9_unit1.pdf",
  "render_dpi": 200,
  "pages": [
    {
      "page_no": 1,
      "image_file": "page_01.png",
      "features_file": "page_01_akaze.npz",
      "width_px": 1654,
      "height_px": 2339,
      "questions": [
        {
          "id": "en_u1_p1_q3",
          "number": "3",
          "type": "choice",
          "stem": "If it ______ tomorrow, we will stay at home.",
          "options": ["A. rains", "B. will rain", "C. rained", "D. is raining"],
          "reference_answer": "A",
          "answer_display": "A. rains",
          "tag": "条件状语从句“主将从现”",
          "analysis": "if引导的条件状语从句，主句用将来时，从句要用一般现在时，所以选A。",
          "bbox": [140, 610, 1360, 145],
          "mark_anchor": [1430, 625],
          "answer_areas": [
            {
              "id": "a1",
              "bbox": [152, 612, 92, 58],
              "hint": "choice_letter"
            }
          ]
        }
      ]
    }
  ]
}
```
- **字段约束**：`subject` 取 `"en" | "math"`；`type` 取 `"choice" | "fill" | "math"`；`options` 非选择题为 `null`；`reference_answer` 支持字符串或等价答案数组（如 `["x1=0,x2=3", "x=0或x=3"]`）；`hint` 取 `"choice_letter" | "english_text" | "math_steps"`，决定发给 `Qwen3.5-2B` 的专用短提示词模板。

---

## 4. HTTP API 契约（默认同源 `http://localhost:8000`）

| 方法与路径 | 用途 | 请求参数 / Body | 响应格式与状态码 |
| :--- | :--- | :--- | :--- |
| `GET /` 及静态资源 | 前端单页应用托管 | 无 | FastAPI 在注册完 `/api/*` 与 `/ws` 后挂载 `frontend/` 为根静态目录（含 `html=True`） |
| `GET /api/health` | 健康与组件探活 | 无 | `200 JSON`: `{"ok":true, "platform":"linux\|darwin", "camera":{"ok":true,"backend":"v4l2","res":[1920,1080]}, "models":{"hwr_vllm":true,"ocr_vllm":true,"cloud_llm":true,"stt":true,"tts":true}, "templates_count":4}` |
| `GET /api/camera/stream` | 实时预览视频流 | `?fps=20&width=1280` (可选) | `200 multipart/x-mixed-replace; boundary=frame`，每帧 `Content-Type: image/jpeg` |
| `POST /api/camera/snapshot` | 手动抓拍并可选触发批改 | `{"grade": true}` | `200 JSON`: `{"capture_id":"cap_1731758001", "image_url":"/api/captures/cap_1731758001.jpg", "width":1920, "height":1080}` |
| `GET /api/captures/{capture_id}.jpg` | 获取定格原图 | 路径参数 `capture_id` | `200 image/jpeg`（内存 LRU 保留最近 20 张，加 `Cache-Control: immutable`） |
| `GET /api/templates` | 已加载试卷列表 | 无 | `200 JSON`: `{"papers":[{"paper_id":"...","title":"...","subject":"en","page_count":2,"question_count":16}]}` |
| `GET /api/templates/{paper_id}` | 单份试卷完整模板 | 路径参数 `paper_id` | `200 JSON`（同第 3 节 Schema），不存在返 `404` |
| `GET /api/session/state` | 页面刷新恢复全量状态 | 无 | `200 JSON`（结构与 WS `session.snapshot` 完全一致） |
| `POST /api/debug/inject-image` | Mac 开发/无相机注入测试图 | `multipart/form-data`: `file` (JPG/PNG), `grade=true` | `200 JSON`（同 `snapshot`，直接走匹配与批改全流程） |

---

## 5. WebSocket 实时协议契约（`ws(s)://<host>/ws`）

采用**单条多路复用 WebSocket 连接**承载全部 JSON 控制/流式事件与二进制 PCM 语音流，保证消息与打断时序严格一致，支持多端（大屏 + 家长手机端）同时在线同步状态。

### 5.1 二进制音频帧格式（4 字节对齐头，支持 JS `new Int16Array(buf, 4)` 零拷贝）
所有二进制 WS 消息均采用 **Little-Endian 4 字节头 (`struct.pack('<BBH', frame_type, flags, turn_seq)`) + Raw PCM16LE 单声道负载**：
- `Byte 0` (`uint8` `frame_type`)：`0x01` = 上行麦克风音频（Client -> Server）；`0x02` = 下行 TTS 音频（Server -> Client）。
- `Byte 1` (`uint8` `flags`)：`0x00` = 常规音频分片；`0x01` = 当前语句最后一个音频分片。
- `Bytes 2..3` (`uint16` `turn_seq`)：对话轮次序号（`0..65535` 循环递增）。前端仅播放 `turn_seq === activeTurnSeq` 的 `0x02` 帧，收到打断或新 `turn_seq` 立即丢弃旧帧。
- `Bytes 4..N` (`int16[]` PCM payload)：
  - **上行 (`0x01`)**：采样率 **`16000 Hz`**，单声道，`Int16LE`，AudioWorklet 每 **20ms** 发送一帧（320 samples = `640 bytes`，连头共 `644 bytes`）。浏览器采集须开启 `{echoCancellation:true, noiseSuppression:true, autoGainControl:true}`。
  - **下行 (`0x02`)**：采样率 **`24000 Hz`**，单声道，`Int16LE`，每块约 80-320ms（1920-7680 samples）。

### 5.2 Client -> Server JSON 事件（均含 `"type"` 字段）
1. **模式设置**：`{"type": "session.set_mode", "auto_capture": true, "auto_grade": true, "voice_mode": "vad"}`（`voice_mode`: `"vad" | "ptt" | "off"`）。
2. **手动控制**：`{"type": "camera.capture", "grade": true}` | `{"type": "session.grade_current"}` | `{"type": "session.reset"}`。
3. **焦点切换（同步语音追问上下文与多端选中态）**：`{"type": "question.select", "question_uid": "en_u1_p1_q3"}`（传 `null` 表示回到整晚全局动态；选中历史页题目时前端自动切换显示该题所属 `capture_id` 定格图）。
4. **文字提问**：`{"type": "chat.send", "client_msg_id": "c_1", "scope": "question", "question_uid": "en_u1_p1_q3", "text": "我还是不懂为什么选A", "speak": true}`（全局提问时 `scope: "all", question_uid: null`）。
5. **主动停止生成/播报**：`{"type": "chat.stop", "turn_seq": 12}`。
6. **按住说话状态**：`{"type": "voice.ptt", "active": true}`（`active: true` 时若服务端正在 TTS，立即视作主动打断）。
7. **前端播放回执（辅助服务端动态抬高 VAD 阈值）**：`{"type": "voice.playback_state", "playing": true, "turn_seq": 12}`。

### 5.3 Server -> Client JSON 事件（所有图像坐标均为 `capture` 原图像素坐标 `0..image_width, 0..image_height`）

#### A. 会话同步与相机/抓拍状态
- **`session.snapshot`**（WS 建连首包及 `GET /api/session/state` 返回结构）：
  ```json
  {
    "type": "session.snapshot",
    "mode": {"auto_capture": true, "auto_grade": true, "voice_mode": "vad"},
    "turn_seq": 1,
    "active_capture_id": "cap_01",
    "selected_question_uid": null,
    "captures": {
      "cap_01": {
        "capture_id": "cap_01", "image_url": "/api/captures/cap_01.jpg",
        "image_width": 1920, "image_height": 1080, "paper_id": "en_g9_unit1",
        "paper_title": "九年级英语 · 课后练习", "subject": "en", "page_no": 1,
        "page_quad": [[412,88],[1510,74],[1532,1012],[398,1026]],
        "question_uids": ["en_u1_p1_q3"]
      }
    },
    "questions": {"en_u1_p1_q3": {"...": "合并 page.matched 题项 + ocr + verdict + explain_text + board_steps 字段"}},
    "threads": {"all": [], "en_u1_p1_q3": [{"msg_id":"m_1","role":"user","text":"...","interrupted":false,"is_barge_in":false}]},
    "events": [{"id":"ev_1","time":"21:03","kind":"detect","text":"检测到作业纸 · 开始识别"}],
    "study_started_at": 1763296260
  }
  ```
- **`camera.state`**（状态变化或稳帧进度更新时推送，最高 5Hz）：
  `{"type":"camera.state", "status":"streaming", "stability":"settling", "motion_score":1.8, "progress":0.65}`（`status`: `"streaming" | "no_camera" | "mock"`；`stability`: `"idle" | "settling" | "captured" | "locked"`）
- **`capture.taken`**：
  `{"type":"capture.taken", "capture_id":"cap_01", "image_url":"/api/captures/cap_01.jpg", "image_width":1920, "image_height":1080, "ts":"21:03:22"}`

#### B. 页面匹配与逐题流式批改
- **`page.matched`**（~30ms 内返回，驱动左栏题目列表与图上题框；未知页兜底时 `mode: "ocr_fallback"`, `paper_id: "unknown_{capture_id}"`）：
  ```json
  {
    "type": "page.matched",
    "capture_id": "cap_01",
    "mode": "template",
    "paper_id": "en_g9_unit1",
    "paper_title": "九年级英语 · 课后练习",
    "subject": "en",
    "page_no": 1,
    "image_width": 1920,
    "image_height": 1080,
    "homography": [[0.66, -0.01, 319.0], [0.01, 0.40, 32.0], [0.0, 0.0, 1.0]],
    "page_quad": [[412,88],[1510,74],[1532,1012],[398,1026]],
    "questions": [
      {
        "question_uid": "en_u1_p1_q3",
        "capture_id": "cap_01",
        "no": "3",
        "qtype": "choice",
        "short_stem": "If it ___ tomorrow, we will stay at home.",
        "stem": "If it ______ tomorrow, we will stay at home.",
        "options": ["A. rains", "B. will rain", "C. rained", "D. is raining"],
        "tag": "条件状语从句“主将从现”",
        "status": "judging",
        "quad": [[468,462],[1445,452],[1447,558],[470,568]],
        "bbox": [468, 452, 979, 116],
        "mark_pos": [1396, 464],
        "answer_quads": [[[476,464],[542,463],[543,505],[477,506]]]
      }
    ]
  }
  ```
  - 题目 `status` 枚举：`"idle"`（待批改，`auto_grade=false` 时）| `"judging"`（判定中）| `"right"`（正确）| `"wrong"`（错误）。`quad` 为顺时针四顶点像素坐标 `[TL, TR, BR, BL]`，用于旋转纸面聚光灯遮罩（Spotlight）与点击命中；`bbox` 为外接轴对齐矩形 `[x, y, w, h]`，用于镜头平移缩放定帧。
- **`page.unmatched`**：`{"type":"page.unmatched", "capture_id":"cap_01", "reason":"no_page | blurry | unknown_failed"}`
- **`question.ocr`**（某题手写识别完成，填右栏“识别原文”）：
  `{"type":"question.ocr", "capture_id":"cap_01", "question_uid":"math_p1_q2", "is_blank":false, "student_raw":"x^2=3x\n两边同时除以x\nx=3", "student_lines":[{"text":"x^2 = 3x","latex":"x^2=3x","low_confidence":false,"quad":null}, {"text":"两边同时除以 x","latex":null,"low_confidence":false,"quad":[[520,610],[790,606],[791,652],[521,656]]}, {"text":"x = 3","latex":"x=3","low_confidence":true,"quad":null}]}`
- **`question.verdict`**（单题出对错，立即在原图 `mark_pos` 画 ✓/✗，错题在 `error_quad` 画红圈）：
  ```json
  {
    "type": "question.verdict",
    "capture_id": "cap_01",
    "question_uid": "math_p1_q2",
    "status": "wrong",
    "is_blank": false,
    "source": "cloud_llm",
    "student_display": "x = 3",
    "reference_display": "x_1 = 0, x_2 = 3",
    "tag": "一元二次方程的解法（因式分解法）",
    "mark_pos": [1380, 560],
    "error_quad": [[520,610],[790,606],[791,652],[521,656]],
    "error_note": "x = 0 呢?",
    "error_step_desc": "两边同时除以 x，丢了 x=0 这个根",
    "step_diagnosis": {
      "orig_expr": "x^2 = 3x",
      "orig_roots": ["0", "3"],
      "operation": "两边同时除以 x",
      "next_expr": "x = 3",
      "next_roots": ["3"],
      "lost_roots": ["0"],
      "summary": "解集变小 -> 丢根"
    }
  }
  ```
  - 注：无步骤诊断的普通题目 `error_note`、`error_step_desc`、`step_diagnosis` 均为 `null`；客观错题 `error_quad` 默认取其 `answer_quads[0]`。
- **`question.explain_delta`**：`{"type":"question.explain_delta", "capture_id":"cap_01", "question_uid":"en_u1_p1_q3", "delta":"if引导的条件状语从句，"}`
- **`question.explain_done`**：`{"type":"question.explain_done", "capture_id":"cap_01", "question_uid":"math_p1_q2", "full_text":"...", "board_steps":[{"latex":"x^2-3x=0","label":"移项"},{"latex":"x(x-3)=0","label":"提公因式"}]}`
- **`page.grade_done`**：`{"type":"page.grade_done", "capture_id":"cap_01", "paper_id":"en_g9_unit1", "page_no":1, "total":8, "right_count":6, "wrong_uids":["en_u1_p1_q3","en_u1_p1_q6"], "summary_text":"8道题对了6道。第3题和第6题错了，我已经圈出来了。"}`
- **`timeline.event`**：`{"type":"timeline.event", "id":"ev_9", "time":"21:03", "kind":"ok", "text":"英语批改完成 · 6/8 正确 · 2题收进错题本"}`（`kind`: `"detect" | "grade" | "ok" | "talk" | "alert"`）

#### C. 语音识别、流式对话与 TTS 打断
- **`vad.state`**：`{"type":"vad.state", "speaking":true}`（驱动底部声波条 `"小宇正在说…"`）
- **`asr.partial`**：`{"type":"asr.partial", "utterance_id":"u_5", "text":"那是不是方程两边"}`
- **`asr.final`**：`{"type":"asr.final", "utterance_id":"u_5", "msg_id":"m_10", "scope":"question", "question_uid":"math_p1_q2", "text":"那是不是方程两边永远都不能同时除以x？", "is_barge_in":false}`（语音用户气泡落库并广播；若 `is_barge_in:true` 前端显示 `打断` 标签）
- **`chat.user_message`**：`{"type":"chat.user_message", "msg_id":"m_10", "client_msg_id":"c_1", "scope":"question", "question_uid":"math_p1_q2", "text":"我还是不懂为什么选A"}`（文字提问 `chat.send` 落库广播，前端按 `client_msg_id` 去重）
- **`chat.message_start`**：`{"type":"chat.message_start", "msg_id":"m_11", "turn_seq":12, "scope":"question", "question_uid":"math_p1_q2", "role":"assistant"}`
- **`chat.delta`**：`{"type":"chat.delta", "msg_id":"m_11", "turn_seq":12, "delta":"不是。关键看除的那个式子"}`
- **`chat.message_done`**：`{"type":"chat.message_done", "msg_id":"m_11", "turn_seq":12, "full_text":"...", "interrupted":false, "board_steps":null}`（若 `interrupted:true`，前端在该气泡尾追加 `被打断` 标签）
- **`tts.start`**：`{"type":"tts.start", "turn_seq":12, "msg_id":"m_11", "sample_rate":24000, "format":"pcm_s16le"}`
- **`tts.end`**：`{"type":"tts.end", "turn_seq":12, "interrupted":false}`
- **`voice.interrupted`**：`{"type":"voice.interrupted", "old_turn_seq":12, "new_turn_seq":13, "reason":"barge_in"}`（前端收到后必须在 10ms 内 `AudioWorklet` 清空环形队列并切换 `activeTurnSeq = 13`）
- **`error`**：`{"type":"error", "code":"CLOUD_LLM_TIMEOUT", "message":"云端判题响应超时，请重试", "recoverable":true}`

---

## 6. 仓库目录结构与部署运行指南

### 6.1 目录结构
```text
/Users/dahetao/Desktop/AI智能学习平台/
├── docs/ARCHITECTURE.md            # 本文档（唯一接口契约真源）
├── .env.example                    # 环境变量模板（Linux 4090 / Mac Dev 双配置注释）
├── backend/
│   ├── pyproject.toml              # Python >= 3.11 依赖定义 (uv 管理)
│   └── app/
│       ├── main.py                 # FastAPI 入口、静态资源挂载、生命周期管理
│       ├── config.py               # Pydantic Settings 读取 .env
│       ├── state.py                # 内存会话状态机与 WS 广播 Hub
│       ├── api/                    # routes.py (HTTP) + ws.py (WebSocket 多路复用)
│       ├── vision/                 # camera.py (V4L2/AVFoundation), stability.py, matcher.py (AKAZE+H), cropper.py
│       ├── grading/                # hwr_client.py (Qwen3.5-2B), ocr_fallback.py, judge.py (规则+SymPy+Qwen27B)
│       └── voice/                  # vad.py (Silero), stt.py (SenseVoice), tts.py (FasterQwen3TTS), normalizer.py
├── frontend/
│   ├── index.html                  # 主页面（三栏看板 + 窄屏自适应）
│   ├── css/styles.css              # 承袭原型视觉变量、暗色模式与 SVG 动画样式
│   ├── js/                         # app.js, store.js, ws.js, stage.js (画面+SVG叠加), sidebar.js, panel.js, audio.js, audio-worklet.js
│   └── vendor/katex/               # 本地化 KaTeX 0.16.x JS/CSS/fonts（断网可用）
├── data/
│   ├── papers/pdf/                 # 4 份原始九年级数学/英语 A4 PDF 试卷
│   ├── mock_captures/              # Mac 联调用实拍/合成带手写测试图
│   └── templates/                  # 离线生成的试卷模板包 {paper_id}/{template.json, page_01.png, page_01_akaze.npz}
├── tools/
│   ├── build_template.py           # 离线建库脚本：PDF 200DPI 渲染 + PaddleOCR-VL 切题提取 + Qwen27B 预解题 + AKAZE 特征预计算
│   └── preview_template.py         # 生成带 bbox/answer_area 可视化框的校验图供人工核对
└── deploy/
    ├── start_4090.sh               # Linux 4090 按序启动 vLLM x2 + FastAPI 脚本
    └── gen_lan_cert.sh             # 局域网非 localhost 访问时用 mkcert 生成 HTTPS 证书（解锁浏览器麦克风）
```

### 6.2 `.env` 核心配置项
```ini
CAMERA_INDEX=0                      # Linux: /dev/video0 (0), Mac: 0 或 1; 设 -1 启用纯图片注入模式
CAMERA_BACKEND=auto                 # auto | v4l2 | avfoundation | mock
HWR_BASE_URL=http://127.0.0.1:8001/v1
HWR_MODEL=Qwen3.5-2B
OCR_BASE_URL=http://127.0.0.1:8002/v1
OCR_MODEL=PaddleOCR-VL-0.9B
CLOUD_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CLOUD_LLM_API_KEY=sk-xxx
CLOUD_LLM_MODEL=qwen3.5-27b         # 可配置为任意 OpenAI 兼容端点与模型名
STT_DEVICE=cuda                     # Linux: cuda, Mac: cpu (SenseVoiceSmall 在 M 系列 CPU 达 17x 实时)
TTS_PROVIDER=qwen3_tts              # qwen3_tts | kokoro | openai_remote | mock
TTS_VOICE=Chelsie                   # Qwen3-TTS CustomVoice 预置音色
```

### 6.3 启动方式
1. **Linux + RTX 4090 生产环境（须按顺序启动以避免 vLLM 显存探测竞争）**：
   ```bash
   # 1) 启动手写识别 Qwen3.5-2B (端口 8001, 约 7.8GB VRAM)
   vllm serve Qwen/Qwen3.5-2B --served-model-name Qwen3.5-2B --port 8001 \
     --trust-remote-code --max-model-len 4096 --max-num-seqs 16 \
     --mm-processor-kwargs '{"max_pixels":524288}' --gpu-memory-utilization 0.32

   # 2) 待 8001 就绪后，启动印刷体兜底 PaddleOCR-VL (端口 8002, 约 4.4GB VRAM)
   vllm serve PaddlePaddle/PaddleOCR-VL --served-model-name PaddleOCR-VL-0.9B --port 8002 \
     --trust-remote-code --max-model-len 8192 --max-num-batched-tokens 16384 \
     --no-enable-prefix-caching --mm-processor-cache-gb 0 --gpu-memory-utilization 0.18

   # 3) 启动 FastAPI 主进程 (端口 8000, 内载 SenseVoice + FasterQwen3TTS + SileroVAD)
   cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   - 注：`vllm` 与 `paddleocr[doc-parser]` 须分置独立虚拟环境，避免 `transformers` / `paddlepaddle` 依赖冲突。
2. **macOS 本地开发环境**：
   - 相机：`CAMERA_BACKEND=avfoundation` 直接调用 Mac 摄像头或外接 USB 俯拍相机；无打印卷时调 `POST /api/debug/inject-image` 注入测试照片。
   - 模型：可通过 `ssh -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 user@4090-box` 隧道直连远端 vLLM（或指向云端多模态兼容接口 / 本地 Mock）；`Silero VAD`、`SenseVoiceSmall` (`STT_DEVICE=cpu`) 与 `faster-qwen3-tts` (自动启用 GGML Metal) 均可在 Apple Silicon 原生离线运行。

---

## 7. 开放风险与缓解措施

1. **纸张卷曲与台灯局部反光/手影**：单应矩阵 `H` 仅精确拟合平面；对策是在模板 `answer_areas` 裁剪时四周固定外扩 `+16px`，桌面垫深色哑光桌垫（增强纸张四角对比度），相机固定曝光避免过曝吃掉浅色铅笔字。
2. **学生涂改（划掉重写）干扰手写 VLM**：`Qwen3.5-2B` 提示词明确约束“忽略被横线/斜线划掉或涂黑的内容，仅输出最终保留的手写答案”，客观题若返回多词异常串则自动升级交由云端 Qwen 27B 结合上下文消歧。
3. **外放扬声器回声导致 VAD 自激假打断**：除开启浏览器 `echoCancellation` 外，建议物理使用带硬件 DSP 消回声的全向会议麦；服务端在 `tts.start` 至 `tts.end + 250ms` 窗口内自动将 VAD 触发门限上调至 `0.70`（持续 160ms），并在 UI 保留“按住说话（PTT）”一键切换。
4. **局域网跨设备打开无法调起麦克风**：浏览器仅在 `localhost` 或 `HTTPS` 下暴露 `navigator.mediaDevices.getUserMedia`；若大屏浏览器与 4090 主机分属两台机器，必须用 `deploy/gen_lan_cert.sh` (`mkcert`) 为 `uvicorn` 配置 `--ssl-keyfile` / `--ssl-certfile`。
5. **云端 Qwen 27B 网络抖动或限流**：所有云端调用设单题首包超时 `4.0s`、总超时 `12.0s` 与最大并发信号量 `8`；客观题因有本地规则秒判与模板预置 `analysis` 兜底，即使外网波动仍可保证 `✓/✗` 与基础解析毫秒级上屏。

---

## 8. 主控修订（v1 定稿，优先级高于上文）

以下修订以“先跑通真实链路、不过度工程化”为原则，与上文冲突时以本节为准。

1. **云端模型**：`CLOUD_LLM_BASE_URL=https://api.dahetao.org/v1`，`CLOUD_LLM_MODEL=qwen3.8-27b`（OpenAI 兼容，流式）。Key 只放 `.env`，不得写进代码、日志或前端。
2. **手写识别开发态**：Mac 上没有本地 Qwen3.5-2B 时，`HWR_BASE_URL/HWR_MODEL` 可直接指向同一云端 API 的 `qwen3-vl-plus`。代码只认 OpenAI 兼容的 chat/completions + image_url(base64)，不区分本地/云端。
3. **新增题目状态 `blank`**：`status` 枚举改为 `idle | judging | right | wrong | blank`。未作答的题不算错：图上画灰色“未作答”标记，右栏直接给出参考答案与解析（即“做题”），`page.grade_done` 增加 `blank_count`。
4. **v1 砍掉**：SymPy 步骤验算（`step_diagnosis` 字段保留，由 Qwen 的 JSON 输出可选填充，否则为 null）；在线 PP-DocLayoutV2 未知页兜底（未匹配页只推 `page.unmatched`）；多端同步之外的任何持久化（会话只在内存）。PaddleOCR-VL 在 v1 只用于 `tools/` 离线建模板。
5. **判题流程（每题一次云端调用，并发上限 8）**：
   - 客观题（`choice`/`fill`）：手写识别结果归一化后与 `reference_answer` 精确匹配 → 本地立即推 `question.verdict`（`source:"local"`）；对的题直接用模板 `analysis`，不调云端。
   - 不匹配 / 主观题 / 空白题 → 调 `qwen3.8-27b`（`enable_thinking:false`），提示词含题干、选项、参考答案、模板解析、学生作答；要求**第一行输出紧凑 JSON**（`{"status":"right|wrong|blank","student_display":"...","error_note":"...","error_line":null|int}`），之后流式输出讲解正文。后端解析出首行 JSON 即推 `question.verdict`，正文走 `question.explain_delta`。
6. **答题区约定**：英语卷多为“在此卷上答题无效”的版式，选择题没有括号。模板里选择题的 `answer_areas` 统一为“题号左侧页边 + 题干横线”区域，交给 HWR 时提示“只读出手写内容，忽略印刷体；没有手写输出空字符串”。空白预判仍用暗像素对比模板同区域（差值阈值），差值很小直接判 `blank` 不调 HWR。
7. **模板文件**：`data/templates/{paper_id}/template.json` + `page_NN.png`（200 DPI 灰度）+ `page_NN_akaze.npz`（由 `tools/build_features.py` 生成；缺失时后端启动自动生成）。听力题 `type:"listening"`，不判分、不画标记。
8. **测试素材**：`tools/synth_capture.py` 把模板页透视贴到书桌背景上（随机旋转 ±15°、光照渐变、轻微模糊），并在答题区用手写风格字体写入答案（可指定对/错/空），输出 1920x1080 JPG 到 `data/mock_captures/`。Mac 无相机时用 `POST /api/debug/inject-image` 或 `CAMERA_BACKEND=mock`（循环播放 mock 图片，模拟“放下-稳定-拿走”）跑全流程。
9. **TTS/STT 开发态**：Mac 上 STT 用 SenseVoiceSmall CPU；TTS 提供 `qwen3_tts`（4090 首选）与 `kokoro`（Mac/轻量）两个实现，再加 `mock`（静音）。所有本地模型都懒加载，缺依赖时 `/api/health` 标红但主流程（批改）照常可用。
10. **前端 mock 模式**：`/?mock=1` 时前端不连 WS，按脚本回放一组契约事件（使用 `data/templates/.../page_01.png` 作为抓拍图），用于独立开发与演示兜底。

## 9. 语音方案定稿（2026-11 调研，取代 1.5 节的模型选型）

### 9.1 STT：Qwen3-ASR-1.7B（替换 SenseVoiceSmall）
- **依据**：Qwen3-ASR（2026-01，Apache 2.0，52 种语言/方言，中英混说）在多份独立评测中是开源准确率第一梯队：普通话实测字错率 2.4%（SenseVoice-Small 4.4%、Paraformer 4.4%、Whisper-large-v3 7.1%）。0.6B 版首 token 延迟 92ms，1.7B 只略慢。
- **上下文偏置（关键提效）**：Qwen3-ASR 支持把任意文本作为 context（≤10k token），实测热词召回率从 31% 提到 72%。我们每轮都把**当前选中题的题干、选项、知识点标签**作为 context 传入，“主将从现”“一元二次方程”“whether”这类学科词识别会明显更准。
- **部署**：4090 上用 vLLM 起 `Qwen/Qwen3-ASR-1.7B`；当前启动脚本使用`--gpu-memory-utilization 0.35 --max-model-len 2048 --kv-cache-memory-bytes 1073741824`，实测进程约6GiB。说话过程中每 600ms 对累计音频调一次出 `asr.partial`，`speech_end` 后终审出 `asr.final`。Mac 开发态退回 `SenseVoiceSmall` CPU（`STT_PROVIDER=sensevoice`）。
- **备选**：`Qwen3-ASR-0.6B`（更快）；`FireRedASR2-AED`（1B，普通话 benchmark 更强，但不支持上下文偏置）。

### 9.2 语气词过滤与防误打断
1. **确定性清洗（0ms）**：`asr.final` 文本先过 `voice/normalizer.py`：去掉独立的“嗯/啊/呃/额/哦/唔/那个/就是说/然后呢”、连续重复词（“我我我”→“我”、“这个这个”→“这个”）、首尾口头语；英文去掉 um/uh/er/you know。字幕显示清洗后的文本。
2. **LLM 兜底（零额外调用）**：对话系统提示词里写明“用户输入来自语音转写，可能含口头语、重复或同音错字，按意图理解，不要指出转写错误”。不另加一次清洗模型调用，以免增加延迟。
3. **附和语不打断**：TTS 播放中 VAD 检测到说话时，先把播放**音量压到 30%（ducking）**而不是立刻停。等 partial 里出现 ≥2 个非语气词字符，才真正打断（`voice.interrupted`）。如果这段话清洗后为空或只是附和（“嗯嗯”“对”“好的”“哦”），就恢复音量继续播，也不发起新一轮对话。新增事件：`{"type":"voice.duck","on":true|false}`，前端据此调节播放增益。

### 9.3 TTS：Qwen3-TTS-12Hz-1.7B-CustomVoice（由 0.6B 升级）
- **依据**：自建 4090 单卡可跑的开源模型里，它的自然度与稳定性综合最好。技术报告里 1.7B 首包 101ms（0.6B 97ms），中英长文 WER 最低，Apache 2.0。主观盲测排名更高的 Fish Audio Drama 3 是 35B MoE，需要 48GB 以上显存，4090 放不下；Ming-3.0-tts 在 4090 上慢于实时，不适合对话；Breeze TTS 2 是非商用许可。
- **语气控制**：CustomVoice 支持自然语言指令。固定用一个中文女声（如 `Vivian`/`Serena`，以模型卡为准），按场景切换 instruct：讲题时“像耐心温和的初中老师，语速稍慢，重点词稍加重”；表扬时“语气轻快、真诚地表扬”；纠错时“温和、不责备”。由后端根据回复类型自动选择。
- **部署**：优先进程内 `faster-qwen3-tts`（CUDA Graph），不可用时用 vLLM-Omni 的 `/v1/audio/speech` 流式 PCM。显存约 5GB。备选：`Fun-CosyVoice3-0.5B-2512`（双向流式 150ms，支持情绪指令）。Mac 开发态用 `kokoro` 或 `mock`。
- 送 TTS 前的数学符号清洗沿用 1.5 节的 normalizer。

### 9.4 VAD
保留 Silero VAD v5（CPU，0 显存，成熟）。FireRedVAD 公布的 F1 更高，作为后续可选替换，v1 不引入。

4090验收修订：句尾静音由`VAD_END_SILENCE_MS`控制，默认**700ms**（原400ms容易截断学生边想边说的提问，允许200–3000ms）。同一连接、同一题目在上一段`asr.final`后**1.2秒内**续说，且上一轮**尚未发送首个`chat.delta`**时，撤销未出字的LLM请求并合并前后两段；通过已有`session.snapshot`移除临时气泡，最终只保留一轮，不发送`voice.interrupted`或误标“打断”。窗口从final开始，不额外固定等待1.2秒；已经出字、正在播音、不同连接或换题时仍按正常barge-in处理。

### 9.5 4090 显存重算（PaddleOCR-VL 不再常驻）
PaddleOCR-VL 只在离线建模板时启动（`deploy/start_ocr.sh`），线上不常驻。当前脚本固定HWR KV缓存1.5GiB、ASR KV缓存1GiB，二者的`gpu-memory-utilization`均为0.35。4090 D实测：HWR进程约6520MiB、ASR进程约6084MiB、含TTS的后端进程约5204MiB；全卡合计约18133 / 24564MiB（约17.7 / 24GiB），包括图形/驱动等占用，仍余约6.3GiB。

## 10. 模板 schema 补充（v1 定稿）
- 试卷级新增 `"passages": {"P1": "完整文章文本……"}`；题目级新增可选 `"passage_id": "P1"`。阅读、完形等共用材料的题通过它引用文章（文章和题目可以跨页）。判题、讲解、对话时，后端必须把对应 passage 一起放进提示词。
- 题目 `type` 枚举：`choice | fill | math | text | listening | writing`。`listening` 不判分、不画标记；`writing` 交给云端按要点点评，不判对错，状态记为 `right` 或 `wrong` 之外的 `blank`/点评文本。
- 题目可选字段：`answer_display`（给学生看的答案文本）、`tag`（知识点）、`analysis`（预置解析，1–3 句）、`ocr_text`（原始 OCR，调试用）。
