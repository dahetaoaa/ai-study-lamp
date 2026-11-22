# AI Study Lamp (AI 学习台灯)

面向桌面学习场景的实时视觉批改与全双工语音辅导系统。通过台灯摄像头采集纸面作答画面，完成稳帧抓拍、试卷模板匹配或无模板自由版面切题、手写体识别、云端实时判题与分步讲解，并在前端三栏交互看板中实时叠加批改标注与语音对话。

## 核心功能

- **实时视觉采集与稳帧抓拍**：支持 USB 相机（Linux V4L2 / macOS AVFoundation）与 Mock 调试模式，通过 MJPEG 流预览，基于运动检测、白纸占比与清晰度评估自动定格抓拍。
- **双模式题目定位**：
  - **模板匹配模式**：基于 AKAZE 特征点与 MAGSAC 单应矩阵将试卷模板题框映射至实时画面，支持透视矫正与空白作答门控。
  - **无模板自由版面模式**：针对未建库页面，结合 `PaddleOCR-VL` 整页识别、版面切分与墨迹门控实现通用题目定位。
- **手写识别与云端判题**：使用 `Qwen3.5-2B`（或云端多模态模型）并发识别手写作答区域，由云端大模型实时求解、判断正误/待确认状态并输出结构化分步解析与板书步骤。
- **全双工语音对话**：集成 `Silero VAD`、`FunASR SenseVoiceSmall` / `Qwen3-ASR` 与 `Qwen3-TTS` / `Kokoro`，支持语音打断（Barge-in）、按住说话（PTT）以及数学公式口语化播报清洗。
- **零构建前端看板**：基于原生 ES Modules 与本地化 `KaTeX`，提供左侧题目列表、中间实景画面与 SVG 像素级批改叠加层、右侧流式解析与追问对话区。

## 目录结构

```text
.
├── backend/                # FastAPI 后端服务
│   ├── app/
│   │   ├── vision/         # 相机采集、稳帧检测、页面匹配、透视矫正、切题与叠加标注
│   │   ├── grading/        # 手写识别客户端与云端判题引擎
│   │   ├── voice/          # VAD / STT / TTS 与语音会话控制
│   │   ├── knowledge/      # 知识点检索服务与基础题库数据
│   │   ├── engine.py       # 批改与会话核心编排
│   │   └── main.py         # HTTP / WebSocket / MJPEG 入口
│   └── tests/              # pytest 单元测试与回归测试
├── frontend/               # 原生 ES Modules 前端界面与静态资源
├── deploy/                 # 模型下载、单机 RTX 4090 启动脚本与部署说明
├── docs/                   # 架构契约、提示词规范与回归文档
├── data/                   # 试卷模板与 Mock 抓拍样例
├── tools/                  # 离线建库、回放诊断与基准评测脚本
└── .env.example            # 环境变量配置模板
```

## 快速开始

### 1. 环境准备

- Python `>=3.11, <3.13`
- [uv](https://docs.astral.sh/uv/) 包管理器
- 复制 `.env.example` 为 `.env`，并填入云端大模型 API 配置（`CLOUD_LLM_BASE_URL`、`CLOUD_LLM_API_KEY`、`CLOUD_LLM_MODEL` 等）。

```bash
cp .env.example .env
```

### 2. 本地开发启动（无 GPU / Mock 相机）

```bash
cd backend
uv sync
CAMERA_BACKEND=mock AUTO_CAPTURE=false TTS_PROVIDER=mock \
  FIXTURE_TEMPLATE="$PWD/tests/fixtures/template.json" \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动后访问 `http://127.0.0.1:8000`。

可选本地语音依赖（CPU/GPU）：

```bash
cd backend
uv sync --extra voice --extra kokoro
```

### 3. Linux + RTX 4090 完整部署

完整部署（含本地 `Qwen3.5-2B` 手写识别、`PaddleOCR-VL-1.5` 整页 OCR、`Qwen3-ASR-1.7B` 与 `Qwen3-TTS`）请参阅 [deploy/README.md](deploy/README.md) 与启动脚本 [deploy/start_4090.sh](deploy/start_4090.sh)。

## 运行测试

```bash
cd backend
uv run pytest -q
```

## 文档

- 系统架构与前后端接口契约：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 提示词与数据边界规范：[docs/PROMPTS.md](docs/PROMPTS.md)
- 视频回归与基准评测说明：[docs/VIDEO_REGRESSION.md](docs/VIDEO_REGRESSION.md)
- 采图样例说明：[docs/CAPTURE_SAMPLES.md](docs/CAPTURE_SAMPLES.md)
- 部署与运维指南：[deploy/README.md](deploy/README.md)

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
