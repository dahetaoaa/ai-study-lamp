#!/usr/bin/env bash
# 线上三vLLM常驻编排（单张 RTX 4090D 24GB 实测共存）：
# 8001 手写识别 Qwen3.5-2B + 8002 整页OCR定位 PaddleOCR-VL-1.5 + 8003 语音识别 Qwen3-ASR-1.7B + 8000 FastAPI(含Qwen3-TTS)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_VLLM="vllm"; [ -x "$ROOT/.venv-vllm/bin/vllm" ] && DEFAULT_VLLM="$ROOT/.venv-vllm/bin/vllm"
HWR_VLLM_BIN="${HWR_VLLM_BIN:-$DEFAULT_VLLM}"
OCR_VLLM_BIN="${OCR_VLLM_BIN:-$DEFAULT_VLLM}"
ASR_VLLM_BIN="${ASR_VLLM_BIN:-$DEFAULT_VLLM}"
UV_BIN="${UV_BIN:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
local_or() { [ -d "$ROOT/models/$1" ] && echo "$ROOT/models/$1" || echo "$2"; }
HWR_MODEL_PATH="${HWR_MODEL_PATH:-$(local_or Qwen3.5-2B Qwen/Qwen3.5-2B)}"
OCR_MODEL_PATH="${OCR_MODEL_PATH:-$(local_or PaddleOCR-VL-1.5 PaddlePaddle/PaddleOCR-VL-1.5)}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-$(local_or Qwen3-ASR-1.7B Qwen/Qwen3-ASR-1.7B)}"
TTS_MODEL="${TTS_MODEL:-$(local_or Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)}"
# 没有系统 gcc 时，Triton 用 ziglang 的 zig cc 编译（见 deploy/bin/cc）
if ! command -v gcc >/dev/null && [ -z "${CC:-}" ] && "$ROOT/.venv-vllm/bin/python" -c "import ziglang" 2>/dev/null; then
  export CC="$ROOT/deploy/bin/cc"
fi
# 局域网开麦克风需要 HTTPS：deploy/certs 下有证书就自动启用
SSL_ARGS=(); SCHEME=http
if [ -f "$ROOT/deploy/certs/cert.pem" ]; then
  SSL_ARGS=(--ssl-keyfile "$ROOT/deploy/certs/key.pem" --ssl-certfile "$ROOT/deploy/certs/cert.pem"); SCHEME=https
fi
HWR_PORT="${HWR_PORT:-8001}"
OCR_PORT="${OCR_PORT:-8002}"
ASR_PORT="${ASR_PORT:-8003}"
API_PORT="${API_PORT:-8000}"
LOG_DIR="${LOG_DIR:-$ROOT/deploy/logs}"
mkdir -p "$LOG_DIR"
pids=()
cleanup() {
  echo '正在停止本脚本启动的服务……'
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
wait_ready() {
  local pid="$1" url="$2" name="$3"
  for ((attempt=0;attempt<900;attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name 在就绪前退出，请检查 $LOG_DIR 内日志" >&2
      return 1
    fi
    if curl --fail --silent --insecure --max-time 2 "$url" >/dev/null; then
      echo "$name 已就绪"
      return 0
    fi
    sleep 1
  done
  echo "$name 等待就绪超时" >&2
  return 1
}

# 密钥只由FastAPI读取项目.env；不在命令行和日志中展开。
echo '启动手写识别 Qwen3.5-2B……'
"$HWR_VLLM_BIN" serve "$HWR_MODEL_PATH" \
  --served-model-name Qwen3.5-2B --host 127.0.0.1 --port "$HWR_PORT" \
  --trust-remote-code --max-model-len 4096 --max-num-seqs 16 \
  --mm-processor-kwargs '{"max_pixels":524288}' --kv-cache-memory-bytes 1610612736 --gpu-memory-utilization 0.35 \
  >"$LOG_DIR/hwr.log" 2>&1 &
pids+=("$!")
wait_ready "${pids[0]}" "http://127.0.0.1:$HWR_PORT/health" '手写识别服务'

echo '启动整页OCR与Spotting定位 PaddleOCR-VL-1.5……'
"$OCR_VLLM_BIN" serve "$OCR_MODEL_PATH" \
  --served-model-name PaddleOCR-VL-1.5 --host 127.0.0.1 --port "$OCR_PORT" \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.18 --kv-cache-memory-bytes 536870912 \
  >"$LOG_DIR/ocr.log" 2>&1 &
pids+=("$!")
wait_ready "${pids[1]}" "http://127.0.0.1:$OCR_PORT/health" '整页OCR服务'

echo '启动上下文偏置语音识别 Qwen3-ASR-1.7B……'
"$ASR_VLLM_BIN" serve "$ASR_MODEL_PATH" \
  --served-model-name Qwen3-ASR-1.7B --host 127.0.0.1 --port "$ASR_PORT" \
  --trust-remote-code --max-model-len 2048 --max-num-seqs 8 --kv-cache-memory-bytes 1073741824 --gpu-memory-utilization 0.35 \
  >"$LOG_DIR/asr.log" 2>&1 &
pids+=("$!")
wait_ready "${pids[2]}" "http://127.0.0.1:$ASR_PORT/health" '语音识别服务'

# 显式覆盖Mac .env中的云端HWR、OCR与CPU语音选项，但不改文件。
export HWR_BASE_URL="http://127.0.0.1:$HWR_PORT/v1" HWR_MODEL=Qwen3.5-2B
export OCR_BASE_URL="http://127.0.0.1:$OCR_PORT/v1" OCR_MODEL=PaddleOCR-VL-1.5
export HWR_API_KEY='' STT_PROVIDER=qwen3_asr STT_MODEL=Qwen3-ASR-1.7B
export STT_BASE_URL="http://127.0.0.1:$ASR_PORT/v1" STT_DEVICE=cuda
export TTS_PROVIDER="${TTS_PROVIDER:-qwen3_tts}" TTS_VOICE="${TTS_VOICE:-Vivian}"
export TTS_WARMUP="${TTS_WARMUP:-true}"
export TTS_MODEL HF_HUB_OFFLINE=1
echo '启动单worker FastAPI……'
"$UV_BIN" run --directory "$ROOT/backend" --no-sync uvicorn app.main:app \
  --host 0.0.0.0 --port "$API_PORT" --workers 1 "${SSL_ARGS[@]}" >"$LOG_DIR/backend.log" 2>&1 &
pids+=("$!")
wait_ready "${pids[3]}" "$SCHEME://127.0.0.1:$API_PORT/api/health" '学习台灯后端'
echo "已启动：$SCHEME://$(hostname -I | cut -d' ' -f1):$API_PORT，日志位于 $LOG_DIR"
# bash>=4.3（Linux）支持wait -n；任一服务退出即终止其他服务，不能静默半运行。
wait -n "${pids[@]}"
