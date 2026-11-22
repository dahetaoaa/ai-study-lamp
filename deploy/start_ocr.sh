#!/usr/bin/env bash
# 独立启动 PaddleOCR-VL-1.5 整页OCR与Spotting定位服务（0.9B，约4.5GB显存）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_VLLM="vllm"; [ -x "$ROOT/.venv-vllm/bin/vllm" ] && DEFAULT_VLLM="$ROOT/.venv-vllm/bin/vllm"
# 没有系统 gcc 时，Triton 用 ziglang 的 zig cc 编译（见 deploy/bin/cc）
if ! command -v gcc >/dev/null && [ -z "${CC:-}" ] && "$ROOT/.venv-vllm/bin/python" -c "import ziglang" 2>/dev/null; then
  export CC="$ROOT/deploy/bin/cc"
fi
MODEL_DIR="$ROOT/models/PaddleOCR-VL-1.5"
exec "${OCR_VLLM_BIN:-$DEFAULT_VLLM}" serve "${OCR_MODEL_PATH:-$([ -d "$MODEL_DIR" ] && echo "$MODEL_DIR" || echo PaddlePaddle/PaddleOCR-VL-1.5)}" \
  --served-model-name PaddleOCR-VL-1.5 --host 127.0.0.1 --port "${OCR_PORT:-8002}" \
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.18 --kv-cache-memory-bytes 536870912
