#!/usr/bin/env bash
# 仅离线建模板时使用；不与线上17GB模型组同时常驻。
set -euo pipefail
exec "${OCR_VLLM_BIN:-vllm}" serve "${OCR_MODEL_PATH:-PaddlePaddle/PaddleOCR-VL}" \
  --served-model-name PaddleOCR-VL-0.9B --host 127.0.0.1 --port "${OCR_PORT:-8002}" \
  --trust-remote-code --max-model-len 8192 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 --gpu-memory-utilization 0.18
