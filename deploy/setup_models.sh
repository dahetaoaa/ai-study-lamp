#!/usr/bin/env bash
# 4090 一次性准备：依赖走清华 PyPI 镜像，模型走 ModelScope，三路并行。
# 用法: deploy/setup_models.sh   日志: deploy/logs/setup_models.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=300
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
cd "$ROOT"
pids=()

(
  uv venv -p 3.12 .venv-ms -q
  uv pip install --python .venv-ms/bin/python -q modelscope
  echo MS_OK
  for m in Qwen/Qwen3.5-2B Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-Tokenizer-12Hz; do
    if ! .venv-ms/bin/modelscope download --model "$m" --local_dir "$MODELS/${m#Qwen/}"; then
      echo "DL_FAIL $m" >&2
      exit 1
    fi
    echo "DL_OK $m"
  done
) &
pids+=("$!")
(
  uv python install 3.12 -q
  uv venv -q --python-preference only-managed -p 3.12 .venv-vllm
  # ASR音频解码依赖必须装在vLLM环境；仅/models可用不代表WAV转写可用。
  uv pip install --python .venv-vllm/bin/python vllm ziglang librosa soundfile av
  echo VLLM_DONE
) &
pids+=("$!")
(
  cd backend
  uv sync --extra voice --extra qwen-tts
  echo VOICE_DONE
) &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo SETUP_FAILED >&2; exit 1; fi
echo ALL_DONE
