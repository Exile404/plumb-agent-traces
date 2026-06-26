#!/usr/bin/env bash
set -euo pipefail

# vLLM serve for Plumb. Long-running: run this in its own terminal (or tmux).
# Overridable via env for the OOM/quant ladder, e.g.
#   PLUMB_MAX_LEN=8192 bash scripts/01_start_vllm.sh
#   PLUMB_GPU_UTIL=0.80 bash scripts/01_start_vllm.sh

MODEL="${PLUMB_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
PORT="${PLUMB_PORT:-8000}"
MAX_LEN="${PLUMB_MAX_LEN:-16384}"
GPU_UTIL="${PLUMB_GPU_UTIL:-0.85}"
SERVED_NAME="${PLUMB_SERVED_NAME:-qwen-coder}"
KV_DTYPE="${PLUMB_KV_DTYPE:-auto}"   # set PLUMB_KV_DTYPE=fp8 to ~halve KV mem and allow PLUMB_MAX_LEN=32768

# Native function-calling for OpenHands (Qwen2.5 = hermes parser). Opt-in: PLUMB_TOOLS=1
TOOL_FLAGS=()
if [[ "${PLUMB_TOOLS:-0}" == "1" ]]; then
  TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser hermes)
fi

# FlashInfer JIT-compiles its sampling kernel at runtime and needs nvcc (full CUDA
# toolkit), absent here (driver only, no /usr/local/cuda). Native sampler avoids it.
# Logprobs are unaffected.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "serving ${MODEL} on :${PORT}  max_len=${MAX_LEN}  gpu_util=${GPU_UTIL}  kv=${KV_DTYPE}  name=${SERVED_NAME}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --kv-cache-dtype "${KV_DTYPE}" \
  --enable-prefix-caching \
  "${TOOL_FLAGS[@]}"
