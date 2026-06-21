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

# FlashInfer JIT-compiles its sampling kernel at runtime and needs nvcc (full CUDA
# toolkit), absent here (driver only, no /usr/local/cuda). Native sampler avoids it.
# Logprobs are unaffected.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "serving ${MODEL} on :${PORT}  max_len=${MAX_LEN}  gpu_util=${GPU_UTIL}  name=${SERVED_NAME}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-prefix-caching
