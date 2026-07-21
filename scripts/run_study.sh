#!/usr/bin/env bash
# Run the npcc conditional-copula simulation study on a CUDA GPU box.
#
# The study is light (~1.5 GB VRAM, ~3 GB RAM measured), so any modest GPU
# runs it. Per-cell checkpoints make it resumable and interruption-safe.
#
# Usage (from the repo root):
#   TABPFN_TOKEN=<token> CUDA_EXTRA=cu128 scripts/run_study.sh
#   # continue after an interruption: same command (it auto-resumes)
#
# Env vars:
#   TABPFN_TOKEN        (required) TabPFN license token.
#   CUDA_EXTRA          CUDA wheel to match the box's driver: cu126 | cu128 |
#                       cu130 | cu132  (default: cu128). `nvidia-smi` shows the
#                       driver; pick the closest CUDA runtime.
#   CONFIG              study config (default: configs/study.toml).
#   OUT                 output dir (default: results/study).
#   GPU_MEM_FRACTION    cap fraction of VRAM (default: 0.9; harmless on big GPUs).
#   WORKERS             concurrent cells (default: 1; keep 1 on <=16 GB GPUs).
set -euo pipefail

CUDA_EXTRA="${CUDA_EXTRA:-cu128}"
CONFIG="${CONFIG:-configs/study.toml}"
OUT="${OUT:-results/study}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.9}"
WORKERS="${WORKERS:-1}"

: "${TABPFN_TOKEN:?set TABPFN_TOKEN (TabPFN license token)}"
export TABPFN_TOKEN

command -v uv >/dev/null 2>&1 || {
  echo "uv not found — install from https://docs.astral.sh/uv/ then re-run." >&2
  exit 1
}

echo ">> syncing deps: --extra ${CUDA_EXTRA} --extra backends --extra experiments"
uv sync --extra "${CUDA_EXTRA}" --extra backends --extra experiments

echo ">> GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo ">> running study -> ${OUT} (resumable via per-cell checkpoints)"
uv run npcc-simstudy \
  --config "${CONFIG}" \
  --out "${OUT}" \
  --workers "${WORKERS}" \
  --resume \
  --gpu-mem-fraction "${GPU_MEM_FRACTION}" \
  "$@"

echo ">> done. Result tables + per-cell checkpoints under: ${OUT}"
