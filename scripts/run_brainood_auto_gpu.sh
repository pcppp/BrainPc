#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml}"
shift || true

MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MAX_USED_MB="${MAX_USED_MB:-500}"
MAX_UTIL="${MAX_UTIL:-10}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "#E# nvidia-smi not found."
  exit 1
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate pc2 >/dev/null 2>&1 || true
fi


QUERY_OUT="$(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)"
BEST_LINE="$(printf '%s\n' "$QUERY_OUT" | awk -F', *' -v min_free="$MIN_FREE_MB" -v max_used="$MAX_USED_MB" -v max_util="$MAX_UTIL" '
  ($4 + 0) >= min_free && ($3 + 0) <= max_used && ($5 + 0) <= max_util {print}
' | sort -t',' -k4,4nr | head -n1)"

if [ -z "$BEST_LINE" ]; then
  echo "#W# No GPU satisfies free>=$MIN_FREE_MB MB, used<=$MAX_USED_MB MB, util<=$MAX_UTIL%. Falling back to the GPU with most free memory."
  BEST_LINE="$(printf '%s\n' "$QUERY_OUT" | sort -t',' -k4,4nr | head -n1)"
fi

GPU_INDEX="$(printf '%s' "$BEST_LINE" | cut -d',' -f1 | xargs)"
GPU_FREE="$(printf '%s' "$BEST_LINE" | cut -d',' -f4 | xargs)"
GPU_USED="$(printf '%s' "$BEST_LINE" | cut -d',' -f3 | xargs)"
GPU_UTIL="$(printf '%s' "$BEST_LINE" | cut -d',' -f5 | xargs)"

echo "#IN# Selected physical GPU $GPU_INDEX (used=${GPU_USED}MB free=${GPU_FREE}MB util=${GPU_UTIL}%)."
echo "#IN# Running config: $CONFIG_PATH"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$REPO_ROOT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "#IN# DRY_RUN=1, command not executed."
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=$PYTHONPATH python -m GOOD.kernel.main --config_path $CONFIG_PATH --gpu_idx 0 $*"
  exit 0
fi

python -m GOOD.kernel.main --config_path "$CONFIG_PATH" --gpu_idx 0 "$@"
