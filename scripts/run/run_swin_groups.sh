#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

echo "[INFO] All grouped SwinIR tasks completed."
# 分两组，可在不同终端分别跑：
# Group0: hazy + rain
# Group1: underwater + lol + snow
# 用法：
#   bash scripts/run/run_swin_groups.sh group0 <GPU_ID>   # 例如 GPU 0
#   bash scripts/run/run_swin_groups.sh group1 <GPU_ID>   # 例如 GPU 2
# 如果不指定 GPU_ID，默认 0。

GROUP0=(hazy rain)
GROUP1=(underwater lol snow)

GROUP_NAME=$1
GPU=${2:-0}

if [ -z "$GROUP_NAME" ]; then
  echo "Usage:"
  echo "  bash run_swin_groups.sh group0 <gpu_id>  # hazy + rain"
  echo "  bash run_swin_groups.sh group1 <gpu_id>  # underwater + lol + snow"
  exit 1
fi

case "$GROUP_NAME" in
  group0) TASKS=("${GROUP0[@]}") ;;
  group1) TASKS=("${GROUP1[@]}") ;;
  *)
    echo "[Error] Unknown group: $GROUP_NAME. Use group0 or group1." >&2
    exit 1
    ;;
esac

for T in "${TASKS[@]}"; do
  echo "======================"
  echo "[INFO] Start training task: $T on GPU $GPU"
  echo "======================"
  bash "$SCRIPT_DIR/run_swin.sh" train "$T" "$GPU"
  status=$?
  if [ $status -ne 0 ]; then
    echo "[ERROR] Task $T failed with code $status. Stop." >&2
    exit $status
  fi
  echo "[INFO] Task $T finished."
  # 可选：sleep 60
done

echo "[INFO] $GROUP_NAME completed."
