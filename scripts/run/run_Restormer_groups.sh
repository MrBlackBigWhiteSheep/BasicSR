#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

# ==== Restormer x1 图像恢复任务路径 =====
restormer_train_underwater="options/train/Restormer/train_Restormer_x1_underwater.yml"
restormer_train_hazy="options/train/Restormer/train_Restormer_x1_hazy.yml"
restormer_train_lol="options/train/Restormer/train_Restormer_x1_lol.yml"
restormer_train_rain="options/train/Restormer/train_Restormer_x1_rain.yml"
restormer_train_snow="options/train/Restormer/train_Restormer_x1_snow.yml"

# GROUP0=(hazy rain)
GROUP0=(underwater lol snow)
# GROUP1=(underwater lol snow)

GROUP_NAME=$1
GPU_ID=${2:-0}

if [ -z "$GROUP_NAME" ]; then
  echo "Usage:"
  echo "  bash scripts/run/run_Restormer_groups.sh group0 <gpu_id>  # hazy + rain"
  echo "  bash scripts/run/run_Restormer_groups.sh group1 <gpu_id>  # underwater + lol + snow"
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

# 对于组内任务，通常在同一个 GPU 上按顺序执行
for T in "${TASKS[@]}"; do
  # 任务配置选择
  case "$T" in
    underwater) TrainConfig=$restormer_train_underwater ;;
    hazy)       TrainConfig=$restormer_train_hazy ;;
    lol)        TrainConfig=$restormer_train_lol ;;
    rain)       TrainConfig=$restormer_train_rain ;;
    snow)       TrainConfig=$restormer_train_snow ;;
  esac

  echo "======================"
  echo "[INFO] Start training task: $T on GPU $GPU_ID"
  echo "======================"

  NUM_GPUS=$(echo $GPU_ID | tr -cd ',' | wc -c)
  NUM_GPUS=$((NUM_GPUS + 1))

  if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch \
      --nproc_per_node=$NUM_GPUS \
      --master_port=4322 \
      --use_env \
      basicsr/train.py -opt $TrainConfig --launcher pytorch --auto_resume
      # basicsr/train.py -opt $TrainConfig --launcher pytorch
  else
    CUDA_VISIBLE_DEVICES=$GPU_ID python basicsr/train.py -opt $TrainConfig --auto_resume
    # CUDA_VISIBLE_DEVICES=$GPU_ID python basicsr/train.py -opt $TrainConfig
  fi

  echo "[INFO] Task $T finished."
  echo "----------------------"
done

echo "[INFO] All tasks in $GROUP_NAME completed."
