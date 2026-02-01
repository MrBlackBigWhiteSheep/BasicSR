#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

# 用法示例 训练水下去噪 Restormer：
# bash scripts/run/run_Restormer.sh train underwater 0

# ==== Restormer x1 图像恢复任务路径 =====
restormer_train_underwater="options/train/Restormer/train_Restormer_x1_underwater.yml"
restormer_train_hazy="options/train/Restormer/train_Restormer_x1_hazy.yml"
restormer_train_lol="options/train/Restormer/train_Restormer_x1_lol.yml"
restormer_train_rain="options/train/Restormer/train_Restormer_x1_rain.yml"
restormer_train_snow="options/train/Restormer/train_Restormer_x1_snow.yml"

# ===== 读取参数 =====
MODE=$1                 # train | test
TASK=${2:-"underwater"}  # underwater | hazy | lol | rain | snow
GPU_ID=${3:-"0"}       # GPU id 列表，形如 "0" 或 "0,1"

# 任务选择
case "$TASK" in
  underwater) TrainConfig=$restormer_train_underwater ;;
  hazy)       TrainConfig=$restormer_train_hazy ;;
  lol)        TrainConfig=$restormer_train_lol ;;
  rain)       TrainConfig=$restormer_train_rain ;;
  snow)       TrainConfig=$restormer_train_snow ;;
  *)
    echo "[Error] Unknown TASK: $TASK. Use underwater|hazy|lol|rain|snow";
    exit 1;
    ;;
esac

# 暂无独立测试配置，默认与训练配置一致
TestConfig=$TrainConfig

# ===== 判断模式 =====
case "$MODE" in
  train)
    NUM_GPUS=$(echo $GPU_ID | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))

    echo ">>> Running TRAIN mode on GPU(s): $GPU_ID (Count: $NUM_GPUS)"

    if [ "$NUM_GPUS" -gt 1 ]; then
      # === 多卡模式 (DDP) ===
      CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=4322 \
        --use_env \
        basicsr/train.py -opt $TrainConfig --launcher pytorch --auto_resume
    else
      # === 单卡模式 ===
      CUDA_VISIBLE_DEVICES=$GPU_ID python basicsr/train.py -opt $TrainConfig --auto_resume
    fi
    ;;

  test)
    echo ">>> Running TEST mode on GPU(s): $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python basicsr/test.py -opt $TestConfig
    ;;

  *)
    echo "Usage:"
    echo "  bash run_Restormer.sh train [task] [gpu_id]  # task: underwater|hazy|lol|rain|snow"
    echo "  bash run_Restormer.sh test  [task] [gpu_id]"
    echo "  默认 task=underwater, gpu_id=0"
    exit 1
    ;;
esac