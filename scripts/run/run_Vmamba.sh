#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

# 用法示例：
#   bash scripts/run/run_Vmamba.sh train underwater 0
#   bash scripts/run/run_Vmamba.sh train hazy 0,1
#   bash scripts/run/run_Vmamba.sh test  rain 0

# 训练配置（Vmamba / Mamber32）
vm_train_underwater="options/train/Vmamba/train_Vmamba_x1_underwater.yml"
vm_train_hazy="options/train/Vmamba/train_Vmamba_x1_hazy.yml"
vm_train_lol="options/train/Vmamba/train_Vmamba_x1_lol.yml"
vm_train_rain="options/train/Vmamba/train_Vmamba_x1_rain.yml"
vm_train_snow="options/train/Vmamba/train_Vmamba_x1_snow.yml"

# 测试配置（与训练相对应）
vm_test_underwater="options/test/Vmamba/test_Vmamba_x1_underwater.yml"
vm_test_hazy="options/test/Vmamba/test_Vmamba_x1_hazy.yml"
vm_test_lol="options/test/Vmamba/test_Vmamba_x1_lol.yml"
vm_test_rain="options/test/Vmamba/test_Vmamba_x1_rain.yml"
vm_test_snow="options/test/Vmamba/test_Vmamba_x1_snow.yml"

# Python executable
PYTHON_BIN=${PYTHON_BIN:-python3}

MODE=$1                    # train | test
TASK=${2:-"underwater"}   # underwater | hazy | lol | rain | snow
GPU_ID=${3:-"0"}

case "$TASK" in
  underwater) TrainConfig=$vm_train_underwater; TestConfig=$vm_test_underwater ;;
  hazy)       TrainConfig=$vm_train_hazy;       TestConfig=$vm_test_hazy ;;
  lol)        TrainConfig=$vm_train_lol;        TestConfig=$vm_test_lol ;;
  rain)       TrainConfig=$vm_train_rain;       TestConfig=$vm_test_rain ;;
  snow)       TrainConfig=$vm_train_snow;       TestConfig=$vm_test_snow ;;
  *)
    echo "[Error] Unknown TASK: $TASK. Use underwater|hazy|lol|rain|snow" >&2
    exit 1
    ;;
 esac

case "$MODE" in
  train)
    NUM_GPUS=$(echo $GPU_ID | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
    echo ">>> Running TRAIN on GPU(s): $GPU_ID (Count: $NUM_GPUS)"
    if [ "$NUM_GPUS" -gt 1 ]; then
      CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=4522 \
        --use_env \
        basicsr/train.py -opt $TrainConfig --launcher pytorch --auto_resume
    else
      CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN basicsr/train.py -opt $TrainConfig --auto_resume
    fi
    ;;
  test)
    echo ">>> Running TEST on GPU(s): $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN basicsr/test.py -opt $TestConfig
    ;;
  *)
    echo "Usage:"
    echo "  bash scripts/run/run_Vmamba.sh train [underwater|hazy|lol|rain|snow] [gpu_id]"
    echo "  bash scripts/run/run_Vmamba.sh test  [underwater|hazy|lol|rain|snow] [gpu_id]"
    exit 1
    ;;
 esac
