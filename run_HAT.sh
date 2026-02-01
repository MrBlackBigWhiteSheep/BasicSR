#!/bin/bash

# 用法示例：
#   bash run_HAT.sh train underwater 0
#   bash run_HAT.sh train hazy 0,1
#   bash run_HAT.sh test  lol 0

hat_train_underwater="options/train/HAT/train_HAT_x1_underwater.yml"
hat_train_hazy="options/train/HAT/train_HAT_x1_hazy.yml"
hat_train_lol="options/train/HAT/train_HAT_x1_lol.yml"
hat_train_rain="options/train/HAT/train_HAT_x1_rain.yml"
hat_train_snow="options/train/HAT/train_HAT_x1_snow.yml"


# Python executable (fallback to python3 if python is missing)
PYTHON_BIN=${PYTHON_BIN:-python3}


MODE=$1                   # train | test
TARGET=${2:-"underwater"} # underwater | hazy | lol | rain | snow
GPU_ID=${3:-"0"}

case "$TARGET" in
  underwater) TrainConfig=$hat_train_underwater ;;
  hazy)            TrainConfig=$hat_train_hazy ;;
  lol)             TrainConfig=$hat_train_lol ;;
  rain)            TrainConfig=$hat_train_rain ;;
  snow)            TrainConfig=$hat_train_snow ;;
  *)
    echo "[Error] Unknown target: $TARGET. Use underwater|hazy|lol|rain|snow"
    exit 1
    ;;
esac

# 默认测试配置同训练配置；如需自定义可额外添加 test_HAT_*.yml
TestConfig=$TrainConfig

case "$MODE" in
  train)
    NUM_GPUS=$(echo $GPU_ID | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))

    echo ">>> Running TRAIN on GPU(s): $GPU_ID (Count: $NUM_GPUS)"

    if [ "$NUM_GPUS" -gt 1 ]; then
      CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=4422 \
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
  echo "  bash run_HAT.sh train [underwater|hazy|lol|rain|snow] [gpu_id]"
  echo "  bash run_HAT.sh test  [underwater|hazy|lol|rain|snow] [gpu_id]"
    exit 1
    ;;
esac
