#!/bin/bash

# ==== 配置路径：SwinIR x1 图像恢复任务 =====
swin_train_underwater="options/train/SwinIR/train_SwinIR_x1_underwater.yml"
swin_train_lol="options/train/SwinIR/train_SwinIR_x1_lol.yml"
swin_train_hazy="options/train/SwinIR/train_SwinIR_x1_hazy.yml"
swin_train_rain="options/train/SwinIR/train_SwinIR_x1_rain.yml"
swin_train_snow="options/train/SwinIR/train_SwinIR_x1_snow.yml"

# ===== 读取参数 =====
MODE=$1              # train | test
TASK=${2:-"underwater"}  # underwater | lol | hazy | rain | snow
GPU_ID=${3:-"0"}    # 默认单卡 3090（GPU 0）

# 任务选择
case "$TASK" in
  underwater) TrainConfig=$swin_train_underwater ;;
  lol)        TrainConfig=$swin_train_lol ;;
  hazy)       TrainConfig=$swin_train_hazy ;;
  rain)       TrainConfig=$swin_train_rain ;;
  snow)       TrainConfig=$swin_train_snow ;;
  *)
    echo "[Error] Unknown TASK: $TASK. Use underwater|lol|hazy|rain|snow";
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
      --master_port=4321 \
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
    echo "  sh run.sh train [gpu_id]    # 训练，例如：sh run.sh train 0"
    echo "  sh run.sh test [gpu_id]     # 测试，例如：sh run.sh test 1"
    echo "  默认 GPU ID 为 0"
    exit 1
    ;;
esac