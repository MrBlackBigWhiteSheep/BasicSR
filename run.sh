#!/bin/bash

# ==== 配置路径 Train =====
train_cfg="options/train/underwater/train_mambav2_lightSR_UEIB.yml"
test_cfg="options/test/underwater/test_mambav2_lightSR_UEIB.yml"

# mambawater_train_cfg1="options/train/underwater/train_mamba_water_enhance.yml"
# mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage1_pretrain.yml"
mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage2_soat.yml"
# mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage3_soat.yml"
# mambawater_train_cfg1="options/train/underwater/train_mamba_water_enhance.yml"
# mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage1_pretrain.yml"
# mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage2_soat.yml"
mambawater_train_cfg1="options/train/underwater/train_mamba_water_stage3_soat.yml"

mambawater_train_cfg2="options/train/underwater/train_mamba_rain.yml"

mambawater_train_cfg3="options/train/underwater/train_mamba_lol.yml"

mambawater_train_cfg4="options/train/underwater/train_mamba_hazy.yml"

mambawater_train_cfg5="options/train/underwater/train_mamba_snow.yml"

# ==== 配置路径 Test =====
mambawater_test_cfg1="options/test/underwater/test_mamba_water.yml"

mambawater_test_cfg2="options/test/underwater/test_mamba_rain.yml"

mambawater_test_cfg3="options/test/underwater/test_mamba_lol.yml"

mambawater_test_cfg4="options/test/underwater/test_mamba_hazy.yml"

mambawater_test_cfg5="options/test/underwater/test_mamba_snow.yml"

# ===== 配置文件选择 =====
TrainConfig=$mambawater_train_cfg1

TestConfig=$mambawater_test_cfg1
# ===== 读取参数 =====
MODE=$1
GPU_ID=${2:-"0"}  # 默认使用 GPU 0，如果未提供第二个参数

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