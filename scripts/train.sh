GPU_NUM=4

# Dataset paths for fine-tuning and reconstruction.
# - Use comma to separate multiple dataset paths.
# - The LAST path MUST be the reconstruction dataset.
# - Single path is also allowed — subfolders will be auto-split by `list_subfolders` in main_finetune.py
DATA_PATH="xx"  # e.g., "PATH_TO_GenImage_sd14_classfied,PATH_TO_GenImage_sd14_recon"

OUTPUT_DIR="xx" # directory to save checkpoints
LOG_DIR="${OUTPUT_DIR}/tensorboard" # path to tensorboard
MOE_CONFIG_PATH="xx" # path to MoE configuration file
PY_ARGS=${@:1}  


## OmniAID
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID \
    --batch_size 32 \
    --img_size 336 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR/ \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage1_hard_sampling" \
    ${PY_ARGS}

torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID \
    --batch_size 32 \
    --img_size 336 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage2_router_training" \
    ${PY_ARGS}



## OmniAID-LoRA
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_LoRA \
    --batch_size 32 \
    --img_size 336 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR/ \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage1_hard_sampling" \
    ${PY_ARGS}

torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_LoRA \
    --batch_size 32 \
    --img_size 336 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage2_router_training" \
    ${PY_ARGS}



## OmniAID-DINO
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO \
    --batch_size 32 \
    --img_size 448 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR/ \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage1_hard_sampling" \
    ${PY_ARGS}

torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO \
    --batch_size 32 \
    --img_size 448 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage2_router_training" \
    ${PY_ARGS}




## OmniAID-DINO-LoRA
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO_LoRA \
    --batch_size 32 \
    --img_size 448 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR/ \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage1_hard_sampling" \
    ${PY_ARGS}

torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO_LoRA \
    --batch_size 32 \
    --img_size 448 \
    --blr 2e-4 \
    --epochs 1 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --is_hybrid True \
    --training_mode "stage2_router_training" \
    ${PY_ARGS}