GPU_NUM=4
EVAL_DATA_PATH="xx" # path to evaluation dataset
OUTPUT_DIR="xx" # directory to save results
RESUME="xx" # path to model weight for eval
MOE_CONFIG_PATH="xx" # path to MoE configuration file
PY_ARGS=${@:1}


#OmniAID
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID \
    --batch_size 100 \
    --img_size 336 \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME \
    --moe_config_path $MOE_CONFIG_PATH \
    --eval True \
    --is_hybrid True
    ${PY_ARGS}

#OmniAID-LoRA
torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_LoRA \
    --batch_size 100 \
    --img_size 336 \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME \
    --moe_config_path $MOE_CONFIG_PATH \
    --eval True \
    --is_hybrid True
    ${PY_ARGS}

## OmniAID-DINO
torchrun --master_port=$MASTER_PORT --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO \
    --batch_size 100 \
    --img_size 448 \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME \
    --moe_config_path $MOE_CONFIG_PATH \
    --eval True \
    --is_hybrid True
    ${PY_ARGS}


## OmniAID-DINO
torchrun --master_port=$MASTER_PORT --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID_DINO_LoRA \
    --batch_size 100 \
    --img_size 448 \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME \
    --moe_config_path $MOE_CONFIG_PATH \
    --eval True \
    --is_hybrid True
    ${PY_ARGS}
