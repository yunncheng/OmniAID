GPU_NUM=4
EVAL_DATA_PATH="/mnt/shared-storage-user/ailab-bigdatasddh/datasets/Deepfake/Chameleon/test"
OUTPUT_DIR="/mnt/shared-storage-user/guoyuncheng/Project/OmniAID/output_mirage/Chameleon"
RESUME="/mnt/shared-storage-user/guoyuncheng/Project/AIDE_s33/output/OmniAID_Mirage/lr2e-4_epoch1_1resize_noaug_gatingcrossentroy_0/router/checkpoint-0.pth"
MOE_CONFIG_PATH="${OUTPUT_DIR}/config.json"
PY_ARGS=${@:1}  


torchrun --nproc-per-node=$GPU_NUM main_finetune.py \
    --model OmniAID \
    --batch_size 100 \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --resume $RESUME \
    --moe_config_path $MOE_CONFIG_PATH \
    --eval True \
    --training_mode "stage2_router_training" \
    ${PY_ARGS}
