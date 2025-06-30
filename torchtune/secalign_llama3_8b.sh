export MODEL="Meta-Llama-3-8B-Instruct"
export DATASET="alpaca"
export NUM_SAMPLES="-1"
export RESPONSE="synthetic"
export ATTACK="NaiveCompletion"
export LR="0.8e-4"
export OUTPUT_DIR="/checkpoint/memorization/chuanguo/meta-llama/${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}_ns${NUM_SAMPLES}_bs256_${LR}_maxlength1024_epoch1"
export DATA_FILES="/home/chuanguo/SecAlign/data/preference_${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}.json"

torchrun --nproc_per_node 8 recipes/lora_dpo_distributed.py \
    --config llama3_8B_lora.yaml \
    output_dir=$OUTPUT_DIR \
    dataset.data_files=$DATA_FILES \
    optimizer.lr=$LR
