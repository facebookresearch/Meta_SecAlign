# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export MODEL="Llama-3.1-8B-Instruct"
export DATASET="alpaca"
export NUM_SAMPLES="-1"
export RESPONSE="synthetic"
export ATTACK="NaiveCompletion"
export LR="1.6e-4"
export OUTPUT_DIR="meta-llama/${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}_ns${NUM_SAMPLES}_bs256_${LR}_maxlength2048"
export DATA_FILES="data/preference_${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}.json"

tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

torchrun --nproc_per_node 8 torchtune/recipes/lora_dpo_distributed.py \
    --config llama3.1_8B_lora.yaml \
    output_dir=$OUTPUT_DIR \
    dataset.data_files=$DATA_FILES \
    optimizer.lr=$LR
