# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export MODEL="Llama-3.3-70B-Instruct"
export DATASET="alpaca"
export NUM_SAMPLES="-1"
export RESPONSE="synthetic"
export ATTACK="NaiveCompletion"
export LR="3.2e-4"
export OUTPUT_DIR="meta-llama/${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}_ns${NUM_SAMPLES}_bs256_${LR}_maxlength2048"
export DATA_FILES="data/preference_${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}.json"

tune download meta-llama/Llama-3.3-70B-Instruct \
    --output-dir meta-llama/Llama-3.3-70B-Instruct \
    --ignore-patterns "original/consolidated*"

torchrun --nproc_per_node 8 torchtune/recipes/full_dpo_distributed.py \
    --config llama3.3_70B_full.yaml \
    output_dir=$OUTPUT_DIR \
    dataset.data_files=$DATA_FILES \
    optimizer.lr=$LR
