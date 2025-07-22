# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export LR=${1:-"3.2e-6"}
export MODEL="Llama-3.1-8B-Instruct"
export DATASET="alpaca"
export NUM_SAMPLES="-1"
export RESPONSE="synthetic"
export ATTACK="NaiveCompletion"
export OUTPUT_DIR="meta-llama/${MODEL}-dpo-full-${ATTACK}-randpos-${RESPONSE}-${DATASET}-ns${NUM_SAMPLES}-bs256-${LR}-maxlength2048"
export DATA_FILES="data/preference_${MODEL}_dpo_${ATTACK}_randpos_${RESPONSE}_${DATASET}.json"

echo "MODEL: $MODEL"
echo "DATASET: $DATASET"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "RESPONSE: $RESPONSE"
echo "ATTACK: $ATTACK"
echo "LR: $LR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATA_FILES: $DATA_FILES"

tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"

torchrun --nproc_per_node 8 torchtune/recipes/full_dpo_distributed.py \
    --config llama3.1_8b_full.yaml \
    output_dir=$OUTPUT_DIR \
    dataset.data_files=$DATA_FILES \
    optimizer.lr=$LR
