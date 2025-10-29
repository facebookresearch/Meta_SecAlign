# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export MODEL="Llama-3.3-70B-Instruct"
export RESPONSE="synthetic"
export RANDPOS="randpos_"
export LR="3.2e-4"
export OUTPUT_DIR="meta-llama/${MODEL}_dpo_NaiveCompletion_${RANDPOS}${RESPONSE}_alpaca_lr${LR}"
export DATA_FILES="data/preference_${MODEL}_dpo_NaiveCompletion_${RANDPOS}${RESPONSE}_alpaca.json"
export CACHE="~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# Generate DATA_FILES, will be skipped if already exists
python generate_data.py -m ${MODEL} --dataset alpaca --self_generated_response --random_inject_pos

tune run --nproc_per_node 8 lora_dpo_distributed \
    --config helpers/llama3.3_70B_lora.yaml \
    output_dir=$OUTPUT_DIR \
    dataset.data_files=$DATA_FILES \
    optimizer.lr=$LR \
    cache_dir=${CACHE/#\~/$HOME}