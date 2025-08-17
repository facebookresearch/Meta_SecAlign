# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export MODEL_PATH=${1:-"meta-llama/Llama-3.3-70B-Instruct"}
export DEFENSE=${2:-"None"}
export USE_LORA=${3:-"True"}
export LORA_PATH=${4:-"meta-llama/Llama-3.3-70B-Instruct_SecAlign_official"}

export MODEL=$(basename "$MODEL_PATH")
cwd=$(pwd)

echo "Running agentdojo benchmark with model: $MODEL_PATH"
echo "Using defense: $DEFENSE"
echo "Using LoRA: $USE_LORA"
echo "LoRA path: $LORA_PATH"
echo "Model Name: ${MODEL}"

mkdir -p $cwd/agentdojo/runs/${MODEL}
if [[ "$MODEL_PATH" == *"Scout"* ]]; then
    nohup vllm serve --port 8000 ${MODEL_PATH}\
        --dtype auto \
        --host 0.0.0.0 \
        --tensor-parallel-size 8 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.8 \
        --disable-hybrid-kv-cache-manager \
        --enforce-eager \
        --chat_template $cwd/data/tool_chat_template_llama4_json.jinja \
        --enable-lora \
        --max-lora-rank 64 \
        --lora-modules meta-Llama-lora=${LORA_PATH} > $cwd/agentdojo/runs/${MODEL}/vllm_server.out &
else
    nohup vllm serve --port 8000 ${MODEL_PATH}\
        --dtype auto \
        --host 0.0.0.0 \
        --tensor-parallel-size 4 \
        --max-model-len 16384 \
        --chat_template $cwd/data/chat_template.jinja \
        --enable-lora \
        --max-lora-rank 64 \
        --lora-modules meta-Llama-lora=${LORA_PATH} > $cwd/agentdojo/runs/${MODEL}/vllm_server.out &
fi

# Wait for the vllm server to start
while ! nc -z localhost 8000; do
    sleep 15
    echo "... still offline"
done

sleep 5

if [ "$USE_LORA" == "True" ]; then
    export MODEL_ID="meta-Llama-lora"
else
    export MODEL_ID=${MODEL_PATH}
fi

echo "Model ID used in vllm: ${MODEL_ID}"

cd agentdojo/src
if [ $DEFENSE == "None" ]; then
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir $cwd/agentdojo/runs/$MODEL --model-id ${MODEL_ID} --tool-delimiter input"
else
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir $cwd/agentdojo/runs/$MODEL --model-id ${MODEL_ID} --tool-delimiter input --defense $DEFENSE"
fi

eval "$CMD"
for ATTACK in "direct" "ignore_previous" "important_instructions"; do
    eval "$CMD --attack $ATTACK"
done
