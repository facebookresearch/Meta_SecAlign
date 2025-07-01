# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export MODEL=$1
export DEFENSE=$2

cwd=$(pwd)
mkdir -p $cwd/agentdojo/runs/${MODEL}
nohup vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype auto \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --enable-lora \
    --max-lora-rank 64 \
    --chat_template $cwd/data/chat_template.jinja \
    --lora-modules meta-llama/Llama-3.3-70B-Instruct-SecAlign=$cwd/${MODEL} > $cwd/agentdojo/runs/${MODEL}/vllm_server.out &

sleep 600   # wait for the vllm server to spin up, may need to increase if checkpoint loading is slow

cd agentdojo/src
if [ $DEFENSE == "None" ]; then
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir $cwd/agentdojo/runs/$MODEL --model-id meta-llama/Llama-3.3-70B-Instruct-SecAlign --tool-delimiter input"
else
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir $cwd/agentdojo/runs/$MODEL --model-id meta-llama/Llama-3.3-70B-Instruct-SecAlign --tool-delimiter input --defense $DEFENSE"
fi

eval "$CMD"
for ATTACK in "direct" "ignore_previous" "important_instructions"; do
    eval "$CMD --attack $ATTACK"
done
