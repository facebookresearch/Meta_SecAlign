if [ $2 == "None" ]; then
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir runs/$1 --model-id meta-llama/Llama-3.3-70B-Instruct"
else
    export CMD="python -m agentdojo.scripts.benchmark --model local --logdir runs/$1 --model-id meta-llama/Llama-3.3-70B-Instruct --defense $2"
fi

eval "$CMD"
for ATTACK in "direct" "ignore_previous" "important_instructions"; do
    eval "$CMD --attack $ATTACK"
done