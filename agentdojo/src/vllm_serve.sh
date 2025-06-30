
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype auto \
    --host 0.0.0.0 \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules '{"name": "meta-llama/Llama-3.3-70B-Instruct-SecAlign", "path": "/fsx-memorization/chuanguo/SecAlign/meta-llama/Llama-3.3-70B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca_ns-1_bs512_3.2e-4_maxlength1024/", "base_model_name": "meta-llama/Llama-3.3-70B-Instruct"}'
