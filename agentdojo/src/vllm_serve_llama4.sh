nohup vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --dtype auto \
    --tensor-parallel-size 8 \
    --max-model-len 8192 > vllm_server_logs_1.out &
