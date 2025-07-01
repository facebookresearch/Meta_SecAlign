# Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks
[Sizhe Chen](https://sizhe-chen.github.io)\*, [Arman Zharmagambetov](https://arman-z.github.io), [David Wagner](https://people.eecs.berkeley.edu/~daw), [Chuan Guo](https://sites.google.com/view/chuanguo)\*

[![](https://img.shields.io/badge/Paper-a8c66c)](https://arxiv.org/pdf/2410.05451) [![](https://img.shields.io/badge/Media-75291c)](https://bair.berkeley.edu/blog/2025/04/11/prompt-injection-defense)

Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level
defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in
a closed-source manner. We believe open-source models are needed by the AI security community,
where co-development of attacks and defenses through open research drives scientific progress in
mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first
open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade
model performancea. We provide complete details of our training recipe, which utilizes an improved
version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks
show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers
security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition
general instruction-following. Our best model—Meta-SecAlign-70B—achieves state-of-the-art
robustness against prompt injection attacks and comparable utility to closed-source commercial LLM
with model-level defense.

# Environment
+ Hardware requirements. 8B: 4 80G A100s for training, 1 16G GPU for evaluation. 70B: 8 141GB H200s for training, 4 (recommending 8 for efficiency) 80G A100s for evaluation.
> git clone https://github.com/facebookresearch/SecAlign \
> cd SecAlign
+ Install environment dependencies for evaluation
> conda env create -n secalign python==3.12 \
> pip install -r requirements.txt
+ Download dependencies (run with 1 or 4 available GPUs to trigger necessary inference)
> python setup.py
+ Configure openai dependencies for utility evaluation: create ```data/openai_configs.yaml``` following ```data/openai_configs_examle.yaml```
+ Configure gemini keys in ```data/gemini_configs.yaml```
> default: \n - client_class: "genai.Client" \n - api_key: "..."

# SecAlign Preference Optimization (Optional)
+ To train SecAlign models from scratch, first install the conda environment for training
> conda env create -n secalign_training python==3.12 \
> conda env update -f requirements_training.yml \
> pip install -r requirements_training.txt --no-deps
Then install the custom torchtune package locally from source
> cd torchtune
> pip install -e .
+ Run the following script to train the 8B or 70B model: ```bash secalign_llama3.1_8b.sh``` or ```bash secalign_llama3.3_70b.sh```

# Evaluation
> python run_tests.py -m [model_path] --lora_alpha [lora_alpha]
+ ```model_path``` is the path to the tested model, e.g., ```meta-llama/Llama-3.1-8B-Instruct_SecAlign, meta-llama/Llama-3.3-70B-Instruct_SecAlign, gpt-4o-mini, gpt-4o, gemini-2.0-flash, gemini-2.5-flash```.
+ ```lora_alpha``` is default to 8 (as in training). Specify it to different values to trade-off utility (```--lora_alpha 0```) and security (```--lora_alpha 8```).
+ This command tests [AlpacaEval2 utility benchmark](https://huggingface.co/datasets/tatsu-lab/alpaca_farm), [AlpacaFarm security benchmark](https://arxiv.org/pdf/2402.06363), [lm_eval utility benchmark](https://github.com/EleutherAI/lm-evaluation-harness): ```meta_mmlu_0shot_instruct```, ```meta_mmlu_pro_instruct``` (5-shot), ```meta_bbh``` (3-shot), ```meta_ifeval```, and ```meta_gpqa_cot```-diamond, [SEP utility/security benchmark](https://arxiv.org/pdf/2403.06833), [TaskTracker security benchmark](https://github.com/microsoft/TaskTracker) , [CyberSecEval2 benchmark](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/), and [InjecAgent security benchmark](https://arxiv.org/pdf/2403.02691)
+ To test AgentDojo, run
> bash agentdojo/src/vllm_serve.sh \
> bash agentdojo/src/run_benchmark_secalign.sh [log_dir] [defense]  \
> bash agentdojo/src/run_benchmark.sh
+ Results will be logged to ```[model_path]/summary.tsv```



# Code Acknowledgements
The majority of SecAlign is licensed under CC-BY-NC, however portions of the project are available under separate license terms: ... This software and/or data was deposited in the BAIR open research Commons repository in 2025. 

Code from other repos: torchtune, TaskTracker (setup.py), lm_eval_harness (lm_eval_config), alpaca_eval (glm_winrate.py)