# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os
import lm_eval
from lm_eval.tasks import TaskManager
import torch
from lm_eval.models.vllm_causallms import VLLM
import nltk
nltk.download('punkt_tab')
from utils import jdump, summary_results, load_vllm_model_with_changed_lora_alpha
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
TASKS = ['meta_gpqa_cot', 'meta_ifeval', 'meta_bbh', 'meta_mmlu_0shot_instruct', 'meta_mmlu_pro_instruct'] # 'ifbench', 

def main(args):
    if os.path.exists(args.model_name_or_path): log_dir = args.model_name_or_path
    else: log_dir = args.model_name_or_path + '-log'; os.makedirs(log_dir, exist_ok=True)

    base_model_path = args.model_name_or_path.split('_')[0]
    if args.model_name_or_path != base_model_path: model_name_or_path_changed_lora_alpha = load_vllm_model_with_changed_lora_alpha(args.model_name_or_path, args.lora_alpha)
    else: model_name_or_path_changed_lora_alpha = None
    lm_obj = VLLM(pretrained=base_model_path, lora_local_path=model_name_or_path_changed_lora_alpha, 
                  enable_lora=args.model_name_or_path != base_model_path, max_lora_rank=128, 
                  dtype="auto", data_parallel_size=args.data_parallel_size, tensor_parallel_size=args.tensor_parallel_size, 
                  max_model_len=8192, add_bos_token=False, batch_size=args.batch_size) # this does not load the vllm model/engine

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = TaskManager(include_path=args.include_path) #lm_eval.tasks.TaskManager(include_path=args.include_path)
    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    
    if os.path.exists(args.model_name_or_path): log_dir = args.model_name_or_path
    else: log_dir = args.model_name_or_path + '-log'; os.makedirs(log_dir, exist_ok=True)
    if 'all' in args.tasks: args.tasks = TASKS
    for task in args.tasks:
        log_file = f"{log_dir}/{task}-loraalpha{args.lora_alpha}-results.json"
        if os.path.exists(log_file) and args.skip_run_tasks:
            print(f"Skipping task {task} as results already exist in {log_file}")
            continue
        time.sleep(10)
        if model_name_or_path_changed_lora_alpha is not None: model_name_or_path_changed_lora_alpha = load_vllm_model_with_changed_lora_alpha(args.model_name_or_path, args.lora_alpha) # in case it is delelted
        results = lm_eval.simple_evaluate( # this loads vllm model/engine every time
            model=lm_obj,
            tasks=[task],
            num_fewshot=0, # CoT examples are already given in the data
            task_manager=task_manager,
            apply_chat_template=False,#args.apply_chat_template
            batch_size=args.batch_size,
        )
        
        if results is not None: 
            if task == 'meta_ifeval':
                results["results"][task]['exact_match,strict-match'] = \
                    (results["results"][task]['prompt_level_strict_acc,none'] + 
                     results["results"][task]['inst_level_strict_acc,none'] +
                     results["results"][task]['prompt_level_loose_acc,none'] +
                     results["results"][task]['inst_level_loose_acc,none']) / 4
            
            summary_results(log_dir + '/summary.tsv', {
                'attack': '-', 
                'ASR/Utility': '%.2f' % (results["results"][task]['exact_match,strict-match'] * 100) + '%', 
                'defense': '-', 
                'instruction_hierarchy': '-',
                'lora_alpha': args.lora_alpha,
                'test_data': task,
            })
            jdump(results, log_file)

    #if model_name_or_path_changed_lora_alpha is not None: os.system('rm -rf ' + model_name_or_path_changed_lora_alpha)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    #parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--include_path", type=str, default='lm_eval_config')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tasks", type=str, default=['all'], nargs='+')
    #parser.add_argument("--add_bos_token", action="store_true", default=False) # False gives better scores, always set to False
    parser.add_argument("--skip_run_tasks", action="store_true", default=False)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--delay_hour", type=float, default=0)
    args = parser.parse_args()
    args.data_parallel_size = int(torch.cuda.device_count() / args.tensor_parallel_size)
    time.sleep(args.delay_hour * 3600)
    main(args)
