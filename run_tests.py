# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

parser = argparse.ArgumentParser(prog='')
parser.add_argument('-m', '--model_name_or_path', type=str, nargs='+')
parser.add_argument('--lora_alpha', type=float, default=[8.0], nargs='+')
parser.add_argument('--norun', action='store_false')
args = parser.parse_args()
args.run = args.norun

cmds = [
    'python test.py --attack none straightforward straightforward_before ignore ignore_before completion completion_ignore completion_llama33_70B completion_ignore_llama33_70B --defense none --test_data data/davinci_003_outputs.json --lora_alpha {lora_alpha} -m {model_name_or_path}',
    #'python test.py --attack straightforward --defense none --test_data data/CySE_prompt_injections.json --lora_alpha {lora_alpha} -m {model_name_or_path}',
    'python test.py --attack none straightforward straightforward_before ignore ignore_before completion completion_ignore completion_llama33_70B completion_ignore_llama33_70B --defense none --test_data data/SEP_dataset_test.json --lora_alpha {lora_alpha} -m {model_name_or_path}',
    'python test_lm_eval.py --lora_alpha {lora_alpha} -m {model_name_or_path}',
    'python test_agentdojo.py -a none -d repeat_user_prompt -m {model_name_or_path} --lora_alpha {lora_alpha}',
    'python test_agentdojo.py -a important_instructions -d repeat_user_prompt -m {model_name_or_path} --lora_alpha {lora_alpha}',
    'python test_injecagent.py --defense sandwich --lora_alpha {lora_alpha} -m {model_name_or_path}',
    #'python test_injecagent.py --defense none --lora_alpha {lora_alpha} -m {model_name_or_path}',
    'python test.py --attack straightforward --defense none --test_data data/TaskTracker_dataset_test.json --lora_alpha {lora_alpha} -m {model_name_or_path}',
]


actual_cmds = []
for model_name_or_path in args.model_name_or_path:
    for lora_alpha in args.lora_alpha:
        for cmd in cmds:
            if 'gpt' in model_name_or_path or 'gemini' in model_name_or_path:
                if 'test_lm_eval.py' in cmd: continue # test_lm_eval.py does not support gpt/gemini models
                lora_alpha = -1
            if 'Llama-3.1-8B' in model_name_or_path:
                if 'test_agentdojo.py' in cmd: continue # test_agentdojo.py does not support Llama-3.1-8B models
            cmd = cmd.format(model_name_or_path=model_name_or_path, lora_alpha=lora_alpha)
            if '70B' in model_name_or_path: cmd += ' --tensor_parallel_size 4' # 70B models and Llama4 need at least tensor_parallel_size=8 for 80G GPUs
            actual_cmds.append(cmd)

for cmd in actual_cmds: print(cmd + '\n')
if not args.run: print('Run flag is set to False. Exiting without executing commands.'); exit()
for cmd in actual_cmds:
    print('\nExecuting...\n' + cmd + '\n')
    os.system(cmd)