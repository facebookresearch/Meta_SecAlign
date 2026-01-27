# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import yaml
from datetime import datetime
import time
import subprocess
from utils import test_parser, load_vllm_model_with_changed_lora_alpha

args = test_parser()
if args.defense != 'repeat_user_prompt' or ('important_instructions' not in args.attack and 'none' not in args.attack):
    print('Warning: not using attack=important_instructions/none with defense=repeat_user_prompt as in Meta-SecAlign paper.')

def print_while_saving(cmd, log_dir):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    with open(os.path.join(log_dir, 'summary.log'), 'w') as f:
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
    proc.wait()

for model_name_or_path in args.model_name_or_path:
    if 'gpt' not in model_name_or_path and 'gemini' not in model_name_or_path:
        base_model_path = model_name_or_path.split('_')[0]
        cmd = 'vllm serve %s --dtype auto --host 0.0.0.0 --tensor-parallel-size %d --max-model-len 24576' % (base_model_path, args.tensor_parallel_size) # --max-model-len 16384
        if '_' in model_name_or_path: # Evaluating defensive-fine-tuned LoRA model
            model_name_or_path = load_vllm_model_with_changed_lora_alpha(model_name_or_path, args.lora_alpha)
            cmd += ' --enable-lora --max-lora-rank 64 --lora-modules %s=%s' % (model_name_or_path, model_name_or_path)
        log_dir = os.getcwd() + '/' + model_name_or_path
        if not os.path.exists(log_dir): log_dir += '-log'
        log_dir += '/agentdojo'
        os.makedirs(log_dir, exist_ok=True)
        server_log = log_dir + '/vllm_server_%s.out' % datetime.now().strftime('%Y%m%d_%H%M%S')
        cmd = 'nohup ' + cmd + ' > ' + server_log + ' 2>&1 &'
        #print(cmd)
        os.system(cmd)
        #with open(server_log, 'a') as out: process = subprocess.Popen(cmd.split(' '), stdout=out, stderr=out, preexec_fn=os.setpgrp, shell=False)
        time.sleep(30)

        while True:
            with open(server_log, 'r') as f:
                txt = f.read()
                pids = re.findall(r'(?<=pid=).*?(?=\))', txt)
                if 'Application startup complete' in txt: break
            print('Waiting another 30s for vLLM server to start...')
            time.sleep(30)
        print('Evaluating AgentDojo on', model_name_or_path, 'with attacks', args.attack, 'and defense', args.defense, end='\n\n\n')
        cmd = "cd agentdojo/src\npython -m agentdojo.scripts.benchmark --model local --logdir %s --model-id %s --tool-delimiter input" % (log_dir, model_name_or_path) 
    else:
        log_dir = os.getcwd() + '/' + model_name_or_path + '-log/agentdojo'
        os.makedirs(log_dir, exist_ok=True)
        cmd = "cd agentdojo/src\npython -m agentdojo.scripts.benchmark --model %s --logdir %s" % (model_name_or_path, log_dir)
        if 'gpt-5' in model_name_or_path: cmd += ' --reasoning-effort %s' % args.gpt5_reasoning_effort
        pids = []
        if 'gpt' in model_name_or_path:
            with open(args.openai_config_path, 'r') as file: config = yaml.safe_load(file)['default']
            usable_keys = []
            for item in config:
                if item.get('azure_deployment', model_name_or_path) == model_name_or_path:
                    if 'azure_deployment' in item: del item['azure_deployment']
                    #print('Found usable key', len(usable_keys), ':', item)
                    usable_keys.append(item)
            if len(usable_keys) == 0:
                print('No usable OpenAI or Azure key found for model', model_name_or_path)
                exit()
            else:
                os.environ['AZURE_API_KEY'] = usable_keys[0]['api_key']
                os.environ['AZURE_API_ENDPOINT'] = usable_keys[0]['azure_endpoint']
                os.environ['AZURE_API_VERSION'] = usable_keys[0]['api_version']
        elif 'gemini' in model_name_or_path:
            with open(args.gemini_config_path, 'r') as file: config = yaml.safe_load(file)['default']
            #usable_key = {'api_key': config[1]['api_key']}
            os.environ['GOOGLE_API_KEY'] = config[1]['api_key']


    if args.defense != 'none': cmd += ' --defense %s' % args.defense
    try:
        for attack in args.attack: 
            if attack == 'none': 
                #os.system(cmd)
                print_while_saving(cmd, log_dir)
            else: 
                #os.system(cmd + ' --attack %s' % attack)
                print_while_saving(cmd + ' --attack %s' % attack, log_dir)
    except Exception as e:
        print('Error occurred:', e)
        break
    if len(pids): os.system('kill -9 %s' % ' '.join(set(pids)))
    