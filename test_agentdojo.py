# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
#mport subprocess, signal
from datetime import datetime
import time
from utils import test_parser, load_vllm_model_with_changed_lora_alpha

args = test_parser()
if args.defense != 'repeat_user_prompt' or 'important_instructions' not in args.attack:
    print('Warning: not using attack=important_instructions with defense=repeat_user_prompt as in Meta-SecAlign paper.')

for model_name_or_path in args.model_name_or_path:
    base_model_path = model_name_or_path.split('_')[0]
    cmd = 'vllm serve %s --dtype auto --host 0.0.0.0 --tensor-parallel-size 4 --max-model-len 24576' % base_model_path # --max-model-len 16384
    if '_' in model_name_or_path: # Evaluating defensive-fine-tuned LoRA model
        model_name_or_path = load_vllm_model_with_changed_lora_alpha(model_name_or_path, args.lora_alpha)
        cmd += ' --enable-lora --max-lora-rank 64 --lora-modules %s=%s' % (model_name_or_path, model_name_or_path)
    
    log_dir = 'agentdojo/runs/' + model_name_or_path
    os.makedirs(log_dir, exist_ok=True)
    server_log = 'agentdojo/runs/' + model_name_or_path + '/vllm_server_%s.out' % datetime.now().strftime('%Y%m%d_%H%M%S')
    os.system('nohup ' + cmd + ' > ' + server_log + ' 2>&1 &')
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
    if args.defense != 'none': cmd += ' --defense %s' % args.defense
    try:
        for attack in args.attack: 
            os.system(cmd + ' --attack %s' % attack)
    except:
        break
    os.system('kill -9 %s' % ' '.join(set(pids)))
    #os.kill(process.pid, signal.SIGKILL)