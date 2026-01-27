# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from utils import generate_preference_dataset

def add_8B_default_model_arguments(parser):
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--lr", type=float, default=1.6e-4, help="Learning rate")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        help="Cache directory for model weights"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="helpers/llama3.1_8B_lora.yaml",
        help="Path to the DPO config file"
    )
    return parser

def add_70B_default_model_arguments(parser):
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    parser.add_argument("--lr", type=float, default=3.2e-4, help="Learning rate")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        help="Cache directory for model weights"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="helpers/llama3.3_70B_lora.yaml",
        help="Path to the DPO config file"
    )
    return parser

def parse_args(add_model_arguments):
    parser = argparse.ArgumentParser(description="Train Llama model with DPO using specified hyperparameters")
    parser.add_argument(
        "--no_self_generated_response",
        action='store_false',
        default=True,
        dest="self_generated_response",
        help="Use self-generated responses for data generation (default: True)"
    )
    parser.add_argument(
        "--no_randomized_injection_position",
        action='store_false',
        default=True,
        dest="randomized_injection_position",
        help="Use randomized injection position for data generation (default: True)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        help="Dataset name (default: alpaca)"
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="Number of processes per node for distributed training (default: 8)"
    )
    parser = add_model_arguments(parser)

    args = parser.parse_args()
    args.cache_dir = os.path.expanduser(args.cache_dir)
    args.output_dir = f"{args.model}_dpo_NaiveCompletion_{'randpos_' if args.randomized_injection_position else ''}{'synthetic' if args.self_generated_response else ''}_{args.dataset}_lr{args.lr}"
    args.data_files = f"data/preference_{args.model.split('/')[-1]}_dpo_NaiveCompletion_{'randpos_' if args.randomized_injection_position else ''}{'synthetic' if args.self_generated_response else ''}_{args.dataset}.json"
    return args


if __name__ == "__main__":
    args = parse_args(add_8B_default_model_arguments)
    #args = parse_args(add_70B_default_model_arguments) # Uncomment this line to use 70B model defaults

    generate_preference_dataset(
        preference_data_path=args.data_files, 
        instruct_dataset=args.dataset,
        self_generated_response=args.self_generated_response,
        randomized_injection_position=args.randomized_injection_position,
        model_name_or_path=args.model
    )

    cmd = f'tune run --nproc_per_node {args.nproc_per_node} lora_dpo_distributed \
        --config {args.config_file} \
        output_dir={args.output_dir} \
        dataset.data_files={args.data_files} \
        optimizer.lr={args.lr} \
        cache_dir={args.cache_dir}'
    print(cmd)
    os.system(cmd)