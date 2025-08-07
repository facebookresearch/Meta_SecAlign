#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------- SBATCH commands ---------- #
#SBATCH --job-name=torchtune-multi-node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --partition=learn
#SBATCH --qos=memorization_high
#SBATCH --time=24:00:00

# ---------- Set env variables ---------- #
# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# You might need to explicitly set the network interface for distributed backends:
# export NCCL_SOCKET_IFNAME=...
# export GLOO_SOCKET_IFNAME=...

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO

# ---------- Launch training ---------- #
# You probably want to load in a virtual env w/ conda...
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate llm_llama4
# module load conda
# conda activate torchtune
# ...or venv
# source torchtune/bin/activate

SHARED_FS=/checkpoint/meta-llama # <----------- Replace w/ your filesystem
OUTPUT_DIR="$SHARED_FS/Llama-4-Scout-17B-16E-Instruct-lr12e-6-multinode-output" # <---- Replace w/ desired checkpoint output directory

# Adjust sbatch --ntasks and sbatch --nodes above and --nnodes below to your specific node count
srun tune run --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    full_dpo_distributed --config llama4_scout_full.yaml output_dir=$OUTPUT_DIR
