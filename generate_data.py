# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils import generate_preference_dataset
import argparse

parser = argparse.ArgumentParser(prog='Generating preference dataset for SecAlign training')
parser.add_argument('-m', '--model_name_or_path', type=str)
parser.add_argument('--dataset', type=str, default="alpaca", choices=["alpaca", "natural"])
parser.add_argument('--self_generated_response', default=False, action='store_true')
parser.add_argument('--random_inject_pos', default=False, action='store_true')
args = parser.parse_args()

# Generate preference dataset
generate_preference_dataset(args.model_name_or_path, instruct_dataset=args.dataset,
                            self_generated_response=args.self_generated_response,
                            random_inject_pos=args.random_inject_pos)
