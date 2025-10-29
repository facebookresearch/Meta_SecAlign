# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import evaluation_lib
from evaluation_lib import InputExample, OutputExample

def extract_answer(text):
    """Extract text between <answer> and </answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def process_results(doc, results):
  
  inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
  
  response = extract_answer(results[0])

  out_strict = evaluation_lib.test_instruction_following_strict(inp, response)
  out_loose = evaluation_lib.test_instruction_following_loose(inp, response)
  
  return {
    "prompt_level_strict_acc": out_strict.follow_all_instructions,
    "inst_level_strict_acc": out_strict.follow_instruction_list,
    "prompt_level_loose_acc": out_loose.follow_all_instructions,
    "inst_level_loose_acc": out_loose.follow_instruction_list,
  }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc