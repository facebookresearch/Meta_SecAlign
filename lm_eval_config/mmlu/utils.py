# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
import datasets

def doc_to_text_instruct(doc: dict) -> str:
    return doc["input_final_prompts"][0]

def process_docs_instruct(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
            #"input_final_prompts": [doc["input_final_prompts"][0].replace('<|start_header_id|>', '<|header_start|>').replace('<|end_header_id|>', '<|header_end|>').replace('<|eot_id|>', '<|eot|>')],
            #"input_final_prompts": [doc["input_final_prompts"][0].replace('<|start_header_id|>user<|end_header_id|>', '<|start_header_id|>system<|end_header_id|>')],
            #"input_final_prompts": [doc["input_final_prompts"][0].replace('\nQuestion: ', '<|eot_id|>\n<|start_header_id|>[INPT]<|end_header_id|>\n\n')],
        }
        return out_doc
    dataset = dataset.select_columns(["input_question", "input_correct_responses", "input_final_prompts", "is_correct","input_question_hash","input_choice_list","output_prediction_text"])
    dataset = dataset.rename_column("is_correct","previously_is_correct")
    dataset = dataset.map(_process_doc)
    return dataset.map(_process_doc)

def doc_to_text(doc: dict) -> str:
    # Strip out the last two characters, which is a space and the answer
    # E.g., "Answer: B" -> "Answer:"
    return doc["input_final_prompts"][0][:-2]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        # input_correct_responses is in format of: "Answer: B"
        answer = doc["input_correct_responses"][0]
        # Indexes are always A: 0, B: 1, C: 2, D: 3
        answer_index = string.ascii_uppercase.index(answer[-1])

        out_doc = {
            "problem": doc["input_question"],
            # The answer is the index of the correct response (0-indexed)
            "gold": answer_index,
        }
        return out_doc

    dataset = dataset.select_columns(
        ["input_question", "input_correct_responses", "input_final_prompts", "is_correct", "input_question_hash",
         "input_choice_list"])
    dataset = dataset.rename_column("is_correct", "previously_is_correct")
    dataset = dataset.map(_process_doc)
    return dataset.map(_process_doc)

def doc_to_target(doc: dict) -> str:
    return doc["gold"]
