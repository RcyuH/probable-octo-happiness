# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import test

from math_verify import parse
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.utils import timeout
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import datasets
from numpy import argsort

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
import re


def extract_solution(solution_str):
    ground_truth_boxed = "\\boxed{" + solution_str + "}"
    gold_extraction_target=(LatexExtractionConfig(),)
    extracted_golds = parse(ground_truth_boxed, gold_extraction_target)
    return False if len(extracted_golds) == 0 else True

def format_question_to_prompt(question):
    # system_prompt = all_prompts[prompt_id]  # default system prompt
    user_prompt = question

    return [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "<think>\n"}
    ]

def process_amc_dataset():
    data_source = "AI-MO/aimo-validation-amc"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    if RESUME and os.path.exists(test_path):
        return datasets.load_dataset("parquet", data_files=test_path)["train"]

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            answer = example.pop("answer")
            data = {
                "data_source": "amc12",
                "prompt": format_question_to_prompt(question),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    test_dataset.to_parquet(test_path)
    print("Size of AMC-12 test dataset:", len(test_dataset))

    return test_dataset

def process_aime24_dataset():
    data_source = "Maxwell-Jia/AIME_2024"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    if RESUME and os.path.exists(test_path):
        return datasets.load_dataset("parquet", data_files=test_path)["train"]

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("Problem")
            answer = example.pop("Answer")
            data = {
                "data_source": "aime24",
                "prompt": format_question_to_prompt(question),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    test_dataset.to_parquet(test_path)
    print("Size of AIME24 test dataset:", len(test_dataset))

    return test_dataset



def process_oldaime_dataset():
    data_source = "old_aime"
    local_dir = "OLDAIME"
    train_path = os.path.join(MY_DATA_DIR, local_dir, "train.parquet")
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")
    if RESUME and os.path.exists(train_path):
        return

    golden_extraction_target=(ExprExtractionConfig(),)
    if os.path.exists(filtered_dataset_path):
        print(f"Loading the filtered dataset from {filtered_dataset_path}...", flush=True)
        dataset = datasets.load_dataset("parquet", data_files=filtered_dataset_path)
    else:
        print(f"Loading the {data_source} dataset from kaggle...", flush=True)
        dataset = datasets.load_dataset("csv", data_files="OLD_AIME.csv")["train"]
        def filter_fn(example):
            if not example.get("Answer"):
                return False
            try:
                float(example["Answer"])
                is_number = True
            except (ValueError, TypeError):
                is_number = False
                print("Answer is not a number:", example["Answer"])
            
            golden_answer = example["Answer"]
            extracted = parse(golden_answer, golden_extraction_target, parsing_timeout=5)
            return len(extracted) > 0
        dataset = dataset.filter(lambda x: x["Year"] < 2024)
        dataset = dataset.filter(filter_fn)
        if not os.path.exists(os.path.dirname(filtered_dataset_path)):
            makedirs(os.path.dirname(filtered_dataset_path))
        dataset.to_parquet(filtered_dataset_path)
    print("Size of OLD-AIME dataset after filtering:", len(dataset))

    train_dataset = dataset
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["Question"]
            answer = example["Answer"]
            example = {
                "data_source": "oldaime",
                "prompt": format_question_to_prompt(question),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
            return example
        return process_fn
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(train_path)
    print("Size of OLD-AIME train dataset:", len(train_dataset))



def process_dapomath_dataset():
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    train_path = os.path.join(MY_DATA_DIR, local_dir, "train.parquet")
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")
    if RESUME and os.path.exists(test_path):
        return

    golden_extraction_target=(ExprExtractionConfig(),)
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")
    if False:
        print(f"Loading the filtered dataset from {filtered_dataset_path}...", flush=True)
        dataset = datasets.load_dataset("parquet", data_files=filtered_dataset_path)
    else:
        print(f"Loading the {data_source} dataset from huggingface...", flush=True)
        dataset = datasets.load_dataset("YouJiacheng/DAPO-Math-17k-dedup", trust_remote_code=True, split="train") # deduplication
        def filter_fn(example):
            if not example['reward_model']['ground_truth']:
                return False
            try:
                # Try to convert answer to a number
                float(example['reward_model']['ground_truth'])
                is_number = True
            except (ValueError, TypeError):
                is_number = False
                print("Answer is not a number:", example['reward_model']['ground_truth'])
            
            # golden_answer = '\\boxed{' + str(example['reward_model']['ground_truth']) + "}"
            golden_answer = example['reward_model']['ground_truth']
            extracted_golds = parse(golden_answer, golden_extraction_target, parsing_timeout=5)
            return len(extracted_golds) > 0
        dataset = dataset.filter(filter_fn)
        if not os.path.exists(os.path.dirname(filtered_dataset_path)):
            makedirs(os.path.dirname(filtered_dataset_path))
        dataset.to_parquet(filtered_dataset_path)
    print("Size of DAPO-Math dataset after filtering:", len(dataset))

    _ = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset, test_dataset = _["train"], _["test"]
    test_dataset = test_dataset.shuffle(42).select(range(100))  # only select the first 100 samples for testing

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")[-1]["content"]
            prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."
            question = question[len(prefix):-len(suffix)]  # reformat the original prompt by removing prefix and suffix
            example['data_source'] = "dapomath"
            example['prompt'] = format_question_to_prompt(question)
            example['reward_model']['ground_truth'] = str(example['reward_model']['ground_truth'])
            example['extra_info'] = {"split": split, "index": idx}
            return example

        return process_fn
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset.to_parquet(test_path)
    train_dataset.to_parquet(train_path)
    print("Size of DAPO-Math train dataset:", len(train_dataset))
    print("Size of DAPO-Math test dataset:", len(test_dataset))


if __name__ == "__main__":
    argsort = argparse.ArgumentParser()
    argsort.add_argument("--resume", action="store_true")   
    RESUME = argsort.parse_args().resume
    MY_DATA_DIR = os.getenv("MY_DATA_DIR")

    aime_dataset = process_aime24_dataset()
    amc_dataset = process_amc_dataset()

    # Repeat datasets
    aime_repeated = datasets.concatenate_datasets([aime_dataset] * 32)
    amc_repeated = datasets.concatenate_datasets([amc_dataset] * 16)
    merged_dataset = datasets.concatenate_datasets([aime_repeated, amc_repeated])

    # Save the merged dataset
    merged_path = os.path.join(MY_DATA_DIR, "merged_math_datasets", "merged_test.parquet")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    merged_dataset.to_parquet(merged_path)

    print(f"Merged dataset saved to {merged_path}")
    print(f"Size of merged dataset: {len(merged_dataset)}")

    """Train dataset"""
    process_dapomath_dataset()
    process_oldaime_dataset()

    print("Done Preprocessing!")