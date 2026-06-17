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


from pprint import pprint
import random
import time
import traceback
import os
import re
from tracemalloc import start
try:
    from math_verify import parse
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.utils import timeout
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify.grader import verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import math

ANSWER_MATCH_BIT = 1
REASONING_ORDER_BIT = 2
def verify_format(model_output: str, prompt_id: int):
    """
    Verify if the answer is in a valid format.
    """
    result = 0
    
    if prompt_id in [0, 1, 2, 3]:
        answer_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    else:
        raise NotImplementedError(f"Prompt ID {prompt_id} is not supported for answer extraction in math verify.")
    if len(answer_matches) == 1:
        result |= ANSWER_MATCH_BIT
    
    think_blocks = [i for i in re.findall(r'<think>.*?</think>', model_output, re.DOTALL)]
    
    concatenated = ''.join(think_blocks)
    
    text_no_whitespace = re.sub(r'\s+', '', model_output)
    concatenated_no_whitespace = re.sub(r'\s+', '', concatenated)
    
    if text_no_whitespace == concatenated_no_whitespace:
        result |= REASONING_ORDER_BIT

    num_steps = len(think_blocks)
    return result, num_steps


def extract_answer(model_output: str, prompt_id: int, keep_box=False) -> str:
    extraction = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    if len(extraction) == 0:
        if keep_box:
            extraction = re.findall(r'(\\boxed{.*})', model_output, re.DOTALL)
        else:
            extraction = re.findall(r'\\boxed{(.*)}', model_output, re.DOTALL)
        if len(extraction) == 0:
            if random.random() < 0.01:
                pprint(f"Warning: No answer extracted from the model output.\n\n======{model_output}")
            return "None extraction"

    return extraction[-1].strip() # use the last extracted answer

def compute_score(data_source, solution_str, ground_truth, extra_info=None, is_valid=False, prompt_id=None) -> bool:
    assert prompt_id is not None, "prompt_id must be provided for math_verify"

    try:
        verify_format_w_timeout = timeout(2)(verify_format)
        format_correctness, num_steps = verify_format_w_timeout(solution_str, prompt_id)
    except TimeoutException:
        print("Timeout detected in format verification, returning 0 score.")
        format_correctness, num_steps = 0, 0
    except Exception:
        print("Error detected in format verification, returning 0 score.")
        format_correctness, num_steps = 0, 0

    try:

        # during training
        if not is_valid:
            if data_source in ["dapomath", "oldaime"]:
                extracted_predictions = extract_answer(solution_str, prompt_id) # only verify the answer part wrapped in <answer>...</answer>
                gold_extraction_target=(ExprExtractionConfig(),) # reduce computation time for training, since DAPOmath only requires ExprExtractionConfig
            else:
                raise NotImplementedError(f"Data source {data_source} is not supported for answer extraction during training in math verify.")
        # during validation
        else:
            if data_source in ['amc12', 'aime24']:
                extracted_predictions = solution_str
                gold_extraction_target = (ExprExtractionConfig(),) # golden_truth is always number
            else:
                raise NotImplementedError(f"Data source {data_source} is not supported for answer extraction during inference in math verify.")
        pred_extraction_target=(
            ExprExtractionConfig(), 
            LatexExtractionConfig(),
        )

        extracted_predictions = parse(extracted_predictions, pred_extraction_target, parsing_timeout=3)
        extracted_golds = parse(ground_truth, gold_extraction_target, parsing_timeout=3)
        
        if random.random() < 0.01:
            print(f"====== [Random Sample] Verify {len(extracted_golds)} golds and {len(extracted_predictions)} predictions ======")
        ret_score = verify(extracted_golds, extracted_predictions, timeout_seconds=3)

        if len(extracted_predictions) == 0:
            extracted_predictions = "N/A extraction"
        elif len(extracted_predictions) == 1:
            extracted_predictions = f"{extracted_predictions[0]}"
        else:
            extracted_predictions = extracted_predictions[1] if isinstance(extracted_predictions[1], str) else f"{extracted_predictions[0]}"

    except Exception:
        ret_score = 0.
        extracted_predictions = "Error extraction"
        traceback.print_exc()
        print("Error detected in math_verify, returning 0 score.")



    return {
        "score": ret_score,
        "acc": 1 if ret_score > 0 else 0,
        "format": format_correctness,
        "pred": extracted_predictions,
        "#steps": num_steps,
    }
