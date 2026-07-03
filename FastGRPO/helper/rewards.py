"""Reward functions for GRPO training."""

import asyncio
import ast
import json
import math
import re
from typing import Dict

from helper.code_verifier import extract_python_code, verify_code_completion

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
except ModuleNotFoundError:
    NormalizationConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None


def accuracy_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
        raise ModuleNotFoundError(
            "math_latex rewards require latex2sympy2_extended and math_verify. "
            "Install the FastGRPO requirements or use a non-math reward type."
        )
    rewards = []
    for content, sol in zip(completions, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(
                    f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = 0.0
        else:
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    return [count_tags(c) for c in completions]


def _normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match_reward_func(completions, solution, **kwargs):
    """Reward 1.0 when normalized completion exactly matches the solution."""
    rewards = []
    for content, sol in zip(completions, solution):
        rewards.append(float(_normalize_text(content) == _normalize_text(sol)))
    return rewards


def contains_reward_func(completions, solution, **kwargs):
    """Reward 1.0 when the normalized solution appears in the completion."""
    rewards = []
    for content, sol in zip(completions, solution):
        normalized_content = _normalize_text(content)
        normalized_solution = _normalize_text(sol)
        rewards.append(float(bool(normalized_solution) and normalized_solution in normalized_content))
    return rewards


def regex_reward_func(completions, pattern, **kwargs):
    """Reward 1.0 when completion matches the configured regex pattern."""
    rewards = []
    for content, cur_pattern in zip(completions, pattern):
        if cur_pattern is None:
            rewards.append(0.0)
            continue
        rewards.append(float(re.search(str(cur_pattern), str(content), flags=re.DOTALL) is not None))
    return rewards


def _extract_code_text(completion):
    """Extract the first fenced code block when present; otherwise return text."""
    if completion is None:
        return ""
    text = str(completion).strip()
    fenced = re.search(r"```(?:[a-zA-Z0-9_+\-.#]*)\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _get_code_language(example):
    metadata = example.get("metadata") or {}
    language = example.get("language") or metadata.get("language") or "python"
    return str(language).lower()


def _get_entry_point(example):
    metadata = example.get("metadata") or {}
    return example.get("entry_point") or metadata.get("entry_point") or metadata.get("function_name")


def _get_expected_substrings(example):
    metadata = example.get("metadata") or {}
    expected = example.get("expected_substrings") or metadata.get("expected_substrings") or []
    if isinstance(expected, str):
        expected = [expected]
    return [str(item) for item in expected if item is not None]


def code_placeholder_reward_func(completion, example):
    """Heuristic code reward used until sandboxed unit-test execution is plugged in.

    This intentionally does not execute generated code. For production code-RLVR,
    provide ``custom_reward_func`` in the task config and run tests in a sandbox.
    The placeholder rewards non-empty extracted code, Python syntax validity when
    applicable, an optional ``entry_point`` occurrence, and optional expected
    substrings.
    """
    code = _extract_code_text(completion)
    if not code:
        return 0.0

    language = _get_code_language(example)
    entry_point = _get_entry_point(example)
    expected_substrings = _get_expected_substrings(example)

    score = float(example.get("non_empty_weight", 0.2))
    remaining = max(1.0 - score, 0.0)

    syntax_weight = float(example.get("syntax_weight", 0.4))
    if language in ("py", "python", "python3"):
        try:
            ast.parse(code)
            score += syntax_weight
        except SyntaxError:
            pass
    else:
        # Non-Python syntax checking is intentionally left to custom evaluators.
        score += min(syntax_weight, remaining)

    if entry_point:
        entry_weight = float(example.get("entry_point_weight", 0.2))
        if re.search(rf"\b{re.escape(str(entry_point))}\b", code):
            score += entry_weight

    if expected_substrings:
        substring_weight = float(example.get("substring_weight", 0.2))
        matched = sum(1 for expected in expected_substrings if expected in code)
        score += substring_weight * matched / len(expected_substrings)

    return float(max(0.0, min(score, 1.0)))


def _get_tests(example):
    metadata = example.get("metadata") or {}
    for key in ("tests", "test", "unit_tests", "test_cases"):
        if key in example and example[key] is not None:
            return example[key]
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _get_test_type(example):
    metadata = example.get("metadata") or {}
    return (
        example.get("test_type")
        or example.get("code_test_type")
        or metadata.get("test_type")
        or metadata.get("code_test_type")
        or "unit"
    )


def _get_timeout_seconds(example):
    metadata = example.get("metadata") or {}
    return (
        example.get("timeout_seconds")
        or example.get("code_timeout_seconds")
        or metadata.get("timeout_seconds")
        or metadata.get("code_timeout_seconds")
        or 5.0
    )


def _has_tests(tests):
    if tests is None:
        return False
    if isinstance(tests, str):
        return bool(tests.strip())
    try:
        return len(tests) > 0
    except TypeError:
        return True


def code_unit_test_reward_func(completion, example):
    """Reward 1.0 only when generated Python passes configured tests."""
    return code_unit_test_reward_details(completion, example)["reward"]


def code_unit_test_reward_details(completion, example):
    """Return unit-test reward and verifier diagnostics for one completion."""
    language = _get_code_language(example)
    tests = _get_tests(example)
    entry_point = _get_entry_point(example)
    test_type = _get_test_type(example)
    extracted_code = extract_python_code(completion, entry_point=entry_point)
    detail = {
        "reward": 0.0,
        "reward_type": "code_unit_test",
        "language": language,
        "test_type": str(test_type),
        "has_tests": _has_tests(tests),
        "has_entry_point": bool(entry_point),
        "completion_chars": len(str(completion or "")),
        "extracted_code_chars": len(extracted_code),
        "passed": False,
        "timed_out": False,
        "error_type": "none",
    }
    if language not in ("py", "python", "python3"):
        detail["error_type"] = "unsupported_language"
        return detail

    result = verify_code_completion(
        completion,
        tests,
        entry_point=entry_point,
        test_type=test_type,
        timeout_seconds=_get_timeout_seconds(example),
        starter_code=example.get("starter_code") or (example.get("metadata") or {}).get("starter_code"),
    )
    detail.update({
        "reward": 1.0 if result.passed else 0.0,
        "passed": bool(result.passed),
        "timed_out": bool(result.timed_out),
        "error_type": result.error_type,
        "stdout_chars": len(result.stdout or ""),
        "stderr_chars": len(result.stderr or ""),
    })
    return detail


def _get_solution(example):
    for key in ("answer", "solution", "ground_truth", "label"):
        if key in example and example[key] is not None:
            return example[key]
    return None


def compute_reward_from_example(completion, example):
    """Dispatch a single completion to the reward configured on its task/example."""
    reward_type = example.get("reward_type", "math_latex")
    solution = _get_solution(example)

    if reward_type in ("math", "math_latex", "latex_accuracy", "accuracy"):
        if solution is None:
            return 0.0
        format_weight = float(example.get("format_weight", 0.2))
        answer_reward = accuracy_reward_func([completion], [solution])[0]
        if format_weight == 0:
            return float(answer_reward)
        format_reward = format_reward_func([completion])[0]
        return float(format_weight * format_reward + answer_reward)

    if reward_type in ("exact", "exact_match"):
        if solution is None:
            return 0.0
        return float(exact_match_reward_func([completion], [solution])[0])

    if reward_type in ("contains", "substring"):
        if solution is None:
            return 0.0
        return float(contains_reward_func([completion], [solution])[0])

    if reward_type == "regex":
        pattern = example.get("pattern")
        if pattern is None:
            metadata = example.get("metadata") or {}
            pattern = metadata.get("pattern")
        return float(regex_reward_func([completion], [pattern])[0])

    if reward_type in ("format", "format_only"):
        return float(format_reward_func([completion])[0])

    if reward_type in ("code", "coding", "code_placeholder", "code_syntax"):
        return code_placeholder_reward_func(completion, example)

    if reward_type in ("code_unit_test", "code_tests", "unit_test", "python_unit_test"):
        return code_unit_test_reward_func(completion, example)

    if reward_type in ("none", "zero"):
        return 0.0

    raise ValueError(
        f"Unsupported reward_type={reward_type!r}. "
        "Use math_latex, exact_match, contains, regex, format_only, code, "
        "code_unit_test, zero, "
        "or provide a custom_reward_func through the multi-task config."
    )


def compute_reward_debug_from_example(completion, example):
    """Compute reward and return lightweight diagnostics for logging/debugging."""
    reward_type = example.get("reward_type", "math_latex")
    solution = _get_solution(example)

    if reward_type in ("math", "math_latex", "latex_accuracy", "accuracy"):
        return _math_latex_reward_details(completion, solution, example)

    if reward_type in ("exact", "exact_match"):
        reward = 0.0 if solution is None else float(exact_match_reward_func([completion], [solution])[0])
        return _basic_reward_detail(reward, reward_type, "missing_solution" if solution is None else "none")

    if reward_type in ("contains", "substring"):
        reward = 0.0 if solution is None else float(contains_reward_func([completion], [solution])[0])
        return _basic_reward_detail(reward, reward_type, "missing_solution" if solution is None else "none")

    if reward_type == "regex":
        pattern = example.get("pattern")
        if pattern is None:
            pattern = (example.get("metadata") or {}).get("pattern")
        reward = float(regex_reward_func([completion], [pattern])[0])
        return _basic_reward_detail(reward, reward_type, "missing_pattern" if pattern is None else "none")

    if reward_type in ("format", "format_only"):
        reward = float(format_reward_func([completion])[0])
        return _basic_reward_detail(reward, reward_type, "format_failed" if reward == 0 else "none")

    if reward_type in ("code", "coding", "code_placeholder", "code_syntax"):
        reward = float(code_placeholder_reward_func(completion, example))
        error_type = "none" if reward >= 1.0 else "partial_or_failed_static_checks"
        return _basic_reward_detail(reward, reward_type, error_type)

    if reward_type in ("code_unit_test", "code_tests", "unit_test", "python_unit_test"):
        return code_unit_test_reward_details(completion, example)

    if reward_type in ("none", "zero"):
        return _basic_reward_detail(0.0, reward_type, "zero_reward")

    raise ValueError(
        f"Unsupported reward_type={reward_type!r}. "
        "Use math_latex, exact_match, contains, regex, format_only, code, "
        "code_unit_test, zero, "
        "or provide a custom_reward_func through the multi-task config."
    )


def _basic_reward_detail(reward, reward_type, error_type):
    reward = float(reward)
    if error_type == "none" and reward <= 0:
        error_type = "incorrect"
    return {
        "reward": reward,
        "reward_type": str(reward_type),
        "passed": reward > 0,
        "error_type": error_type,
    }


def _math_latex_reward_details(completion, solution, example):
    reward_type = example.get("reward_type", "math_latex")
    if solution is None:
        return _basic_reward_detail(0.0, reward_type, "missing_solution")
    if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
        raise ModuleNotFoundError(
            "math_latex rewards require latex2sympy2_extended and math_verify. "
            "Install the FastGRPO requirements or use a non-math reward type."
        )

    format_weight = float(example.get("format_weight", 0.2))
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) == 0:
        reward = 1.0
        return {
            "reward": reward,
            "reward_type": str(reward_type),
            "passed": True,
            "error_type": "gold_parse_failed",
            "gold_parse_failed": True,
            "answer_parse_failed": False,
            "format_reward": float(format_reward_func([completion])[0]) if format_weight else 0.0,
        }

    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    try:
        answer_reward = float(verify(answer_parsed, gold_parsed))
        verify_error = None
    except Exception as exc:
        answer_reward = 0.0
        verify_error = str(type(exc).__name__)

    format_reward = float(format_reward_func([completion])[0]) if format_weight else 0.0
    reward = answer_reward if format_weight == 0 else format_weight * format_reward + answer_reward
    if verify_error:
        error_type = "verify_exception"
    elif len(answer_parsed) == 0:
        error_type = "answer_parse_failed"
    elif answer_reward <= 0:
        error_type = "wrong_answer"
    else:
        error_type = "none"
    return {
        "reward": float(reward),
        "reward_type": str(reward_type),
        "passed": answer_reward > 0,
        "error_type": error_type,
        "gold_parse_failed": False,
        "answer_parse_failed": len(answer_parsed) == 0,
        "answer_reward": answer_reward,
        "format_reward": format_reward,
        "verify_error": verify_error or "",
    }
