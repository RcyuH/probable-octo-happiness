"""Reward functions for GRPO training."""

import asyncio
import ast
import json
import math
import re
from typing import Dict

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

    if reward_type in ("none", "zero"):
        return 0.0

    raise ValueError(
        f"Unsupported reward_type={reward_type!r}. "
        "Use math_latex, exact_match, contains, regex, format_only, code, zero, "
        "or provide a custom_reward_func through the multi-task config."
    )
