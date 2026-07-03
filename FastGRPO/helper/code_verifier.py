"""Python code verification helpers for FastGRPO code rewards."""

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-.#]*)\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CodeVerificationResult:
    passed: bool
    timed_out: bool
    error_type: str
    stdout: str = ""
    stderr: str = ""


def extract_python_code(completion, entry_point=None):
    """Extract executable Python from a model completion."""
    text = str(completion or "").strip()
    fenced_blocks = _FENCE_RE.findall(text)
    if fenced_blocks:
        text = fenced_blocks[-1].strip()

    if entry_point:
        function_match = re.search(rf"(^|\n)\s*def\s+{re.escape(str(entry_point))}\s*\(", text)
        class_match = re.search(rf"(^|\n)\s*class\s+{re.escape(str(entry_point))}\b", text)
        matches = [match for match in (function_match, class_match) if match is not None]
        if matches:
            text = text[min(match.start() for match in matches):].strip()

    return textwrap.dedent(text).strip()


def verify_code_completion(
    completion,
    tests,
    entry_point=None,
    test_type="unit",
    timeout_seconds=5.0,
    starter_code=None,
):
    """Run generated Python against configured tests in a subprocess.

    This is a bounded local subprocess runner, not a hardened sandbox. Use it only
    with datasets and generated code you are willing to execute on the training
    machine.
    """
    candidate = extract_python_code(completion, entry_point=entry_point)
    if not candidate:
        return CodeVerificationResult(False, False, "empty_completion")

    timeout_seconds = _coerce_timeout(timeout_seconds)
    test_type = str(test_type or "unit").lower()
    if test_type in ("stdin_stdout", "stdio", "io", "input_output"):
        return _verify_stdin_stdout(
            candidate,
            tests,
            timeout_seconds=timeout_seconds,
            starter_code=starter_code,
        )

    test_code = _normalize_tests(tests)
    if not test_code.strip():
        return CodeVerificationResult(False, False, "empty_tests")

    return _verify_unit_tests(
        candidate,
        test_code,
        timeout_seconds=timeout_seconds,
        starter_code=starter_code,
    )


def _verify_unit_tests(candidate, test_code, timeout_seconds, starter_code=None):
    with tempfile.TemporaryDirectory(prefix="fastgrpo_code_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        runner_path = tmp_path / "test_runner.py"
        solution_path.write_text(_build_solution(candidate, starter_code), encoding="utf-8")
        runner_path.write_text(_build_unit_test_runner(test_code), encoding="utf-8")
        return _run_python_script(runner_path, tmp_path, timeout_seconds)


def _verify_stdin_stdout(candidate, tests, timeout_seconds, starter_code=None):
    test_cases = _normalize_io_tests(tests)
    if not test_cases:
        return CodeVerificationResult(False, False, "empty_tests")

    stdout_parts = []
    stderr_parts = []
    with tempfile.TemporaryDirectory(prefix="fastgrpo_code_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        solution_path.write_text(_build_solution(candidate, starter_code), encoding="utf-8")

        for index, test_case in enumerate(test_cases):
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(solution_path)],
                    cwd=str(tmp_path),
                    input=test_case["input"],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return CodeVerificationResult(
                    False,
                    True,
                    "timeout",
                    stdout=_clean_subprocess_text(exc.stdout),
                    stderr=_clean_subprocess_text(exc.stderr),
                )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            if proc.returncode != 0:
                return CodeVerificationResult(
                    False,
                    False,
                    _classify_error(stderr),
                    stdout=stdout,
                    stderr=stderr,
                )
            if not _outputs_match(stdout, test_case["output"]):
                return CodeVerificationResult(
                    False,
                    False,
                    "wrong_answer",
                    stdout=f"test_case={index}\n{stdout}",
                    stderr=stderr,
                )

    return CodeVerificationResult(
        True,
        False,
        "none",
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
    )


def _run_python_script(script_path, cwd, timeout_seconds):
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return CodeVerificationResult(
            False,
            True,
            "timeout",
            stdout=_clean_subprocess_text(exc.stdout),
            stderr=_clean_subprocess_text(exc.stderr),
        )

    if proc.returncode == 0:
        return CodeVerificationResult(True, False, "none", proc.stdout or "", proc.stderr or "")
    return CodeVerificationResult(
        False,
        False,
        _classify_error(proc.stderr or ""),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _build_solution(candidate, starter_code=None):
    pieces = []
    if starter_code:
        pieces.append(textwrap.dedent(str(starter_code)).strip())
    pieces.append(candidate)
    return "\n\n".join(piece for piece in pieces if piece) + "\n"


def _build_unit_test_runner(test_code):
    return "\n".join(
        [
            "import inspect",
            "import sys",
            "import types",
            "import unittest",
            "from pathlib import Path",
            "",
            "def _fastgrpo_blocked_exit(*args, **kwargs):",
            "    raise RuntimeError('sys.exit is disabled during code verification')",
            "",
            "sys.exit = _fastgrpo_blocked_exit",
            "",
            "sys.path.insert(0, str(Path(__file__).resolve().parent))",
            "from solution import *",
            "",
            "_test_module = types.ModuleType('_fastgrpo_tests')",
            "_test_module.__dict__.update(globals())",
            "_test_module.__dict__['__name__'] = '_fastgrpo_tests'",
            f"exec({test_code!r}, _test_module.__dict__)",
            "",
            "_plain_failures = []",
            "for _name, _value in sorted(_test_module.__dict__.items()):",
            "    if _name.startswith('test_') and inspect.isfunction(_value):",
            "        try:",
            "            _value()",
            "        except Exception as exc:",
            "            _plain_failures.append((_name, exc))",
            "if _plain_failures:",
            "    _name, _exc = _plain_failures[0]",
            "    raise AssertionError(f'{_name} failed: {_exc}')",
            "",
            "_suite = unittest.defaultTestLoader.loadTestsFromModule(_test_module)",
            "if _suite.countTestCases():",
            "    _result = unittest.TextTestRunner(verbosity=0).run(_suite)",
            "    if not _result.wasSuccessful():",
            "        raise AssertionError('unittest failures')",
            "",
        ]
    )


def _normalize_tests(tests):
    if tests is None:
        return ""
    if isinstance(tests, str):
        return textwrap.dedent(tests).strip()
    if isinstance(tests, Mapping):
        return json.dumps(tests, ensure_ascii=False)
    if isinstance(tests, Iterable):
        return "\n".join(str(item).strip() for item in tests if str(item).strip())
    return str(tests)


def _normalize_io_tests(tests):
    if tests is None:
        return []
    if isinstance(tests, Mapping):
        tests = [tests]
    elif isinstance(tests, str):
        stripped = tests.strip()
        try:
            parsed = json.loads(stripped)
            tests = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            tests = [tests]

    normalized = []
    for item in tests:
        if isinstance(item, Mapping):
            if "input" in item or "output" in item:
                normalized.append({
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", "")),
                })
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                normalized.append({
                    "input": str(parsed.get("input", "")),
                    "output": str(parsed.get("output", "")),
                })
    return normalized


def _outputs_match(actual, expected):
    actual_clean = str(actual).strip()
    expected_clean = str(expected).strip()
    if actual_clean == expected_clean:
        return True
    return actual_clean.split() == expected_clean.split()


def _classify_error(stderr):
    if "AssertionError" in stderr:
        return "assertion"
    if "SyntaxError" in stderr:
        return "syntax"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import"
    return "runtime"


def _coerce_timeout(timeout_seconds):
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout = 5.0
    return max(timeout, 0.05)


def _clean_subprocess_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
