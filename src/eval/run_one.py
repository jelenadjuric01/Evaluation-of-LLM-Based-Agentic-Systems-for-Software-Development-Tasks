# src/eval/run_one.py
"""
Single-task demo runner for the bug-fixing agent.

Overview
--------
This script demonstrates the bug-fixing loop on a *single* hard-coded task:
- A simple buggy function (`is_palindrome`) is defined inline.
- A corresponding test suite is also defined inline.
- The agent (LLM policy) generates one patch.
- The patch is executed in a sandbox against the tests.
- Results are printed to the console.

Usage
-----
    python -m src.eval.run_one --timeout 5

Notes
-----
- This is primarily for debugging and demonstration.
- For full evaluation across many tasks, see `run_benchmark.py`.
- Replace `BUGGY` and `TESTS` with HumanEvalFix tasks for real evaluation.
"""

import argparse
from src.agent.policy import generate_patch
from src.sandbox.runner import run_python


# -------------------------------------------------------------------------
# Example buggy function and tests.
# In practice, these would come from a HumanEvalFix `Task`.
# -------------------------------------------------------------------------
BUGGY = """\
def is_palindrome(s: str) -> bool:
    # BUG: currently compares the string to itself → always True if non-empty
    return s == s
"""

TESTS = """\
from candidate import is_palindrome

def test_basic():
    # Empty string and single character are palindromes
    assert is_palindrome("") is True
    assert is_palindrome("a") is True
    # Even-length palindrome
    assert is_palindrome("abba") is True
    # Non-palindromes
    assert is_palindrome("abc") is False
    assert is_palindrome("abca") is False
"""


def main():
    """
    CLI entrypoint: run one patch generation + test cycle.

    Steps
    -----
    1. Parse CLI args (`--timeout`).
    2. Generate one patch for the hard-coded buggy function.
    3. Run the patch against the hard-coded tests in the sandbox.
    4. Print patch and test results.
    """
    parser = argparse.ArgumentParser(description="Run single bug-fix demo task")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Timeout (seconds) for test execution")
    args = parser.parse_args()

    # --- Step 1: Generate candidate patch ---
    print(">>> Generating patch...")
    patch = generate_patch(BUGGY, failure_summary="")  # no failure summary in first attempt
    print("=== PATCH ===")
    print(patch)

    # --- Step 2: Run candidate patch against tests ---
    print(">>> Running tests...")
    result = run_python(patch, TESTS, timeout_s=args.timeout)

    # --- Step 3: Print results in readable format ---
    print("=== RESULT ===")
    for k, v in result.items():
        if k != "workdir":  # avoid printing temp dir path
            print(f"{k}: {v}")

    # High-level outcome
    if result.get("passed"):
        print("Passed!")
    else:
        print("Failed. See failures above.")


if __name__ == "__main__":
    main()
