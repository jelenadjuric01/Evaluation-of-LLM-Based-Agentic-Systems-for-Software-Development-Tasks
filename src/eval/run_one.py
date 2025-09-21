import argparse
from src.agent.policy import generate_patch
from src.sandbox.runner import run_python


# --- Example buggy function + tests (replace later with HumanEvalFix task) ---
BUGGY = """\
def is_palindrome(s: str) -> bool:
    # BUG: compares to itself, always True for len>0
    return s == s
"""

TESTS = """\
from candidate import is_palindrome

def test_basic():
    assert is_palindrome("") is True
    assert is_palindrome("a") is True
    assert is_palindrome("abba") is True
    assert is_palindrome("abc") is False
    assert is_palindrome("abca") is False
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    print(">>> Generating patch...")
# in src/eval/run_one.py
    patch = generate_patch(BUGGY, failure_summary="")
    print("=== PATCH ===")
    print(patch)
    print(">>> Running tests...")
    result = run_python(patch, TESTS, timeout_s=args.timeout)

    print("=== RESULT ===")
    for k, v in result.items():
        if k != "workdir":
            print(f"{k}: {v}")
    if result.get("passed"):
        print("Passed!")
    else:
        print("Failed. See failures above.")

if __name__ == "__main__":
    main()
