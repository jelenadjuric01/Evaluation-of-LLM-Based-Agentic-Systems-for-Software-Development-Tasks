# src/sandbox/runner.py
"""
Lightweight Python sandbox runner for bug-fixing tasks.

Overview
--------
This module executes a candidate Python function against HumanEval-style tests
in an isolated temporary directory. It does **not** use pytest; instead, it
runs the test file directly as a Python script, capturing stdout/stderr.

Sandbox Features
----------------
- Each run gets a fresh temporary directory.
- Candidate function is saved as `candidate.py`.
- Tests are saved as `test_candidate.py` with `from candidate import ...`.
- The temp dir is added to `sys.path` inside the test script so imports work.
- Execution is done with `python -I -B` to avoid user site packages and .pyc files.
- Timeout is enforced to avoid infinite loops or hangs.
- Cleanup removes the temp directory after execution.

Return Format
-------------
The result dictionary has the following keys:
- passed : bool   → True if returncode==0, else False
- stdout : str    → Captured standard output
- stderr : str    → Captured standard error
- failures : str  → Condensed failure info (stderr + stdout if failed)
- duration_s : float → Execution time in seconds

Usage
-----
>>> result = run_python(candidate_code, tests, timeout_s=5)
>>> if result["passed"]:
...     print("All tests passed!")
... else:
...     print("Failures:", result["failures"])
"""

import sys
import tempfile
import subprocess
import shutil
import time
from pathlib import Path


def run_python(candidate_code: str, tests: str, timeout_s: int = 5):
    """
    Execute candidate code + tests in a sandboxed temp directory.

    Parameters
    ----------
    candidate_code : str
        Source code for the candidate function/module (usually a single `def`).
        Written to `candidate.py`.
    tests : str
        Test code string. It should already import the entry point via
        `from candidate import <func>`. Written to `test_candidate.py`.
    timeout_s : int, default 5
        Max seconds to allow the process to run before killing it.

    Returns
    -------
    dict
        {
          "passed": bool,
          "stdout": str,
          "stderr": str,
          "failures": str,
          "duration_s": float
        }

    Notes
    -----
    - Uses `subprocess.run` with `-I -B` flags:
      * `-I`: isolate from user site-packages.
      * `-B`: disable .pyc writing.
    - If timeout occurs, returns with `passed=False` and failures="TIMEOUT".
    - Cleans up temp directory after run.
    """
    work = Path(tempfile.mkdtemp(prefix="agentfix_"))
    try:
        # --- Step 1: Write candidate code ---
        (work / "candidate.py").write_text(candidate_code, encoding="utf-8")

        # --- Step 2: Wrap tests with sys.path injection so `candidate` can be imported ---
        sys_path_line = f"import sys; sys.path.insert(0, r\"{work}\")\n"
        test_file = work / "test_candidate.py"
        test_file.write_text(sys_path_line + tests, encoding="utf-8")

        # --- Step 3: Run test file as a plain Python script ---
        cmd = [sys.executable, "-I", "-B", str(test_file)]
        start = time.time()
        try:
            p = subprocess.run(
                cmd,
                cwd=str(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            # Hard timeout → mark as failed
            return {
                "passed": False,
                "stdout": (e.stdout or ""),
                "stderr": "TIMEOUT",
                "failures": "TIMEOUT",
                "duration_s": round(time.time() - start, 3),
            }

        dur = time.time() - start
        passed = (p.returncode == 0)

        # Collect failures only if not passed
        failures = ""
        if not passed:
            failures = ((p.stderr or "") + ("\n" + p.stdout if p.stdout else "")).strip()

        return {
            "passed": passed,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "failures": failures,
            "duration_s": round(dur, 3),
        }

    finally:
        # Always clean up temp dir to avoid clutter
        shutil.rmtree(work, ignore_errors=True)
