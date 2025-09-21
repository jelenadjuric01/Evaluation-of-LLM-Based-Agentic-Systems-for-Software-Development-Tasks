# src/sandbox/runner.py
import sys, tempfile, subprocess, shutil, time
from pathlib import Path

def run_python(candidate_code: str, tests: str, timeout_s: int = 5):
    """
    Execute candidate_code + HumanEvalPack test script directly with Python.
    HumanEvalPack tests call check(entry_point) at module top-level (no pytest).
    Returns {passed: bool, stdout, stderr, failures, duration_s}.
    """
    work = Path(tempfile.mkdtemp(prefix="agentfix_"))
    try:
        # Write candidate and tests
        (work / "candidate.py").write_text(candidate_code, encoding="utf-8")

        # Ensure the temp dir is importable for `from candidate import ...`
        sys_path_line = f"import sys; sys.path.insert(0, r\"{work}\")\n"
        test_file = work / "test_candidate.py"
        test_file.write_text(sys_path_line + tests, encoding="utf-8")

        # Run the test script directly (no pytest)
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
            return {
                "passed": False,
                "stdout": (e.stdout or ""),
                "stderr": "TIMEOUT",
                "failures": "TIMEOUT",
                "duration_s": round(time.time() - start, 3),
            }

        dur = time.time() - start
        passed = (p.returncode == 0)
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
        shutil.rmtree(work, ignore_errors=True)