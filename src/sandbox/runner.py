import os, sys, tempfile, textwrap, subprocess, json, shutil, time, uuid
from pathlib import Path

def run_python(candidate_code: str, tests: str, timeout_s: int = 5, mem_mb: int = 512):
    """
    Executes candidate_code + tests in an isolated temp dir and returns a dict:
    {passed: bool, stdout: str, stderr: str, failures: str}
    """
    work = Path(tempfile.mkdtemp(prefix="agentfix_"))
    try:
        # Write module under test
        mod = work / "candidate.py"
        mod.write_text(candidate_code, encoding="utf-8")

        # Write tests (pytest-style minimal)
        test_file = work / "test_candidate.py"
        test_file.write_text(tests, encoding="utf-8")

        # Runner script: run pytest with short output
        runner = work / "run_tests.py"
        runner.write_text(textwrap.dedent("""
            import sys, pytest
            sys.exit(pytest.main(["-q", "test_candidate.py", "--maxfail=1", "-q"]))
        """).strip()+"\n", encoding="utf-8")

        env = os.environ.copy()
        # Basic isolation: no network signals here, but we don't import non-stdlib by default
        # You can harden further by patching socket, resource limits, etc.

        cmd = [sys.executable, "-I", "-B", str(runner)]
        try:
            # POSIX rlimits if available
            preexec_fn = None
            if hasattr(os, "setsid"):
                preexec_fn = os.setsid
            start = time.time()
            p = subprocess.run(
                cmd, cwd=str(work), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_s, preexec_fn=preexec_fn
            )
            dur = time.time() - start
            passed = (p.returncode == 0)
            return {
                "passed": passed,
                "stdout": p.stdout,
                "stderr": p.stderr,
                "failures": "" if passed else (p.stdout + "\n" + p.stderr),
                "duration_s": round(dur, 3),
                "workdir": str(work),
            }
        except subprocess.TimeoutExpired as e:
            return {"passed": False, "stdout": e.stdout or "", "stderr": "TIMEOUT", "failures": "TIMEOUT"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
