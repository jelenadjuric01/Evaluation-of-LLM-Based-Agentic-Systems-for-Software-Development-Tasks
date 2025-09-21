# src/eval/run_benchmark.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict, Any, List

from src.eval.humanevalfix import load_tasks, Task
from src.agent.policy import generate_patch
from src.sandbox.runner import run_python


def _summarize_failures(fail_text: str, max_chars: int = 800) -> str:
    """
    Summarize direct-exec test output for the LLM.
    Keep the tail (assertion/traceback), drop long headers.
    """
    t = (fail_text or "").strip()
    if not t:
        return "Tests failed with no output. Possible wrong function name/signature or silent exit."
    if len(t) <= max_chars:
        return t
    return "\n".join(t.splitlines()[-30:])



def run_task(task: Task, max_steps: int, timeout_s: int) -> Dict[str, Any]:
    """Run agent loop for a single task."""
    code = task.buggy_code
    failure_summary = ""
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(max_steps):
        patch = generate_patch(code, failure_summary=failure_summary, func_name=task.entry_point)
        result = run_python(patch, task.tests, timeout_s=timeout_s)

        trace.append({
            "step": step,
            "patch": patch,
            "passed": result["passed"],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "failures": result.get("failures", ""),
            "duration_s": result.get("duration_s", 0.0),
        })

        if result["passed"]:
            return {
                "task_id": task.task_id,
                "passed": True,
                "steps_used": step + 1,
                "trace": trace,
                "wall_time_s": round(time.time() - t0, 3),
            }

        # prepare feedback for next step
        failure_summary = _summarize_failures(result.get("failures", ""))
        code = patch  # optional: base next attempt on last patch

    return {
        "task_id": task.task_id,
        "passed": False,
        "steps_used": max_steps,
        "trace": trace,
        "wall_time_s": round(time.time() - t0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Sample N tasks (optional).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--report", type=str, default="out/report.json")
    args = ap.parse_args()

    tasks = load_tasks(sample=args.sample, seed=args.seed)
    results: List[Dict[str, Any]] = []
    passed = 0

    for t in tasks:
        print(f"\n=== Running {t.task_id} ===")
        out = run_task(t, max_steps=args.max_steps, timeout_s=args.timeout)
        results.append(out)
        print(f"Result: {'PASS' if out['passed'] else 'FAIL'} in {out['steps_used']} step(s).")
        passed += int(out["passed"])

    total = len(tasks)
    pass_at_1 = passed / total if total else 0.0
    print(f"\n=== Summary ===\nPassed {passed}/{total}  pass@1={pass_at_1:.3f}")

    Path("out").mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()