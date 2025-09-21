# src/eval/run_benchmark.py
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

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


def run_task(task: Task, max_steps: int, timeout_s: int, gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the agent loop for a single task.
    Returns a dict with: task_id, passed, steps_used, wall_time_s, trace[...]
    """
    code = task.buggy_code
    failure_summary = ""
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(max_steps):
        # 1) Generate candidate
        patch = generate_patch(
            code,
            failure_summary=failure_summary,
            func_name=task.entry_point,
            gen_cfg=gen_cfg,
        )

        # 2) Execute tests
        result = run_python(patch, task.tests, timeout_s=timeout_s)

        # 3) Log step
        trace.append({
            "step": step,
            "patch": patch,
            "passed": result["passed"],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "failures": result.get("failures", ""),
            "duration_s": result.get("duration_s", 0.0),
        })

        # 4) Success → return
        if result["passed"]:
            return {
                "task_id": task.task_id,
                "passed": True,
                "steps_used": step + 1,
                "trace": trace,
                "wall_time_s": round(time.time() - t0, 3),
            }

        # 5) Prepare feedback for next iteration
        failure_summary = _summarize_failures(result.get("failures", ""))

        # 6) Optional: on the final step, try a couple low-temp variants
        last_step = (step == max_steps - 1)
        greedy_run = not bool(gen_cfg.get("do_sample", False))
        if last_step and greedy_run:
            candidates: List[Dict[str, Any]] = []

            for _ in range(2):  # try 2 extra variants
                alt_cfg = {
                    **gen_cfg,
                    "do_sample": True,
                    "temperature": 0.2 if gen_cfg.get("temperature") is None else gen_cfg["temperature"],
                    "top_p": 0.9 if gen_cfg.get("top_p") is None else gen_cfg["top_p"],
                    "top_k": 20 if gen_cfg.get("top_k") is None else gen_cfg["top_k"],
                }
                alt_patch = generate_patch(
                    code,
                    failure_summary=failure_summary,
                    func_name=task.entry_point,
                    gen_cfg=alt_cfg,
                )
                alt_res = run_python(alt_patch, task.tests, timeout_s=timeout_s)
                candidates.append({"patch": alt_patch, "result": alt_res})

                trace.append({
                    "step": step,
                    "variant": True,
                    "patch": alt_patch,
                    "passed": alt_res["passed"],
                    "stdout": alt_res.get("stdout", ""),
                    "stderr": alt_res.get("stderr", ""),
                    "failures": alt_res.get("failures", ""),
                    "duration_s": alt_res.get("duration_s", 0.0),
                })

                if alt_res["passed"]:
                    return {
                        "task_id": task.task_id,
                        "passed": True,
                        "steps_used": step + 1,
                        "trace": trace,
                        "wall_time_s": round(time.time() - t0, 3),
                    }

            # none of the variants passed; fall through and finalize fail

        # 7) Base next attempt on the latest candidate (nudges minimal diffs)
        code = patch

    # Exhausted budget without passing
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
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--report", type=str, default="out/report.json")

    # Decoding knobs
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)

    args = ap.parse_args()

    # Load tasks (HumanEvalPack / python / test split)
    tasks = load_tasks(sample=args.sample, seed=args.seed)

    # Build generation config dictionary (passed down to policy)
    gen_cfg: Dict[str, Any] = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    results: List[Dict[str, Any]] = []
    passed = 0

    for t in tasks:
        print(f"\n=== Running {t.task_id} ===")
        out = run_task(t, max_steps=args.max_steps, timeout_s=args.timeout, gen_cfg=gen_cfg)
        results.append(out)
        print(f"Result: {'PASS' if out['passed'] else 'FAIL'} in {out['steps_used']} step(s).")
        passed += int(out["passed"])

    total = len(tasks)
    pass_at_1 = (passed / total) if total else 0.0
    print(f"\n=== Summary ===\nPassed {passed}/{total}  pass@1={pass_at_1:.3f}")

    Path("out").mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(
            {"pass_at_1": pass_at_1, "passed": passed, "total": total, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
