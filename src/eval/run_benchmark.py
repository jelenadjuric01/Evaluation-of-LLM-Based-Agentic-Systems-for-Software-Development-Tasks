# src/eval/run_benchmark.py
"""
Benchmark runner for HumanEval bug-fixing tasks.

Overview
--------
This script ties everything together:
- Loads HumanEval tasks (buggy code + tests).
- Calls the LLM policy to propose minimal patches.
- Executes tests in a sandbox for each patch.
- Iterates for multiple repair attempts (`max_steps`).
- Records traces, prints summary, and saves a JSON report.

Key Features
------------
- Supports multiple generation strategies: greedy, sampling, beam search.
- Summarizes test failures so the LLM can iteratively improve patches.
- Measures wall-clock runtime and steps used per task.
- Saves full traces for reproducibility and debugging.

Usage
-----
From CLI:
    python -m src.eval.run_benchmark --sample 5 --max_steps 3 --timeout 15 --model_id Qwen/Qwen2.5-Coder-1.5B-Instruct

This will:
1. Sample 5 tasks.
2. Allow up to 3 repair attempts per task.
3. Use 15s timeout per execution.
4. Save a full report to out/report.json.

Terminology
-----------
- pass@1: Fraction of tasks solved on the *first attempt* (not retries).
- trace: History of patches, failures, and stdout/stderr for each attempt.

Dependencies
------------
- Hugging Face Transformers
- Datasets
- A sandbox runner (see `src/sandbox/runner.py`).
"""

from __future__ import annotations
import argparse, json, time
import os
from pathlib import Path
from typing import Dict, Any, List

from src.eval.humanevalfix import load_tasks, Task
from src.agent.policy import (
    generate_patch,
    ModelConfig,
    create_deterministic_config,
    create_sampling_config,
)
from src.sandbox.runner import run_python
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _summarize_failures(fail_text: str, max_chars: int = 800) -> str:
    """
    Summarize test failure output for feedback to the LLM.

    Strategy
    --------
    - Strip leading/trailing whitespace.
    - If empty, return a generic diagnostic (likely wrong signature).
    - If short enough, return as-is.
    - If too long, keep only the tail (~last 30 lines) so the model sees
      assertion errors and tracebacks instead of huge logs.

    Parameters
    ----------
    fail_text : str
        Raw stderr/stdout from test execution.
    max_chars : int, default=800
        Maximum characters allowed before truncation.

    Returns
    -------
    str
        Condensed failure summary string.
    """
    t = (fail_text or "").strip()
    if not t:
        return "Tests failed with no output. Possible wrong function name/signature or silent exit."
    if len(t) <= max_chars:
        return t
    return "\n".join(t.splitlines()[-30:])


def run_task(task: Task, max_steps: int, timeout_s: int, config: ModelConfig) -> Dict[str, Any]:
    """
    Run the agent loop for a single HumanEval task.

    Workflow
    --------
    - Start with the dataset's buggy code.
    - Generate a patch with the LLM (conditioned on previous failure summary).
    - Run patched code against the provided tests in a sandbox.
    - If passed: return immediately with success info.
    - Else: summarize failures and retry (up to `max_steps`).

    Parameters
    ----------
    task : Task
        The HumanEval task (buggy function, tests, entry point).
    max_steps : int
        Maximum number of repair attempts (LLM calls).
    timeout_s : int
        Timeout (seconds) for each sandbox test execution.
    config : ModelConfig
        LLM configuration for generation.

    Returns
    -------
    Dict[str, Any]
        Structured result containing:
        - task_id
        - passed (bool)
        - steps_used (int)
        - trace (list of attempts with patches, outputs, failures)
        - wall_time_s (float, wall-clock runtime)
    """
    code = task.buggy_code
    failure_summary = ""
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(max_steps):
        # Generate candidate patch
        patch = generate_patch(
            code,
            failure_summary=failure_summary,
            func_name=task.entry_point,
            config=config,
        )

        # Run patched code in sandbox
        result = run_python(patch, task.tests, timeout_s=timeout_s)

        # Record trace for this attempt
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
            # Success: return early
            return {
                "task_id": task.task_id,
                "passed": True,
                "steps_used": step + 1,
                "trace": trace,
                "wall_time_s": round(time.time() - t0, 3),
            }

        # Prepare feedback for next attempt
        failure_summary = _summarize_failures(result.get("failures", ""))
        code = patch  # Next attempt builds on last patch

    # Exhausted attempts → fail
    return {
        "task_id": task.task_id,
        "passed": False,
        "steps_used": max_steps,
        "trace": trace,
        "wall_time_s": round(time.time() - t0, 3),
    }


def create_config_from_args(args) -> ModelConfig:
    """
    Build a ModelConfig from CLI args.

    Rules
    -----
    - If `--temperature` is set, use sampling with top-p/top-k.
    - Else if `--num_beams > 1`, use beam search (deterministic).
    - Else default to greedy deterministic.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    ModelConfig
        Ready-to-use configuration for `generate_patch`.
    """
    config_kwargs = {}

    # Model-level overrides
    if getattr(args, 'model_id', None):
        config_kwargs['model_id'] = args.model_id
    if getattr(args, 'device', None):
        config_kwargs['device'] = args.device
    if getattr(args, 'seed', None) is not None:
        config_kwargs['seed'] = args.seed

    # Generation overrides
    if getattr(args, 'max_new_tokens', None):
        config_kwargs['max_new_tokens'] = args.max_new_tokens
    if getattr(args, 'repetition_penalty', None):
        config_kwargs['repetition_penalty'] = args.repetition_penalty
    if getattr(args, 'num_beams', None):
        config_kwargs['num_beams'] = args.num_beams

    # Strategy selection
    if getattr(args, 'temperature', None) is not None:
        return create_sampling_config(
            temperature=args.temperature,
            top_p=getattr(args, 'top_p', 0.9),
            top_k=getattr(args, 'top_k', None),
            **config_kwargs
        )
    elif getattr(args, 'num_beams', None) and args.num_beams > 1:
        config_kwargs['num_beams'] = args.num_beams
        config_kwargs['early_stopping'] = True
        return ModelConfig(do_sample=False, **config_kwargs)
    else:
        return create_deterministic_config(**config_kwargs)


def main():
    """
    CLI entrypoint for running the HumanEval bug-fixing benchmark.

    Responsibilities
    ----------------
    - Parse arguments (benchmark settings + model config).
    - Create appropriate ModelConfig.
    - Load tasks (optionally sample).
    - Run evaluation loop, printing per-task results.
    - Compute and print overall pass@1.
    - Save JSON report with full results + config.
    """
    ap = argparse.ArgumentParser(
        description="Run HumanEval bug fixing benchmark with configurable model"
    )

    # Benchmark settings
    ap.add_argument("--sample", type=int, default=None,
                    help="Sample N tasks (optional).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--max_steps", type=int, default=1,
                    help="Max repair attempts per task")
    ap.add_argument("--timeout", type=int, default=10,
                    help="Timeout per test execution (s)")
    ap.add_argument("--report", type=str, default="out/report.json",
                    help="Output report path")

    # Model configuration
    ap.add_argument("--model_id", type=str, default=None,
                    help="Hugging Face model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    ap.add_argument("--device", type=str, default=None,
                    help="Device (cuda/cpu, default: auto-detect)")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="Max tokens to generate (default: 1024)")
    ap.add_argument("--repetition_penalty", type=float, default=None,
                    help="Repetition penalty (default: 1.05)")

    # Mutually exclusive generation strategies
    generation_group = ap.add_mutually_exclusive_group()
    generation_group.add_argument("--temperature", type=float, default=None,
                                  help="Enable sampling with given temperature")
    generation_group.add_argument("--num_beams", type=int, default=None,
                                  help="Enable beam search with N beams")

    # Extra sampling parameters
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="Top-p nucleus sampling (default=0.9)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="Top-k sampling cutoff")

    args = ap.parse_args()

    # Build model config
    config = create_config_from_args(args)

    # Echo configuration summary
    print("Model Configuration:")
    print(f"  Model ID: {config.model_id}")
    print(f"  Device: {config.device}")
    mode = "Sampling" if config.do_sample else (
        "Beam Search" if config.num_beams > 1 else "Greedy")
    print(f"  Generation mode: {mode}")
    if config.do_sample:
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Top-k: {config.top_k}")
    elif config.num_beams > 1:
        print(f"  Num beams: {config.num_beams}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Repetition penalty: {config.repetition_penalty}")
    print()

    # Load tasks (with optional sub-sampling)
    tasks = load_tasks(sample=args.sample, seed=args.seed)

    results: List[Dict[str, Any]] = []
    passed = 0

    # Main evaluation loop
    for t in tasks:
        print(f"\n=== Running {t.task_id} ===")
        out = run_task(t, max_steps=args.max_steps, timeout_s=args.timeout, config=config)
        results.append(out)
        print(f"Result: {'PASS' if out['passed'] else 'FAIL'} in {out['steps_used']} step(s).")
        passed += int(out["passed"])

    total = len(tasks)
    pass_at_1 = passed / total if total else 0.0
    print(f"\n=== Summary ===\nPassed {passed}/{total}  pass@1={pass_at_1:.3f}")

    # Save report with config and results
    config_info = {
        "model_id": config.model_id,
        "device": config.device,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "num_beams": config.num_beams,
        "max_new_tokens": config.max_new_tokens,
        "repetition_penalty": config.repetition_penalty,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps({
            "pass_at_1": pass_at_1,
            "passed": passed,
            "total": total,
            "config": config_info,
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
