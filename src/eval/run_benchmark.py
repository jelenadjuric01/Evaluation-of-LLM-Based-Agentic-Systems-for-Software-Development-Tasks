# src/eval/run_benchmark.py
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict, Any, List

from src.eval.humanevalfix import load_tasks, Task
from src.agent.policy import generate_patch, ModelConfig, create_deterministic_config, create_sampling_config
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


def run_task(task: Task, max_steps: int, timeout_s: int, config: ModelConfig) -> Dict[str, Any]:
    """Run agent loop for a single task with given model configuration."""
    code = task.buggy_code
    failure_summary = ""
    trace: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(max_steps):
        patch = generate_patch(
            code, 
            failure_summary=failure_summary, 
            func_name=task.entry_point,
            config=config
        )
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


def create_config_from_args(args) -> ModelConfig:
    """Create ModelConfig from command line arguments."""
    config_kwargs = {}
    
    # Model settings
    if hasattr(args, 'model_id') and args.model_id:
        config_kwargs['model_id'] = args.model_id
    if hasattr(args, 'device') and args.device:
        config_kwargs['device'] = args.device
    if hasattr(args, 'seed') and args.seed is not None:
        config_kwargs['seed'] = args.seed
    
    # Generation parameters
    if hasattr(args, 'max_new_tokens') and args.max_new_tokens:
        config_kwargs['max_new_tokens'] = args.max_new_tokens
    if hasattr(args, 'repetition_penalty') and args.repetition_penalty:
        config_kwargs['repetition_penalty'] = args.repetition_penalty
    if hasattr(args, 'num_beams') and args.num_beams:
        config_kwargs['num_beams'] = args.num_beams
    
    # Sampling vs deterministic
    if hasattr(args, 'temperature') and args.temperature is not None:
        # If temperature is provided, use sampling
        return create_sampling_config(
            temperature=args.temperature,
            top_p=getattr(args, 'top_p', 0.9),
            top_k=getattr(args, 'top_k', None),
            **config_kwargs
        )
    elif hasattr(args, 'num_beams') and args.num_beams is not None and args.num_beams > 1:
        # If num_beams > 1, use beam search
        config_kwargs['num_beams'] = args.num_beams
        config_kwargs['early_stopping'] = True
        return ModelConfig(do_sample=False, **config_kwargs)
    else:
        # Default to deterministic
        return create_deterministic_config(**config_kwargs)


def main():
    ap = argparse.ArgumentParser(description="Run HumanEval bug fixing benchmark with configurable model")
    
    # Benchmark settings
    ap.add_argument("--sample", type=int, default=None, help="Sample N tasks (optional).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_steps", type=int, default=1, help="Max repair attempts per task")
    ap.add_argument("--timeout", type=int, default=10, help="Timeout per test execution")
    ap.add_argument("--report", type=str, default="out/report.json", help="Output report path")
    
    # Model configuration
    ap.add_argument("--model_id", type=str, default=None, 
                    help="Hugging Face model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    ap.add_argument("--device", type=str, default=None, 
                    help="Device to use (cuda/cpu, default: auto-detect)")
    ap.add_argument("--max_new_tokens", type=int, default=None, 
                    help="Max tokens to generate (default: 1024)")
    ap.add_argument("--repetition_penalty", type=float, default=None, 
                    help="Repetition penalty (default: 1.05)")
    
    # Generation strategy
    generation_group = ap.add_mutually_exclusive_group()
    generation_group.add_argument("--temperature", type=float, default=None,
                                help="Temperature for sampling (enables do_sample=True)")
    generation_group.add_argument("--num_beams", type=int, default=None,
                                help="Number of beams for beam search")
    
    # Additional sampling parameters (only used with --temperature)
    ap.add_argument("--top_p", type=float, default=0.9, 
                    help="Top-p for sampling (default: 0.9)")
    ap.add_argument("--top_k", type=int, default=None, 
                    help="Top-k for sampling")
    
    args = ap.parse_args()

    # Create model configuration
    config = create_config_from_args(args)
    
    print("Model Configuration:")
    print(f"  Model ID: {config.model_id}")
    print(f"  Device: {config.device}")
    print(f"  Generation mode: {'Sampling' if config.do_sample else 'Beam Search' if config.num_beams > 1 else 'Greedy'}")
    if config.do_sample:
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Top-k: {config.top_k}")
    elif config.num_beams > 1:
        print(f"  Num beams: {config.num_beams}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Repetition penalty: {config.repetition_penalty}")
    print()

    tasks = load_tasks(sample=args.sample, seed=args.seed)
    results: List[Dict[str, Any]] = []
    passed = 0

    for t in tasks:
        print(f"\n=== Running {t.task_id} ===")
        out = run_task(t, max_steps=args.max_steps, timeout_s=args.timeout, config=config)
        results.append(out)
        print(f"Result: {'PASS' if out['passed'] else 'FAIL'} in {out['steps_used']} step(s).")
        passed += int(out["passed"])

    total = len(tasks)
    pass_at_1 = passed / total if total else 0.0
    print(f"\n=== Summary ===\nPassed {passed}/{total}  pass@1={pass_at_1:.3f}")

    # Save configuration info in report
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

    Path("out").mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "config": config_info,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()