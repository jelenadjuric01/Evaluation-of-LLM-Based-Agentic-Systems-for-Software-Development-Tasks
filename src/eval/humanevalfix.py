# src/eval/humanevalfix.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random

from datasets import load_dataset  

@dataclass
class Task:
    task_id: str
    buggy_code: str
    tests: str
    entry_point: str  # function name

def load_tasks(sample: Optional[int] = None, seed: int = 42) -> List[Task]:
    """
    Loads HumanEvalPack (Python) test split from Hugging Face.
    We use fields:
      - buggy_solution -> buggy_code
      - test           -> tests (already calls check(entry_point))
      - entry_point    -> entry_point

    Returns a list[Task]. Optionally random-sample N tasks.
    """
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    tasks: List[Task] = []

    for row in ds:
        buggy = row["buggy_solution"].rstrip()
        entry = row["entry_point"]
        tests = row["test"].rstrip()

        # Prepend an import so tests can refer to the symbol by name.
        # The dataset's tests end with `check(<entry_point>)`, so importing
        # the symbol into the module's namespace is enough.
        tests_wrapped = f"from candidate import {entry}\n\n{tests}"

        tasks.append(Task(
            task_id=row["task_id"],
            buggy_code=buggy,
            tests=tests_wrapped,
            entry_point=entry,
        ))

    if sample is not None and sample < len(tasks):
        random.seed(seed)
        tasks = random.sample(tasks, sample)

    return tasks
