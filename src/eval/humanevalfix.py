# src/eval/humanevalfix.py
"""
HumanEvalPack task loader for "bug-fix" experiments.

Overview
--------
This module reads the **HumanEvalPack (Python)** test split from Hugging Face
(`bigcode/humanevalpack`) and converts each row into a lightweight `Task`
object containing:

- `task_id`: benchmark identifier (e.g., "HumanEval/0")
- `buggy_code`: the intentionally broken reference implementation
- `tests`: the unit tests that call `check(entry_point)` to validate fixes
- `entry_point`: the function name that must be fixed (e.g., "fib")

Why this wrapper?
-----------------
Most agents/policies expect a simple structure (code + tests + entry point).
We also prepend an import line into the test string so that `check(...)`
can resolve the symbol from a file named `candidate.py`.

Typical usage
-------------
>>> tasks = load_tasks(sample=10, seed=123)
>>> t = tasks[0]
>>> t.entry_point, t.task_id
('fib', 'HumanEval/0')

Notes
-----
- The dataset's `test` field already contains calls to `check(entry_point)`.
- We **do not** alter test logic—only add a single import line:
  `from candidate import <entry_point>`.
- Sampling is optional and reproducible via `seed`.

Dependencies
------------
- datasets >= 2.x  (Hugging Face Datasets)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random

from datasets import load_dataset  # HF Datasets: pip install datasets


@dataclass
class Task:
    """
    A single HumanEvalPack task.

    Attributes
    ----------
    task_id : str
        Unique benchmark identifier (e.g., "HumanEval/0").
    buggy_code : str
        The original buggy function source provided by the dataset.
        This should be written to `candidate.py` before running tests.
    tests : str
        Python test code that ends with a call to `check(entry_point)`.
        We inject `from candidate import <entry_point>` as the first line so
        that the test code can import the user's (or agent's) candidate fix.
    entry_point : str
        The function name that must be implemented/fixed.
    """
    task_id: str
    buggy_code: str
    tests: str
    entry_point: str  # function name


def load_tasks(sample: Optional[int] = None, seed: int = 42) -> List[Task]:
    """
    Load HumanEvalPack (Python) test split and convert rows to `Task` objects.

    Field mapping (dataset → Task)
    ------------------------------
    - `buggy_solution` → `buggy_code`
    - `test`           → `tests` (wrapped with an import line)
    - `entry_point`    → `entry_point`
    - `task_id`        → `task_id`

    Test wrapping
    -------------
    The dataset's tests end with `check(<entry_point>)`. We prepend a single
    import so that when this `tests` string is executed in a module context,
    it can resolve the function from a file named `candidate.py`:

        from candidate import <entry_point>

    Parameters
    ----------
    sample : Optional[int], default None
        If provided and less than the number of available tasks, return a
        *random* subset of exactly `sample` tasks. Useful for smoke tests.
    seed : int, default 42
        Random seed used only when `sample` is provided, to ensure reproducible
        sub-sampling.

    Returns
    -------
    List[Task]
        A list of prepared tasks ready for execution in your evaluation loop.

    Examples
    --------
    >>> tasks = load_tasks()
    >>> # Write the buggy code to candidate.py, then run `exec(task.tests, {})`
    >>> # or save tests to a file and run via subprocess/pytest.

    Notes
    -----
    - Requires internet access or a local cache for `bigcode/humanevalpack`.
    - We do *not* validate or modify the dataset content beyond the import line.
    - If you need a deterministic full ordering without sampling, leave
      `sample=None`.

    Raises
    ------
    Any exceptions originating from `datasets.load_dataset` (e.g., connectivity
    or missing dataset) will propagate to the caller.
    """
    # Load the Python configuration of HumanEvalPack, test split only.
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")

    tasks: List[Task] = []

    for row in ds:
        # Dataset provides a full module for the buggy solution (typically one function).
        buggy = row["buggy_solution"].rstrip()
        entry = row["entry_point"]
        tests = row["test"].rstrip()

        # Prepend an import so tests can refer to the symbol by name.
        # The dataset's tests end with `check(<entry_point>)`, so importing
        # the symbol into the module's namespace is enough.
        # Assumption: the evaluation harness will place the candidate fix into
        # a file/module named `candidate.py`.
        tests_wrapped = f"from candidate import {entry}\n\n{tests}"

        tasks.append(
            Task(
                task_id=row["task_id"],
                buggy_code=buggy,
                tests=tests_wrapped,
                entry_point=entry,
            )
        )

    # Optional reproducible sub-sampling for quick runs.
    if sample is not None and sample < len(tasks):
        random.seed(seed)
        tasks = random.sample(tasks, sample)

    return tasks
