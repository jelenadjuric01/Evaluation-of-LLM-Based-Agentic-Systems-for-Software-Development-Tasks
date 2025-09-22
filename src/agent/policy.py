# src/agent/policy.py
"""
Policy for generating minimal Python bug fixes using HF Transformers (CPU/GPU).

Overview
--------
This module wraps a causal LLM to propose *minimal* patches for a single Python
function that fails tests. It focuses on:
- Configurable, reproducible generation (greedy/sampling/beam search).
- Robust extraction of exactly one function definition from model output.
- Safe fallback to return the original buggy code if parsing fails.

Notes
-----
If you see a NumPy warning from PyTorch (NumPy 2.x vs Torch built on 1.x),
either `pip install "numpy<2.0"` or upgrade Torch to a NumPy-2-compatible build.

Typical Usage
-------------
>>> cfg = create_deterministic_config(model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct")
>>> fixed = generate_patch(buggy_code, failure_summary, func_name="foo", config=cfg)
if fixed.startswith("def "):
    # Write back to file or run tests
    ...

Design Choices
--------------
- We keep a light global cache for tokenizer/model to avoid reloading.
- We validate that the returned text parses as Python and starts with `def `.
- We never return prose or fenced code blocks—only raw function source.
"""

from __future__ import annotations

import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Default Model config =====
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # HF checkpoint for a small coder model
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_SEED = 42


@dataclass
class ModelConfig:
    """
    Configuration for model loading and text generation.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID to load (tokenizer + causal LM).
    device : str
        Device spec for Transformers; typically "cpu" or "cuda".
        Passed to `device_map`. For single-device use, "cpu" or "cuda" is fine.
    dtype : torch.dtype
        Torch dtype for model weights (float16 if CUDA available, else float32).
    seed : int
        Seed used for PyTorch (and best-effort Python/numpy) to aid reproducibility.

    Generation
    ----------
    max_new_tokens : int
        Max new tokens to generate.
    do_sample : bool
        If False, use deterministic decoding (greedy / beam search).
        If True, enable sampling with `temperature`, `top_p` and/or `top_k`.
    temperature : Optional[float]
        Sampling temperature; ignored if `do_sample=False`.
    top_p : Optional[float]
        Nucleus sampling cutoff; ignored if `do_sample=False`.
    top_k : Optional[int]
        Top-k sampling cutoff; ignored if `do_sample=False`.
    repetition_penalty : float
        Penalize repeating tokens to reduce loops.

    Beam Search (used when `do_sample=False`)
    -----------------------------------------
    num_beams : int
        Number of beams for beam search (1 == greedy).
    early_stopping : bool
        Stop generation when all beams are finished.
    length_penalty : float
        >1.0 discourages short outputs; <1.0 encourages them.
    no_repeat_ngram_size : int
        Forbid repeating n-grams of this size in the generated text.

    Notes
    -----
    - Temperature and top-p/top-k are *only* applied when `do_sample=True`.
    - When switching models/devices/dtypes, the cache will reload the model.
    """

    # Model settings
    model_id: str = DEFAULT_MODEL_ID
    device: str = DEFAULT_DEVICE
    dtype: torch.dtype = DEFAULT_DTYPE
    seed: int = DEFAULT_SEED

    # Generation parameters
    max_new_tokens: int = 1024
    do_sample: bool = False  # Set to True for sampling-based generation
    temperature: Optional[float] = None  # Only used if do_sample=True
    top_p: Optional[float] = None       # Only used if do_sample=True
    top_k: Optional[int] = None         # Only used if do_sample=True
    repetition_penalty: float = 1.05

    # Additional generation parameters (deterministic/beam)
    num_beams: int = 1
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def __post_init__(self) -> None:
        """Validate and normalize config right after initialization."""
        if self.do_sample and self.temperature is None:
            # Sensible default so users don't forget to set it.
            self.temperature = 1.0
        if not self.do_sample and self.temperature is not None:
            # Avoid silent confusion where a set temperature has no effect.
            print("Warning: temperature is set but do_sample=False. Temperature will be ignored.")


# ===== Prompt config =====
# System prompt enforces "minimal patch" and guardrails (no I/O, no API changes).
SYSTEM = (
    "You are a careful Python bug fixer. Produce the MINIMAL patch. "
    "Keep the same function signature and behavior unless tests require otherwise. "
    "Avoid I/O, networking, and randomness."
)

# Cached model/tokenizer with config
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_current_config: Optional[ModelConfig] = None


def _seed_everything(seed: int) -> None:
    """
    Best-effort seeding for reproducibility across Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    try:
        import random, numpy as np  # Local import to avoid hard dependency on numpy.
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        # NumPy may be absent—it's fine, we still seed PyTorch.
        pass
    torch.manual_seed(seed)


def load_llm(config: Optional[ModelConfig] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load (and cache) tokenizer + model according to `config`.

    The cache is invalidated and reloaded if `model_id`, `device`, or `dtype`
    change compared to the last loaded configuration.

    Parameters
    ----------
    config : Optional[ModelConfig]
        Desired configuration; if None, uses defaults.

    Returns
    -------
    (tokenizer, model) : Tuple[AutoTokenizer, AutoModelForCausalLM]
        HF tokenizer and causal LM, ready for generation on the requested device.
    """
    global _tokenizer, _model, _current_config

    if config is None:
        config = ModelConfig()

    # Determine whether we need to reload the model/tokenizer.
    need_reload = (
        _model is None
        or _tokenizer is None
        or _current_config is None
        or _current_config.model_id != config.model_id
        or _current_config.device != config.device
        or _current_config.dtype != config.dtype
    )

    if need_reload:
        _seed_everything(config.seed)
        _tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device,  # Single-device map ("cpu" or "cuda")
        )
        # Set pad token to EOS if missing to avoid warnings in generation
        if _model.config.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _model.config.pad_token_id = _tokenizer.eos_token_id

        _current_config = config

    return _tokenizer, _model


def _extract_function(text: str, func_name: Optional[str] = None) -> str:
    """
    Extract exactly one Python `def ...` block from raw model output.

    Extraction strategy (first match wins):
    1) Prefer fenced code block: ```python ... ```.
    2) If `func_name` is provided, capture that exact top-level `def` block.
    3) Otherwise capture the first top-level `def` block.
    4) If the whole output already starts with `def `, return it as-is.

    We also strip occasional sentinel noise lines (e.g., "__END_SENTINEL__", "X").

    Parameters
    ----------
    text : str
        Raw model output (may contain Markdown, prose, multiple functions).
    func_name : Optional[str]
        Expected function name; improves robustness when output includes
        multiple functions or extra text.

    Returns
    -------
    str
        Candidate function source (may still fail AST parse and be rejected later).
    """
    # (1) Attempt to recover content from fenced code blocks if present.
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Lines that occasionally show up as terminators or noise in model output.
    junk = {"__END_SENTINEL__", "X"}

    def strip_junk(s: str) -> str:
        """Remove lines that exactly match known sentinel markers."""
        return "\n".join(ln for ln in s.splitlines() if ln.strip() not in junk).strip()

    # (2) Named function capture:
    # Capture from `def <func_name>(...)` up to the next *top-level* non-indented line.
    if func_name:
        T = text + "\n__END_SENTINEL__"
        m = re.search(rf"(?ms)^(def\s+{re.escape(func_name)}\s*\(.*?)(?=^\S|\Z)", T)
        if m:
            return strip_junk(m.group(1))

    # (3) First top-level def block (same top-level boundary logic as above).
    m = re.search(r"(?ms)^(def\s+\w+\s*\(.*?)(?=^\S|\Z)", text + "\n__END_SENTINEL__")
    if m:
        return strip_junk(m.group(1))

    # (4) Output already starts with `def `
    if text.lstrip().startswith("def "):
        return strip_junk(text)

    # If nothing matched, return cleaned text (likely empty or partial).
    return strip_junk(text)


def _is_valid_python(src: str) -> bool:
    """
    Quick syntax validation by parsing the candidate function with `ast`.

    Parameters
    ----------
    src : str
        Python source code to validate.

    Returns
    -------
    bool
        True if `ast.parse` succeeds; False on `SyntaxError`.
    """
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def generate_patch(
    buggy_code: str,
    failure_summary: str = "",
    func_name: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> str:
    """
    Generate a *minimal* corrected function as raw Python source (no fences, no prose).

    The LLM receives:
    - The buggy function (required).
    - A short failure summary (optional but recommended).
    - Strong constraints to keep the same signature / behavior except to fix the bug.

    Safety & Fallback
    -----------------
    If extraction or parsing fails, the function returns `buggy_code` unchanged to
    avoid import errors or broken pipelines.

    Parameters
    ----------
    buggy_code : str
        The buggy *function-only* source (i.e., starts with `def ...`).
    failure_summary : str, optional
        Short description / traceback snippets summarizing failing tests.
    func_name : Optional[str], optional
        Name of the function to fix. Helps pick the right block if model
        returns multiple candidates.
    config : Optional[ModelConfig], optional
        Model/generation configuration. Uses defaults if None.

    Returns
    -------
    str
        The corrected *single* function source beginning with `def ...`.
        If validation fails, returns the original `buggy_code`.
    """
    if config is None:
        config = ModelConfig()

    tok, model = load_llm(config)

    # User message explains the task and *strict* formatting expectations.
    user = (
        "You will receive a buggy Python function and a short failure summary from executing tests.\n"
        "GOAL: Produce a MINIMAL patch that makes the tests pass.\n\n"
        "Rules:\n"
        f"- Output EXACTLY ONE Python function named `{func_name or 'the same name as in the buggy code'}` "
        "with the SAME SIGNATURE as in the buggy code.\n"
        "- Do NOT add helper functions, classes, imports, prints, logging, or comments.\n"
        "- Do NOT change the public API, parameter order/names, return types, or overall semantics beyond fixing the bug.\n"
        "- No I/O, no networking, no randomness; keep the code deterministic.\n"
        "- Output ONLY raw Python source (no Markdown fences/backticks, no prose before/after).\n\n"
        "### Buggy function:\n"
        f"{buggy_code}\n\n"
        "### Failure summary (may be empty):\n"
        f"{failure_summary}\n\n"
        "### Output:\n"
        "Start with `def ` and emit only the corrected function."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

    # Build a chat prompt string (model-dependent formatting).
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and move to the correct device.
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build generation kwargs from config. We explicitly set the values we care about
    # so behavior is predictable across model versions.
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
        "num_beams": config.num_beams,
        "early_stopping": config.early_stopping,
        "length_penalty": config.length_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
    }

    # Apply sampling-only controls when sampling is enabled; keep them unset otherwise
    # to avoid accidental nondeterminism.
    if config.do_sample:
        if config.temperature is not None:
            gen_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            gen_kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k
    else:
        # Explicitly null out sampling args to make intent obvious.
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated portion.
    gen_text = tok.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Extract one function, then sanitize for any leftover sentinel noise.
    code = _extract_function(gen_text, func_name=func_name).strip()
    code = "\n".join(
        ln for ln in code.splitlines()
        if ln.strip() not in {"__END_SENTINEL__", "X"}
    ).strip()

    # Validate: must start with `def ` and be syntactically valid Python.
    if not code.startswith("def ") or not _is_valid_python(code):
        return buggy_code.strip()

    return code


# ===== Convenience builders for common configs =====

def create_deterministic_config(**kwargs) -> ModelConfig:
    """
    Create a deterministic (non-sampling) configuration.

    Examples
    --------
    >>> cfg = create_deterministic_config(max_new_tokens=512)
    """
    return ModelConfig(do_sample=False, **kwargs)


def create_sampling_config(temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> ModelConfig:
    """
    Create a sampling configuration using temperature + nucleus sampling.

    Parameters
    ----------
    temperature : float
        Higher values => more random output (typical range 0.3–1.0).
    top_p : float
        Nucleus cutoff (typical range 0.8–0.95).

    Examples
    --------
    >>> cfg = create_sampling_config(temperature=0.6, top_p=0.9, max_new_tokens=256)
    """
    return ModelConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )


def create_beam_search_config(num_beams: int = 3, **kwargs) -> ModelConfig:
    """
    Create a beam-search configuration (still deterministic).

    Parameters
    ----------
    num_beams : int
        Number of beams to expand; 3–5 is common for short outputs.

    Examples
    --------
    >>> cfg = create_beam_search_config(num_beams=4, max_new_tokens=256)
    """
    return ModelConfig(
        do_sample=False,
        num_beams=num_beams,
        early_stopping=True,
        **kwargs
    )
