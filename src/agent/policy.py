# src/agent/policy.py
"""
Policy for generating minimal Python bug fixes using Transformers (CPU).
- Deterministic (greedy) decoding for reproducibility.
- Robust extraction of a single function from LLM output.
- Safe fallback to buggy code if parsing/extraction fails.

If you see a NumPy warning from PyTorch (NumPy 2.x vs Torch built on 1.x),
either `pip install "numpy<2.0"` or upgrade Torch to a NumPy-2-compatible build.
"""

from __future__ import annotations
import ast
import re
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Model config =====
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Transformers checkpoint
DEVICE = "cpu"                                  # keep CPU for local dev
DTYPE = torch.float32                           # CPU dtype
MAX_NEW_TOKENS = 1024                           # enough to re-emit function
REPETITION_PENALTY = 1.05                       # mild anti-repeat
SEED = 42                                       # reproducible

# ===== Prompt config =====
SYSTEM = (
    "You are a careful Python bug fixer. Produce the MINIMAL patch. "
    "Keep the same function signature and behavior unless tests require otherwise. "
    "Avoid I/O, networking, and randomness."
)

# Cached model/tokenizer
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def _seed_everything(seed: int = SEED) -> None:
    try:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def load_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load and cache tokenizer/model on CPU with correct pad token."""
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        _seed_everything(SEED)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        # Ensure pad token is set to eos if missing (prevents warnings)
        if _model.config.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _model.config.pad_token_id = _tokenizer.eos_token_id
    return _tokenizer, _model


def _extract_function(text: str, func_name: Optional[str] = None) -> str:
    """
    Extract a single Python function definition from model output.
    Strategy:
      1) Prefer fenced code blocks (```python ... ```).
      2) If func_name provided, capture that def block up to next top-level stmt.
      3) Else, first 'def ...' block.
      4) Else, if output starts with 'def ', return as-is.
    Strips any sentinel/junk lines if they appear.
    """
    # 1) Fenced code block
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    junk = {"__END_SENTINEL__", "X"}

    def strip_junk(s: str) -> str:
        return "\n".join(ln for ln in s.splitlines() if ln.strip() not in junk).strip()

    # 2) Named function capture
    if func_name:
        T = text + "\n__END_SENTINEL__"
        m = re.search(rf"(?ms)^(def\s+{re.escape(func_name)}\s*\(.*?)(?=^\S|\Z)", T)
        if m:
            return strip_junk(m.group(1))

    # 3) First def block
    m = re.search(r"(?ms)^(def\s+\w+\s*\(.*?)(?=^\S|\Z)", text + "\n__END_SENTINEL__")
    if m:
        return strip_junk(m.group(1))

    # 4) Output starts with def
    if text.lstrip().startswith("def "):
        return strip_junk(text)

    return strip_junk(text)


def _is_valid_python(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def generate_patch(
    buggy_code: str,
    failure_summary: str = "",
    func_name: Optional[str] = None,
) -> str:
    """
    Generate ONLY the corrected function source (no fences, no prose).
    If extraction or parsing fails, return `buggy_code` so imports don't error.
    """
    tok, model = load_llm()

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
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tok.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )

    gen_text = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = _extract_function(gen_text, func_name=func_name).strip()

    # Final sanitization (remove any stray sentinels if they ever appear)
    code = "\n".join(ln for ln in code.splitlines() if ln.strip() not in {"__END_SENTINEL__", "X"}).strip()

    # Validate and fallback if needed
    if not code.startswith("def ") or not _is_valid_python(code):
        return buggy_code.strip()

    return code
