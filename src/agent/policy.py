# --- in src/agent/policy.py ---
from __future__ import annotations
import os, ast, re
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Model config =====
MODEL_ID = os.getenv("MODEL_DIR", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.05
SEED = 42

SYSTEM = (
    "You are a careful Python bug fixer. Produce the MINIMAL patch. "
    "Keep the same function signature and behavior unless tests require otherwise. "
    "Avoid I/O, networking, and randomness."
)

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None

def _seed_everything(seed: int = SEED) -> None:
    try:
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)

def load_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        _seed_everything(SEED)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        # ensure pad token is set
        if _model.config.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _model.config.pad_token_id = _tokenizer.eos_token_id
        # neutralize any sampling defaults in generation_config (avoid warnings)
        gc = _model.generation_config
        gc.do_sample = False; gc.temperature = None; gc.top_p = None; gc.top_k = None
        if hasattr(gc, "typical_p"): gc.typical_p = None
        if hasattr(gc, "penalty_alpha"): gc.penalty_alpha = None
    return _tokenizer, _model

def _build_user_prompt(buggy_code: str, failure_summary: str, func_name: Optional[str]) -> str:
    name_rule = f"`{func_name}`" if func_name else "the same name as in the buggy code"
    return (
        "You will receive a buggy Python function and a failure summary from executing tests.\n"
        "GOAL: Produce a MINIMAL patch that makes the tests pass.\n\n"
        "Rules:\n"
        f"- Output EXACTLY ONE Python function named {name_rule} with the SAME SIGNATURE as in the buggy code.\n"
        "- Do NOT add helper functions, classes, imports, prints, logging, annotations, or comments.\n"
        "- Do NOT change parameter names/order or the public API; preserve return TYPE and SHAPE.\n"
        "- Prefer fixing a small detail (bounds, off-by-one, initial toggle) over refactoring.\n"
        "- Infer the intended behavior from the failing assertion(s). Mentally verify on the shown inputs.\n"
        "- For well-known names (e.g., fib), assume canonical definitions unless tests indicate otherwise.\n"
        "- No I/O, no networking, no randomness.\n"
        "- Output ONLY raw Python source (no Markdown/backticks/prose).\n\n"
        "### Buggy function:\n"
        f"{buggy_code}\n\n"
        "### Failure summary (may be empty):\n"
        f"{failure_summary}\n\n"
        "### Output:\n"
        "Start with `def ` and emit only the corrected function."
    )

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
    gen_cfg: Optional[dict] = None,
) -> str:
    """
    Return ONLY the corrected function source (no fences, no prose).
    gen_cfg: {do_sample: bool, temperature, top_p, top_k, max_new_tokens}
    """
    gen_cfg = gen_cfg or {}
    tok, model = load_llm()

    user = _build_user_prompt(buggy_code, failure_summary, func_name)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt")
    # move to device (GPU on Kaggle)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = bool(gen_cfg.get("do_sample", False))
    gen_args = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", MAX_NEW_TOKENS),
        "do_sample": do_sample,
        "repetition_penalty": REPETITION_PENALTY,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
    }
    if do_sample:
        if gen_cfg.get("temperature") is not None: gen_args["temperature"] = gen_cfg["temperature"]
        if gen_cfg.get("top_p") is not None:       gen_args["top_p"] = gen_cfg["top_p"]
        if gen_cfg.get("top_k") is not None:       gen_args["top_k"] = gen_cfg["top_k"]

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_args)

    gen_text = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = _extract_function(gen_text, func_name=func_name).strip()
    code = "\n".join(ln for ln in code.splitlines() if ln.strip() not in {"__END_SENTINEL__", "X"}).strip()

    if not code.startswith("def ") or not _is_valid_python(code):
        return buggy_code.strip()
    return code


