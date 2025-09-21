# src/agent/policy.py
"""
Policy for generating minimal Python bug fixes using Transformers (CPU).
- Configurable generation parameters for flexible model behavior
- Robust extraction of a single function from LLM output.
- Safe fallback to buggy code if parsing/extraction fails.

If you see a NumPy warning from PyTorch (NumPy 2.x vs Torch built on 1.x),
either `pip install "numpy<2.0"` or upgrade Torch to a NumPy-2-compatible build.
"""

from __future__ import annotations
import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Default Model config =====
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Transformers checkpoint
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_SEED = 42

@dataclass
class ModelConfig:
    """Configuration for the language model and generation parameters."""
    # Model settings
    model_id: str = DEFAULT_MODEL_ID
    device: str = DEFAULT_DEVICE
    dtype: torch.dtype = DEFAULT_DTYPE
    seed: int = DEFAULT_SEED
    
    # Generation parameters
    max_new_tokens: int = 1024
    do_sample: bool = False  # Set to True for sampling-based generation
    temperature: Optional[float] = None  # Only used if do_sample=True
    top_p: Optional[float] = None  # Only used if do_sample=True
    top_k: Optional[int] = None  # Only used if do_sample=True
    repetition_penalty: float = 1.05
    
    # Additional generation parameters
    num_beams: int = 1  # For beam search
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.do_sample and self.temperature is None:
            self.temperature = 1.0  # Default temperature for sampling
        if not self.do_sample and self.temperature is not None:
            print("Warning: temperature is set but do_sample=False. Temperature will be ignored.")

# ===== Prompt config =====
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
    try:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def load_llm(config: Optional[ModelConfig] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load and cache tokenizer/model with the given configuration."""
    global _tokenizer, _model, _current_config
    
    if config is None:
        config = ModelConfig()
    
    # Check if we need to reload due to config change
    need_reload = (
        _model is None or 
        _tokenizer is None or 
        _current_config is None or
        _current_config.model_id != config.model_id or
        _current_config.device != config.device or
        _current_config.dtype != config.dtype
    )
    
    if need_reload:
        _seed_everything(config.seed)
        _tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device,
        )
        # Ensure pad token is set to eos if missing (prevents warnings)
        if _model.config.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _model.config.pad_token_id = _tokenizer.eos_token_id
        
        _current_config = config
    
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
    config: Optional[ModelConfig] = None,
) -> str:
    """
    Generate ONLY the corrected function source (no fences, no prose).
    If extraction or parsing fails, return `buggy_code` so imports don't error.
    
    Args:
        buggy_code: The buggy function code to fix
        failure_summary: Summary of test failures
        func_name: Name of the function to fix
        config: Model configuration (uses default if None)
    """
    if config is None:
        config = ModelConfig()
    
    tok, model = load_llm(config)

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
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build generation kwargs from config
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
    
    # Add sampling parameters only if do_sample is True
    if config.do_sample:
        if config.temperature is not None:
            gen_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            gen_kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k
    else:
        # For deterministic generation, explicitly set these to None
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_text = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = _extract_function(gen_text, func_name=func_name).strip()

    # Final sanitization (remove any stray sentinels if they ever appear)
    code = "\n".join(ln for ln in code.splitlines() if ln.strip() not in {"__END_SENTINEL__", "X"}).strip()

    # Validate and fallback if needed
    if not code.startswith("def ") or not _is_valid_python(code):
        return buggy_code.strip()

    return code


# Convenience functions for common configurations
def create_deterministic_config(**kwargs) -> ModelConfig:
    """Create a deterministic configuration (greedy decoding)."""
    return ModelConfig(do_sample=False, **kwargs)


def create_sampling_config(temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> ModelConfig:
    """Create a sampling configuration with temperature and top-p."""
    return ModelConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )


def create_beam_search_config(num_beams: int = 3, **kwargs) -> ModelConfig:
    """Create a beam search configuration."""
    return ModelConfig(
        do_sample=False,
        num_beams=num_beams,
        early_stopping=True,
        **kwargs
    )