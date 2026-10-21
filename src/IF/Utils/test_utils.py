from __future__ import annotations


import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from functools import wraps
import time


@dataclass
class CaseResult:
    name: str
    success: bool
    elapsed_s: float
    error: Optional[str]


def _init_distributed() -> tuple[bool, int, torch.device, bool]:
    use_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if use_distributed else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_master = (not use_distributed) or dist.get_rank() == 0
    return use_distributed, local_rank, device, is_master


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in lookup:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = lookup[dtype_name]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _load_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    device_map: str | Dict[str, str] | None = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    try:
        from utils import load_base_model  # type: ignore
    except Exception:
        load_base_model = None

    if load_base_model is not None:
        model, tokenizer = load_base_model(model_key, device=device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model, tokenizer

    if not allow_hf_loader:
        raise RuntimeError(
            "utils.load_base_model not available. "
            "Set --project-dir to include it or pass --allow-hf-loader."
        )"""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map
    )
    if device_map is None:
        model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def _prepare_inputs(tokenizer, prompt: str, device: torch.device):
    enc = tokenizer(prompt, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    if "attention_mask" in enc:
        labels = labels.masked_fill(enc["attention_mask"].eq(0), -100)
    return enc, labels


def _uses_full_backward_hooks(gm_kwargs: Dict[str, object]) -> bool:
    return bool(gm_kwargs.get("small_matrices", False))


def _load_model_and_inputs(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    train_prompt: str,
    eval_prompt: str,
):
    model, tokenizer = _load_model(
        model_id,
        device=device,
        dtype=dtype,
    )
    enc_train, labels_train = _prepare_inputs(tokenizer, train_prompt, device)
    enc_eval, labels_eval = _prepare_inputs(tokenizer, eval_prompt, device)
    return model, tokenizer, enc_train, labels_train, enc_eval, labels_eval


def _cleanup_model(model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cross_entropy_for_lm(
    logits,
    labels,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
):
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    return F.cross_entropy(
        logits_flat,
        labels_flat,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def timedWrapper(func):
    @wraps(func)
    def timeFunc(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeFunc
