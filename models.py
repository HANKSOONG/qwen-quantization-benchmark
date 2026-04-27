"""
Model loading helpers for BF16, INT8, and NF4 4-bit quantization.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_id)


def _load(model_id: str, **kwargs) -> tuple:
    """Internal: load model, return (model, load_time_s, vram_gb)."""
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    load_time = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated(0) / 1024**3
    return model, load_time, vram


def load_bf16(model_id: str) -> tuple:
    """Load model in BF16 (full precision baseline)."""
    return _load(model_id, dtype=torch.bfloat16, device_map="cuda:0")


def load_int8(model_id: str) -> tuple:
    """Load model with bitsandbytes 8-bit quantization (LLM.int8())."""
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    return _load(model_id, quantization_config=cfg, device_map="cuda:0")


def load_nf4(model_id: str) -> tuple:
    """
    Load model with 4-bit NF4 quantization (QLoRA scheme).

    NF4 (NormalFloat4) uses a non-uniform 4-bit representation optimized
    for weights that follow a normal distribution.
    Double quantization further reduces memory by also quantizing the
    quantization scale factors (~0.4 extra bits/param saving).
    """
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to BF16 for compute
        bnb_4bit_use_double_quant=True,
    )
    return _load(model_id, quantization_config=cfg, device_map="cuda:0")
