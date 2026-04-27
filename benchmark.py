"""
Qwen2.5-7B-Instruct Quantization Benchmark
==========================================
Compares BF16 vs INT8 vs NF4 4-bit across multiple prompts.

Usage:
    python benchmark.py

Results are saved to results/results.json and results/benchmark_results.png.
"""
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single A6000 (47.4 GB) is sufficient for 7B

import torch

from models import load_bf16, load_int8, load_nf4, load_tokenizer
from utils import plot_results, run_inference, save_results

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 200

PROMPTS = [
    "Explain neural network quantization in simple terms.",
    "Describe the trade-off between memory usage, latency, and output quality"
    " in deployment-oriented machine learning systems.",
    "You are optimizing a model for real-time perception on an edge device."
    " Explain why quantization, profiling, and latency measurement matter.",
]

# (label, loader_function)
CONFIGS = [
    ("BF16",      load_bf16),
    ("INT8",      load_int8),
    ("NF4 4-bit", load_nf4),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PyTorch : {torch.__version__}")
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model   : {MODEL_ID}")

    tokenizer = load_tokenizer(MODEL_ID)
    all_results = []

    for label, loader in CONFIGS:
        print_section(f"Loading {label}")
        model, load_time, vram_gb = loader(MODEL_ID)
        print(f"Load time : {load_time:.1f}s")
        print(f"VRAM      : {vram_gb:.2f} GB")

        prompt_results = []
        for i, prompt in enumerate(PROMPTS, 1):
            print(f"\n--- Prompt {i}/{len(PROMPTS)} ---")
            print(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            text = format_prompt(tokenizer, prompt)
            result = run_inference(model, tokenizer, text, MAX_NEW_TOKENS)

            print(f"  Tokens    : {result['n_tokens']}")
            print(f"  Elapsed   : {result['elapsed_s']:.2f}s")
            print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
            print(f"  Response  :\n    {result['response'][:200]}...")

            prompt_results.append({
                "prompt": prompt,
                **{k: v for k, v in result.items() if k != "response"},
                "response_preview": result["response"][:300],
            })

        all_results.append({
            "precision": label,
            "load_time_s": load_time,
            "vram_gb": vram_gb,
            "prompts": prompt_results,
        })

        del model
        torch.cuda.empty_cache()

    save_results(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
