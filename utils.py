"""
Utility functions for VRAM measurement, inference timing, and result visualization.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def get_vram_gb(device: int = 0) -> float:
    """Return currently allocated VRAM on the given device in GB."""
    return torch.cuda.memory_allocated(device) / 1024**3


def run_inference(model, tokenizer, text: str, max_new_tokens: int = 200) -> dict:
    """
    Run a single greedy-decode inference pass and return timing + response.

    Args:
        model: loaded HuggingFace causal LM
        tokenizer: corresponding tokenizer
        text: already-formatted prompt string (after apply_chat_template)
        max_new_tokens: maximum tokens to generate

    Returns:
        dict with keys: response, n_tokens, elapsed_s, tokens_per_sec
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # warmup pass
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                            temperature=None, top_p=None, top_k=None)
    torch.cuda.synchronize()

    # timed pass
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_new = outputs.shape[1] - inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)
    return {
        "response": response,
        "n_tokens": n_new,
        "elapsed_s": elapsed,
        "tokens_per_sec": n_new / elapsed,
    }


def save_results(results: list[dict], out_dir: str = "results") -> None:
    """Save benchmark results to JSON."""
    Path(out_dir).mkdir(exist_ok=True)
    path = Path(out_dir) / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def plot_results(results: list[dict], out_dir: str = "results") -> None:
    """
    Build a summary DataFrame and save a 3-panel bar chart.

    Each entry in `results` has keys: precision, vram_gb, load_time_s,
    plus per-prompt entries under 'prompts'.
    """
    Path(out_dir).mkdir(exist_ok=True)

    rows = []
    for r in results:
        avg_tps = sum(p["tokens_per_sec"] for p in r["prompts"]) / len(r["prompts"])
        rows.append({
            "Precision": r["precision"],
            "VRAM (GB)": r["vram_gb"],
            "Avg Throughput (tok/s)": avg_tps,
            "Load Time (s)": r["load_time_s"],
        })
    df = pd.DataFrame(rows)
    print("\n--- Benchmark Summary ---")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cols = ["VRAM (GB)", "Avg Throughput (tok/s)", "Load Time (s)"]
    colors = ["steelblue", "seagreen", "salmon"]

    for ax, col, color in zip(axes, cols, colors):
        vals = df[col].astype(float)
        ax.bar(df["Precision"], vals, color=color)
        ax.set_title(col)
        ax.set_ylabel(col)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle("Qwen2.5-7B-Instruct: BF16 vs INT8 vs NF4", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = Path(out_dir) / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {out_path}")
