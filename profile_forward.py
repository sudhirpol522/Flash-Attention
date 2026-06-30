"""
profile_forward.py
Profile the forward pass of GPT-2 with the PyTorch SDPA backend vs the custom
CUDA flash kernels, isolating the attention compute.

It sweeps a few prompt lengths and times the *forward pass* in two regimes:
  - prefill : a single forward_step over the whole prompt (T = prompt_len)
  - decode  : a single forward_step of one token against a full KV cache

For every config it reports torch vs cuda latency and the torch/cuda speedup,
runs a logit correctness gate first, then writes:
  - results.csv          (raw numbers, via pandas)
  - plots/latency_prefill.png, plots/latency_decode.png  (grouped bars)
  - plots/speedup.png    (speedup vs prompt_len for prefill & decode)

Usage:
    python profile_forward.py --model gpt2 --batch 8
    python profile_forward.py --model gpt2 --batch 4 --prompt_lens 128 256 512 1024
"""
import argparse
import os

import torch

from cuda_gpt2 import GPT

_HERE = os.path.dirname(os.path.abspath(__file__))


def cuda_time(fn, iters=50, warmup=10):
    """Median-free mean timing using CUDA events (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def correctness_gate(model, V, B, prompt_len, device):
    """Compare torch vs cuda logits over prefill + one decode step."""
    idx = torch.randint(0, V, (B, prompt_len), device=device)
    caches_t = model._alloc_cache(B, prompt_len + 4, device, torch.float32)
    caches_c = model._alloc_cache(B, prompt_len + 4, device, torch.float32)

    lt = model.forward_step(idx, caches_t, 0, "torch")
    lc = model.forward_step(idx, caches_c, 0, "cuda")
    prefill_err = (lt - lc).abs().max().item()

    nt = lt[:, -1:].argmax(-1)
    dt = model.forward_step(nt, caches_t, prompt_len, "torch")
    dc = model.forward_step(nt, caches_c, prompt_len, "cuda")
    decode_err = (dt - dc).abs().max().item()

    ok = prefill_err < 5e-2 and decode_err < 5e-2
    print("\n== Correctness gate ==")
    print(f"prefill  max|Δlogits| = {prefill_err:.3e}")
    print(f"decode   max|Δlogits| = {decode_err:.3e}")
    print("RESULT:", "PASS" if ok else "FAIL (inspect kernels before trusting timings)")
    return ok


@torch.no_grad()
def profile(model, V, B, prompt_lens, device, iters, warmup):
    rows = []
    for L in prompt_lens:
        idx = torch.randint(0, V, (B, L), device=device)
        tok = torch.randint(0, V, (B, 1), device=device)

        for backend in ["torch", "cuda"]:
            # --- prefill: forward over the whole prompt --------------------
            def prefill():
                caches = model._alloc_cache(B, L + 1, device, torch.float32)
                model.forward_step(idx, caches, 0, backend)

            ms_prefill = cuda_time(prefill, iters=max(iters // 5, 2),
                                   warmup=max(warmup // 2, 1))

            # --- decode: one token against a warmed, full cache ------------
            caches = model._alloc_cache(B, L + 1, device, torch.float32)
            model.forward_step(idx, caches, 0, backend)  # warm to length L

            def decode():
                model.forward_step(tok, caches, L, backend)

            ms_decode = cuda_time(decode, iters=iters, warmup=warmup)

            rows.append(dict(model=model_name, batch=B, prompt_len=L,
                             backend=backend,
                             prefill_ms=ms_prefill, decode_ms=ms_decode))
            print(f"[L={L:>5} {backend:>5}] prefill {ms_prefill:8.3f} ms   "
                  f"decode {ms_decode:7.3f} ms")
    return rows


def summarize_and_plot(rows, out_dir):
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_HERE, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # pivot to torch/cuda columns per prompt_len
    piv_pre = df.pivot_table(index="prompt_len", columns="backend",
                             values="prefill_ms")
    piv_dec = df.pivot_table(index="prompt_len", columns="backend",
                             values="decode_ms")

    print("\n== Forward-pass latency (ms) ==")
    for L in piv_pre.index:
        tp, cp = piv_pre.loc[L, "torch"], piv_pre.loc[L, "cuda"]
        td, cd = piv_dec.loc[L, "torch"], piv_dec.loc[L, "cuda"]
        print(f"L={L:>5} | prefill torch {tp:8.3f}  cuda {cp:8.3f}  "
              f"({tp / cp:4.2f}x) | decode torch {td:7.3f}  cuda {cd:7.3f}  "
              f"({td / cd:4.2f}x)")

    # --- plots ------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        def grouped_bar(piv, title, fname):
            Ls = list(piv.index)
            x = np.arange(len(Ls))
            w = 0.38
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - w / 2, piv["torch"].values, w, label="torch (SDPA)")
            ax.bar(x + w / 2, piv["cuda"].values, w, label="cuda (flash)")
            ax.set_xticks(x)
            ax.set_xticklabels([str(L) for L in Ls])
            ax.set_xlabel("prompt_len")
            ax.set_ylabel("latency (ms)")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            p = os.path.join(out_dir, fname)
            fig.savefig(p, dpi=120)
            plt.close(fig)
            print(f"Wrote {p}")

        grouped_bar(piv_pre, "Prefill forward latency: torch vs cuda",
                    "latency_prefill.png")
        grouped_bar(piv_dec, "Decode-step latency: torch vs cuda",
                    "latency_decode.png")

        # speedup curves
        fig, ax = plt.subplots(figsize=(7, 4))
        Ls = list(piv_pre.index)
        ax.plot(Ls, (piv_pre["torch"] / piv_pre["cuda"]).values,
                marker="o", label="prefill speedup")
        ax.plot(Ls, (piv_dec["torch"] / piv_dec["cuda"]).values,
                marker="s", label="decode speedup")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("prompt_len")
        ax.set_ylabel("speedup (torch / cuda)")
        ax.set_title("Speedup of custom CUDA flash vs PyTorch SDPA")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "speedup.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"Wrote {p}")
    except Exception as e:  # plotting is best-effort
        print(f"[warn] plotting skipped: {e}")


model_name = "gpt2"  # set in main(), referenced by profile()


def main():
    global model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt_lens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "plots"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    device = "cuda"
    torch.manual_seed(0)
    model_name = args.model

    print(f"Loading {args.model} ...")
    model = GPT.from_pretrained(args.model).to(device).eval()
    V = model.cfg.vocab_size

    correctness_gate(model, V, args.batch, args.prompt_lens[0], device)

    print("\n== Profiling forward pass (torch vs cuda) ==")
    rows = profile(model, V, args.batch, args.prompt_lens, device,
                   args.iters, args.warmup)
    summarize_and_plot(rows, args.out_dir)


if __name__ == "__main__":
    main()
