"""
benchmark.py
Validate the custom kernels against PyTorch and benchmark generation throughput.

Usage:
    python benchmark.py --model gpt2 --batch 8 --prompt_len 256 --new_tokens 128

Run this FIRST and confirm the correctness check passes before trusting timings.
"""
import argparse
import time

import torch

from cuda_gpt2 import GPT


def cuda_time(fn, iters=20, warmup=5):
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
    return start.elapsed_time(end) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--atol", type=float, default=2e-3)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA device required"
    device = "cuda"
    torch.manual_seed(0)

    print(f"Loading {args.model} ...")
    model = GPT.from_pretrained(args.model).to(device).eval()
    V = model.cfg.vocab_size

    idx = torch.randint(0, V, (args.batch, args.prompt_len), device=device)

    # ---- Correctness: compare logits of torch vs cuda over prefill + decode --
    print("\n== Correctness check ==")
    caches_t = model._alloc_cache(args.batch, args.prompt_len + 4, device, torch.float32)
    caches_c = model._alloc_cache(args.batch, args.prompt_len + 4, device, torch.float32)

    with torch.no_grad():
        lt = model.forward_step(idx, caches_t, 0, "torch")
        lc = model.forward_step(idx, caches_c, 0, "cuda")
    prefill_err = (lt - lc).abs().max().item()
    print(f"prefill  max|Δlogits| = {prefill_err:.3e}")

    # one decode step from the same cache state
    nt = lt[:, -1:].argmax(-1)
    with torch.no_grad():
        dt = model.forward_step(nt, caches_t, args.prompt_len, "torch")
        dc = model.forward_step(nt, caches_c, args.prompt_len, "cuda")
    decode_err = (dt - dc).abs().max().item()
    print(f"decode   max|Δlogits| = {decode_err:.3e}")

    ok = prefill_err < 5e-2 and decode_err < 5e-2  # fp32 flash vs sdpa tolerance
    print("RESULT:", "PASS" if ok else "FAIL (inspect kernels before trusting timings)")

    # ---- Timing ------------------------------------------------------------
    print("\n== Throughput ==")
    results = {}
    for backend in ["torch", "cuda"]:
        gen = lambda: model.generate(idx, args.new_tokens, backend=backend)
        ms = cuda_time(gen, iters=10, warmup=3)
        toks = args.batch * args.new_tokens
        results[backend] = ms
        print(f"{backend:>5}: {ms:8.2f} ms/gen   "
              f"{ms / args.new_tokens:7.3f} ms/token   "
              f"{toks / (ms / 1e3):9.1f} tok/s")

    if "torch" in results and "cuda" in results:
        spd = results["torch"] / results["cuda"]
        print(f"\nspeedup (torch/cuda): {spd:.2f}x")

    # ---- Decode-only microbenchmark (where the KV cache matters most) ------
    print("\n== Decode-step latency (single token, full cache) ==")
    for backend in ["torch", "cuda"]:
        caches = model._alloc_cache(args.batch, args.prompt_len + 1, device, torch.float32)
        with torch.no_grad():
            model.forward_step(idx, caches, 0, backend)  # warm cache
        tok = torch.randint(0, V, (args.batch, 1), device=device)
        step = lambda: model.forward_step(tok, caches, args.prompt_len, backend)
        ms = cuda_time(step, iters=50, warmup=10)
        print(f"{backend:>5}: {ms:7.3f} ms/step")


if __name__ == "__main__":
    main()
