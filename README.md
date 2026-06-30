# Flash-Attention GPT-2: Custom CUDA Kernels vs PyTorch

GPT-2 inference with a **switchable attention backend**, so you can directly
compare the PyTorch baseline against hand-written CUDA flash-attention kernels
on an identical KV cache. The cache, weights, and everything outside attention
are shared between the two paths, so the benchmark isolates the attention
compute.

- `backend="torch"` — `F.scaled_dot_product_attention` (PyTorch SDPA baseline)
- `backend="cuda"`  — custom `flash_prefill_causal` + `flash_decode` CUDA kernels

Weights are loaded from the HuggingFace GPT-2 checkpoints in fp32.

> Requires an NVIDIA GPU + CUDA toolkit. The CUDA extension is JIT-compiled on
> first import via `torch.utils.cpp_extension.load`. Designed to be pulled and
> run in **Google Colab Pro** (GPU runtime).

## Repository layout / task division

| File | Responsibility |
| --- | --- |
| `kvcache_kernels.cu` | The CUDA part: `flash_prefill_causal`, `flash_decode`, `append_kv`, online-softmax flash attention + pybind11 bindings |
| `cuda_gpt2.py` | Model + integration: GPT-2 modules, KV cache, switchable backend, HF weight loading |
| `benchmark.py` | Validation + throughput: logit correctness check, generation throughput, decode-step latency |
| `profile_forward.py` | Profiling: forward-pass (prefill + decode) latency torch vs cuda, CSV + plots |
| `colab_run.ipynb` | One-click Colab Pro runner |
| `requirements.txt` | Python dependencies |

## Quick start (Colab Pro)

Set the runtime to **GPU** (Runtime → Change runtime type → GPU), then:

```bash
git clone https://github.com/sudhirpol522/Flash-Attention.git
cd Flash-Attention
pip install -r requirements.txt

# 1) Correctness + throughput (run this FIRST, confirm PASS)
python benchmark.py --model gpt2 --batch 8 --prompt_len 256 --new_tokens 128

# 2) Forward-pass profiling (writes results.csv + plots/)
python profile_forward.py --model gpt2 --batch 8 --prompt_lens 128 256 512
```

Or just open `colab_run.ipynb` in Colab and run all cells.

The first `cuda`-backend call triggers the one-time JIT compile of
`kvcache_kernels.cu` (takes ~1-2 min); subsequent runs are cached.

## What the profiler produces

- `results.csv` — raw prefill/decode latency per `(prompt_len, backend)`
- `plots/latency_prefill.png`, `plots/latency_decode.png` — grouped bars, torch vs cuda
- `plots/speedup.png` — torch/cuda speedup vs prompt length

## Kernel notes

- GPT-2 head dimension is 64, so each block uses `blockDim.x == head_dim` and a
  key-tile size `BK = 64`.
- Both kernels use the standard flash-attention **online softmax** (running max
  `m`, running denominator `l`, rescaled accumulator) so they are numerically
  stable and never materialize the full attention matrix.
- `flash_prefill_causal`: grid `(B*nh, T)`, each query `qi` attends to keys
  `0..qi` (causal).
- `flash_decode`: grid `(B*nh)`, the single query attends to all `seqlen`
  cached keys.

Correctness is checked against PyTorch SDPA in `benchmark.py` /
`profile_forward.py` (max abs logit difference well under the fp32 tolerance).
