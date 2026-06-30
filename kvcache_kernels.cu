// kvcache_kernels.cu
// Custom CUDA flash-attention kernels for the switchable-backend GPT-2.
//
// Exposed to Python (via torch.utils.cpp_extension.load) as:
//   ext.flash_prefill_causal(q, k, v)               -> y     (B, nh, T, hd)
//   ext.flash_decode(q, k_cache, v_cache, seqlen)   -> y     (B, nh, hd)
//   ext.append_kv(new_k, new_v, k_cache, v_cache, context_lens)  (utility)
//
// All tensors are fp32, CUDA, contiguous. GPT-2 head_dim == 64.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Key tile size for the online-softmax loop. head_dim is 64 for the whole
// GPT-2 family, so a single warp-pair block (blockDim.x == head_dim) is used.
static constexpr int BK = 64;

// ---------------------------------------------------------------------------
// append_kv (utility; the Python path writes the cache directly, but this is
// kept available for completeness / alternative cache-management paths)
//
// new_k, new_v : (B, nh, d)               one token per sequence
// k_cache, v_cache : (B, nh, max_len, d)
// context_lens : (B,)                     length BEFORE this append
// ---------------------------------------------------------------------------
__global__ void append_kv(
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ context_lens,
    int max_len, int d)
{
    int b = blockIdx.x, h = blockIdx.y;
    int pos = context_lens[b];
    int src = (b * gridDim.y + h) * d;
    int dst = ((b * gridDim.y + h) * max_len + pos) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) {
        k_cache[dst + e] = new_k[src + e];
        v_cache[dst + e] = new_v[src + e];
    }
}
// launch: grid(B, nh), block(min(d, 256))

// ---------------------------------------------------------------------------
// flash_decode_kernel
//
// q : (B*nh, d)
// k_cache, v_cache : (B*nh, max_len, d)
// o : (B*nh, d)
// blockDim.x == d  (GPT-2: 64);  one block per (B*nh)
// ---------------------------------------------------------------------------
template <const int BK_>            // key tile size, e.g. 64
__global__ void flash_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ o,
    int seq_len, int max_len, int d, float scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;                 // owns output dim `tid`

    const float* qp = q + (size_t)bh * d;
    const float* kp = k_cache + (size_t)bh * max_len * d;
    const float* vp = v_cache + (size_t)bh * max_len * d;

    extern __shared__ float smem[];
    float* q_s = smem;                     // d
    float* s_s = q_s + d;                  // BK_

    q_s[tid] = qp[tid];
    __syncthreads();

    float m_i = -INFINITY, l_i = 0.f, acc = 0.f;

    for (int j0 = 0; j0 < seq_len; j0 += BK_) {
        // scores for this key tile
        for (int jj = tid; jj < BK_; jj += blockDim.x) {
            int j = j0 + jj;
            float dot = -INFINITY;
            if (j < seq_len) {
                dot = 0.f;
                const float* kj = kp + (size_t)j * d;
                for (int e = 0; e < d; e++) dot += q_s[e] * kj[e];
                dot *= scale;
            }
            s_s[jj] = dot;
        }
        __syncthreads();

        // online-softmax rescale + V accumulation
        float m_tile = -INFINITY;
        for (int jj = 0; jj < BK_; jj++) m_tile = fmaxf(m_tile, s_s[jj]);
        float m_new = fmaxf(m_i, m_tile);
        float corr  = __expf(m_i - m_new);
        acc *= corr;

        float l_tile = 0.f;
        for (int jj = 0; jj < BK_; jj++) {
            int j = j0 + jj;
            if (j < seq_len) {
                float p = __expf(s_s[jj] - m_new);
                l_tile += p;
                acc += p * vp[(size_t)j * d + tid];
            }
        }
        l_i = l_i * corr + l_tile;
        m_i = m_new;
        __syncthreads();
    }
    o[(size_t)bh * d + tid] = acc / l_i;
}
// launch: grid(B*nh), block(d), smem = (d + BK) * sizeof(float)

// ---------------------------------------------------------------------------
// flash_prefill_causal_kernel
//
// q, k, v : (B*nh, T, d)
// o : (B*nh, T, d)
// One block per (bh, query index qi); blockDim.x == d. Each query attends to
// keys 0..qi (causal). Same online-softmax math as decode.
// ---------------------------------------------------------------------------
template <const int BK_>
__global__ void flash_prefill_causal_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int T, int d, float scale)
{
    int bh = blockIdx.x;
    int qi = blockIdx.y;                    // query position this block owns
    int tid = threadIdx.x;                 // owns output dim `tid`

    const float* qp = q + ((size_t)bh * T + qi) * d;
    const float* kp = k + (size_t)bh * T * d;
    const float* vp = v + (size_t)bh * T * d;

    extern __shared__ float smem[];
    float* q_s = smem;                     // d
    float* s_s = q_s + d;                  // BK_

    q_s[tid] = qp[tid];
    __syncthreads();

    float m_i = -INFINITY, l_i = 0.f, acc = 0.f;
    int seq = qi + 1;                      // causal: attend to keys 0..qi

    for (int j0 = 0; j0 < seq; j0 += BK_) {
        for (int jj = tid; jj < BK_; jj += blockDim.x) {
            int j = j0 + jj;
            float dot = -INFINITY;
            if (j < seq) {
                dot = 0.f;
                const float* kj = kp + (size_t)j * d;
                for (int e = 0; e < d; e++) dot += q_s[e] * kj[e];
                dot *= scale;
            }
            s_s[jj] = dot;
        }
        __syncthreads();

        float m_tile = -INFINITY;
        for (int jj = 0; jj < BK_; jj++) m_tile = fmaxf(m_tile, s_s[jj]);
        float m_new = fmaxf(m_i, m_tile);
        float corr  = __expf(m_i - m_new);
        acc *= corr;

        float l_tile = 0.f;
        for (int jj = 0; jj < BK_; jj++) {
            int j = j0 + jj;
            if (j < seq) {
                float p = __expf(s_s[jj] - m_new);
                l_tile += p;
                acc += p * vp[(size_t)j * d + tid];
            }
        }
        l_i = l_i * corr + l_tile;
        m_i = m_new;
        __syncthreads();
    }
    o[((size_t)bh * T + qi) * d + tid] = acc / l_i;
}
// launch: grid(B*nh, T), block(d), smem = (d + BK) * sizeof(float)

// ===========================================================================
// Host wrappers (Python-facing)
// ===========================================================================

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

// q, k, v : (B, nh, T, hd) contiguous fp32 CUDA -> y : (B, nh, T, hd)
torch::Tensor flash_prefill_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v);
    CHECK_F32(q);  CHECK_F32(k);  CHECK_F32(v);
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    int B  = q.size(0);
    int nh = q.size(1);
    int T  = q.size(2);
    int d  = q.size(3);
    TORCH_CHECK(d == BK, "head_dim must equal BK (", BK, "); got ", d);

    auto o = torch::empty_like(q);
    float scale = 1.0f / std::sqrt((float)d);

    dim3 grid(B * nh, T);
    dim3 block(d);
    size_t smem = (size_t)(d + BK) * sizeof(float);

    flash_prefill_causal_kernel<BK><<<grid, block, smem>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        o.data_ptr<float>(), T, d, scale);
    return o;
}

// q : (B, nh, hd);  k_cache/v_cache : (B, nh, max_len, hd) -> y : (B, nh, hd)
torch::Tensor flash_decode(torch::Tensor q, torch::Tensor k_cache,
                           torch::Tensor v_cache, int64_t seqlen) {
    CHECK_CUDA(q); CHECK_CUDA(k_cache); CHECK_CUDA(v_cache);
    CHECK_F32(q);  CHECK_F32(k_cache);  CHECK_F32(v_cache);
    q = q.contiguous(); k_cache = k_cache.contiguous(); v_cache = v_cache.contiguous();

    int B       = q.size(0);
    int nh      = q.size(1);
    int d       = q.size(2);
    int max_len = k_cache.size(2);
    TORCH_CHECK(d == BK, "head_dim must equal BK (", BK, "); got ", d);

    auto o = torch::empty({B, nh, d}, q.options());
    float scale = 1.0f / std::sqrt((float)d);

    dim3 grid(B * nh);
    dim3 block(d);
    size_t smem = (size_t)(d + BK) * sizeof(float);

    flash_decode_kernel<BK><<<grid, block, smem>>>(
        q.data_ptr<float>(), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        o.data_ptr<float>(), (int)seqlen, max_len, d, scale);
    return o;
}

// Utility binding: append a single token's K/V into the cache.
// new_k/new_v : (B, nh, hd); k_cache/v_cache : (B, nh, max_len, hd)
// context_lens : (B,) int32
void append_kv_cuda(torch::Tensor new_k, torch::Tensor new_v,
                    torch::Tensor k_cache, torch::Tensor v_cache,
                    torch::Tensor context_lens) {
    CHECK_CUDA(new_k); CHECK_CUDA(new_v); CHECK_CUDA(k_cache); CHECK_CUDA(v_cache);
    int B       = new_k.size(0);
    int nh      = new_k.size(1);
    int d       = new_k.size(2);
    int max_len = k_cache.size(2);

    dim3 grid(B, nh);
    dim3 block(d < 256 ? d : 256);
    append_kv<<<grid, block>>>(
        new_k.data_ptr<float>(), new_v.data_ptr<float>(),
        k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
        context_lens.data_ptr<int>(), max_len, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_prefill_causal", &flash_prefill_causal,
          "Causal flash-attention prefill (fp32)");
    m.def("flash_decode", &flash_decode,
          "Flash-attention single-token decode over KV cache (fp32)");
    m.def("append_kv", &append_kv_cuda,
          "Append one token's K/V into the cache (fp32)");
}
