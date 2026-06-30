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

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

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
// flash_prefill_causal_v2_kernel  (FA2 query-outer schedule)
//
// Q, K, V, O : (B, nh, N, d)
// The query tile lives in the grid (blockIdx.z), so parallelism grows with N.
// Each block keeps its tile's m / l / O accumulator in shared memory across the
// whole key loop (zero HBM round-trips), loops only over key tiles, and stops
// at the diagonal for causal masking (no wasted upper-triangle work).
// ---------------------------------------------------------------------------
template <const int Br, const int Bc>
__global__ void flash_prefill_causal_v2_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, int N, int d, int Tc,
    float scale, float* __restrict__ O)
{
    int tx = threadIdx.x;                  // Br*Bc threads
    int bx = blockIdx.x, by = blockIdx.y;  // batch, head
    int i  = blockIdx.z;                   // QUERY TILE  <-- now in the grid

    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);

    extern __shared__ float smem[];
    float* Qi   = smem;                    // Br*d
    float* Kj   = Qi + Br * d;             // Bc*d
    float* Vj   = Kj + Bc * d;             // Bc*d
    float* Sij  = Vj + Bc * d;             // Br*Bc
    float* Oi   = Sij + Br * Bc;           // Br*d   (accumulator, stays in smem)
    float* mi   = Oi + Br * d;             // Br
    float* li   = mi + Br;                 // Br
    float* corr = li + Br;                 // Br

    int s_row = tx / Bc, s_col = tx % Bc;

    for (int e = tx; e < Br * d; e += Br * Bc) {
        int row = e / d, col = e % d, gr = i * Br + row;
        Qi[e] = (gr < N) ? Q[qkv_off + gr * d + col] : 0.f;
        Oi[e] = 0.f;
    }
    if (tx < Br) { mi[tx] = -INFINITY; li[tx] = 0.f; }
    __syncthreads();

    int q_last = min(i * Br + Br - 1, N - 1);
    int j_max  = q_last / Bc;              // causal: never touch future key tiles

    for (int j = 0; j <= j_max; j++) {
        for (int e = tx; e < Bc * d; e += Br * Bc) {
            int row = e / d, col = e % d, gk = j * Bc + row;
            Kj[e] = (gk < N) ? K[qkv_off + gk * d + col] : 0.f;
            Vj[e] = (gk < N) ? V[qkv_off + gk * d + col] : 0.f;
        }
        __syncthreads();

        int q_pos = i * Br + s_row, k_pos = j * Bc + s_col;
        float acc = 0.f;
        for (int k = 0; k < d; k++) acc += Qi[s_row * d + k] * Kj[s_col * d + k];
        acc *= scale;
        if (k_pos > q_pos || k_pos >= N) acc = -INFINITY;   // causal mask in diagonal tile
        Sij[s_row * Bc + s_col] = acc;
        __syncthreads();

        if (s_col == 0) {                                   // online softmax, one thread/row
            float m_old = mi[s_row], m_tile = -INFINITY;
            for (int c = 0; c < Bc; c++) m_tile = fmaxf(m_tile, Sij[s_row * Bc + c]);
            float m_new = fmaxf(m_old, m_tile);
            float c_old = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
            float l_tile = 0.f;
            for (int c = 0; c < Bc; c++) {
                float p = (Sij[s_row * Bc + c] == -INFINITY) ? 0.f
                                                             : __expf(Sij[s_row * Bc + c] - m_new);
                Sij[s_row * Bc + c] = p;
                l_tile += p;
            }
            li[s_row]   = li[s_row] * c_old + l_tile;
            mi[s_row]   = m_new;
            corr[s_row] = c_old;
        }
        __syncthreads();

        for (int col = s_col; col < d; col += Bc) {         // Oi = corr*Oi + P·Vj
            float pv = 0.f;
            for (int c = 0; c < Bc; c++) pv += Sij[s_row * Bc + c] * Vj[c * d + col];
            Oi[s_row * d + col] = Oi[s_row * d + col] * corr[s_row] + pv;
        }
        __syncthreads();
    }

    for (int col = s_col; col < d; col += Bc) {             // write once, normalized
        int gr = i * Br + s_row;
        if (gr < N) {
            float denom = (li[s_row] > 0.f) ? li[s_row] : 1.f;
            O[qkv_off + gr * d + col] = Oi[s_row * d + col] / denom;
        }
    }
}
// launch: grid(B, nh, Tr), block(Br*Bc),
//         smem = (Br*d + 2*Bc*d + Br*Bc + Br*d + 3*Br) * sizeof(float)

// ===========================================================================
// Host wrappers (Python-facing)
// ===========================================================================

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

// Q, K, V : (B, nh, N, hd) contiguous fp32 CUDA -> O : (B, nh, N, hd)
// FA2 query-outer schedule: grid is (B, nh, Tr) so parallelism grows with N.
torch::Tensor flash_prefill_causal(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_F32(Q);  CHECK_F32(K);  CHECK_F32(V);
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    const int Br = 16, Bc = 16;
    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);
    int Tr = CEIL_DIV(N, Br), Tc = CEIL_DIV(N, Bc);

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / std::sqrt((float)d);
    size_t smem = (size_t)(Br * d + 2 * Bc * d + Br * Bc + Br * d + 3 * Br) * sizeof(float);

    dim3 grid(B, nh, Tr), block(Br * Bc);
    flash_prefill_causal_v2_kernel<Br, Bc><<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, scale, O.data_ptr<float>());
    return O;
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
