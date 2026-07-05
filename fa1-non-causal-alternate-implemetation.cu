//fa1 ha phila K and V chya matrix chya dimension vr stride krto which is nothing but along d dimension of K and V

// mental model ha ahe ki 

template <const int Br, const int Bc>
__global__ void flash_attn_1_kernel(const float* Q, const float* K, const float* V,
                                    int N, int d, int Tr, int Tc, float scale,
                                    float* l, float* m, float* O) {
    int tid = threadIdx.x;                       // Br*Bc threads
    int sample = blockIdx.x, head = blockIdx.y;

    int qkv_off = (N * d * gridDim.y * sample) + (N * d * head); // q, k and v che seprate memory initialisation ahe tyamule N * d  otherwise N * d * 3 asta
    int lm_off  = (N * gridDim.y * sample) + (N * head);

    extern __shared__ float smem[];
    float* Qi     = smem;
    float* Kj     = Qi  + Br * d;
    float* Vj     = Kj  + Bc * d;
    float* Sij    = Vj  + Bc * d;
    float* Oi     = Sij + Br * Bc;
    float* li     = Oi  + Br * d; //li he old sum ahe ata pryant jevdya tile houn gelya ahet tyanchya
    float* li_new = li + Br; //linew he new sum + old sum ahe (Imporatnt)
    float* mi     = li_new + Br; //mi he old max ahe ata pryant jevdya tile houn gelya ahet tyanchya
    float* mi_new = mi + Br; //mi_new he new max , old max ahe (Imporatnt) => global max
    float* mij    = mi_new + Br; //mij he local max ahe mhnje current tile max ahe

    int s_row = tid / Bc;
    int s_col = tid % Bc;

    for (int j = 0; j < Tc; j++) {
       
        for (int idx = tid; idx < Bc * d; idx += Br * Bc) {
            int row = idx / d, col = idx % d;
            bool valid = (j * Bc + row < N);
            Kj[row * d + col] = valid ? K[qkv_off + (j * Bc + row) * d + col] : 0.f; // jr out of bound gela tr 0 fill
            Vj[row * d + col] = valid ? V[qkv_off + (j * Bc + row) * d + col] : 0.f;
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {           
            for (int idx = tid; idx < Br * d; idx += Br * Bc) {
                int row = idx / d, col = idx % d;
                if (i * Br + row < N) {
                    Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
                    Oi[row * d + col] = O[qkv_off + (i * Br + row) * d + col];
                }
            }
            if (s_col == 0) {
                mi[s_row] = m[lm_off + i * Br + s_row];
                li[s_row] = l[lm_off + i * Br + s_row];
            }
            __syncthreads();

            // S = scale * Qi Kj^T, mask padded key columns
            float acc = 0.f;
            for (int k = 0; k < d; k++)
                acc += Qi[s_row * d + k] * Kj[s_col * d + k];
            Sij[s_row * Bc + s_col] = (j * Bc + s_col < N) ? acc * scale : -INFINITY;
            __syncthreads();                     // <-- required before row-wise softmax

            if (s_col == 0) {
                float row_m = -INFINITY, row_l = 0.f;
                for (int c = 0; c < Bc; c++)
                    row_m = fmaxf(row_m, Sij[s_row * Bc + c]);
                for (int c = 0; c < Bc; c++) {
                    float e = expf(Sij[s_row * Bc + c] - row_m);
                    Sij[s_row * Bc + c] = e;
                    row_l += e;
                }
                mij[s_row]    = row_m;
                mi_new[s_row] = fmaxf(mi[s_row], row_m);
                li_new[s_row] = expf(mi[s_row] - mi_new[s_row]) * li[s_row]
                              + expf(row_m - mi_new[s_row]) * row_l;
            }
            __syncthreads();


            //mental model ha ahe ki for BrxBc and BcXd matrix multiplication ahe tyamule phila output matrix vr thread lihayche which means lets say output matrix ha 2x4 ahe tr [[0,1,2,3],[4,5,6,7]] ahe , phila fkt thread 0 ha 0 element solve krnar, 1 ha 1 element solve krnar, 4th element output matrix cha thread 2 solve krnar and 5th element output matrix cha thread 3 solve krnar and so on. tyamule aaplyala next element sathi every thread Bc evdya distance ni pudhe stride krayla lagnar. tyamule every thread would calculate d/Bc evde elements
            
            int global_row = i * Br + s_row;
            for (int col = s_col; col < d; col += Bc) {
                float pv = 0.f;
                for (int c = 0; c < Bc; c++)
                    pv += Sij[s_row * Bc + c] * Vj[c * d + col];
                if (global_row < N) {
                    O[qkv_off + global_row * d + col] =
                        (1.f / li_new[s_row]) *
                        (li[s_row] * expf(mi[s_row] - mi_new[s_row]) * Oi[s_row * d + col] // he old value update krnyasathi
                         + expf(mij[s_row] - mi_new[s_row]) * pv); // he new value update krnyasathi
                }
            }
            if (s_col == 0 && global_row < N) {
                m[lm_off + global_row] = mi_new[s_row];
                l[lm_off + global_row] = li_new[s_row];
            }
            __syncthreads();  
        }
    }
}