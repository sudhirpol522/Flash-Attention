template <const int Br, const int Bc>
__global__ void flash_attn_1_kernel(float* Q, float* K, float* V, int N, int d, int Tr, int Tc, float scale, float* l, float* m, float* O) {
    int tid = threadIdx.x;  // Br * Bc threads
    
    int sample_number = blockIdx.x;
    int head_number = blockIdx.y;
    
    int qkv_off = (N*d*gridDim.y*sample_number)+(N*d*head_number);
    int lm_off = (N*gridDim.y*sample_number)+(N*head_number);
    
    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br*d;
    float* Vj = Kj + Bc * d;
    float* Sij = Vj + Bc * d;
    float* Oi = Sij + Br * Bc;
    float* li = Oi + Br * d;
    float* li_new = li + Br;
    float* mi = li_new + Br;
    float* mi_new = mi + Br;
    float* mij_dash = mi_new + Br;
    
    for(int j=0;j<Tc;j++){
        int number_of_items = d/Br; # if total number of elements in that blocks are Bcxd and threads we have are BrxBc then items per thread would be d/Br
        for(int e=0;e<number_of_items;e++){
            int idx= e*(Br*Bc)+tid;  # aapn he block jumping of threads sathi krtoy
            if (idx<Bc*d){
                int row = idx/d;
                int col = idx%d;
                
                if(j*Bc+row<N){
                    Kj[row*Bc+col]=K[qkv_off+(j*Bc+row)*d+col];
                    Vj[row*Bc+col]=V[qkv_off+(j*Bc+row)*d+col];
                }
            }
        }
        __syncthreads();
        for(int i=0;i<Tr;j++){
        int number_of_items = d/Bc; # if total number of elements in that blocks are Brxd and threads we have are BrxBc then items per thread would be d/Bc
        for(int e=0;e<number_of_items;e++){
            int idx= e*(Br*Bc)+tid;  # aapn he block jumping of threads sathi krtoy
            if (idx<Br*d){
                int row = idx/d;
                int col = idx%d;
                
                if(i*Bc+row<N){
                    Qj[row*Bc+col]=Q[qkv_off+(i*Br+row)*d+col];
                    Oi[row*Bc+col]=O[qkv_off+(i*Br+row)*d+col];
                }
            }
        }
        
        
    }
    __syncthreads();
    
    int srow= tid/Bc;
    int scol=tid%Bc;
    
    if (scol==0){
        mi
    }
    

    
    
    
    }