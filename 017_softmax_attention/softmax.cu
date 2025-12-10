#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void softmax_max_kernel(const float* input, float* partial_max, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Find max in this thread's stride
    float thread_max = -INFINITY;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }
    
    // Reduce max across block
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

__global__ void softmax_exp_sum_kernel(const float* input, float* output, 
                                       float global_max, float* partial_sum, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Compute exp and sum in this thread's stride
    float thread_sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float val = expf(input[i] - global_max);
        output[i] = val;
        thread_sum += val;
    }
    
    // Reduce sum across block
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        partial_sum[blockIdx.x] = sdata[0];
    }
}

__global__ void softmax_normalize_kernel(float* output, float global_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] /= global_sum;
    }
}

// ~85% pytorch
void softmax(const float* input_d, float* output_d, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = min((N + threadsPerBlock - 1) / threadsPerBlock, 256);
    const int smem_size = threadsPerBlock * sizeof(float);
    
    // Allocate temporary buffers for partial results
    float *partial_max_d, *partial_sum_d;
    CUDA_OK(cudaMalloc(&partial_max_d, blocksPerGrid * sizeof(float)));
    CUDA_OK(cudaMalloc(&partial_sum_d, blocksPerGrid * sizeof(float)));
    
    // Step 1: Find global max
    softmax_max_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(
        input_d, partial_max_d, N);
    
    // Reduce partial maxes on host (for simplicity and small overhead)
    float *partial_max_h = (float*)malloc(blocksPerGrid * sizeof(float));
    CUDA_OK(cudaMemcpy(partial_max_h, partial_max_d, 
                       blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    float global_max = partial_max_h[0];
    for (int i = 1; i < blocksPerGrid; i++) {
        global_max = fmaxf(global_max, partial_max_h[i]);
    }
    free(partial_max_h);
    
    // Step 2: Compute exp(x - global_max) and partial sums
    softmax_exp_sum_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(
        input_d, output_d, global_max, partial_sum_d, N);
    
    // Reduce partial sums on host
    float *partial_sum_h = (float*)malloc(blocksPerGrid * sizeof(float));
    CUDA_OK(cudaMemcpy(partial_sum_h, partial_sum_d, 
                       blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_sum += partial_sum_h[i];
    }
    free(partial_sum_h);
    
    // Step 3: Normalize
    softmax_normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        output_d, global_sum, N);
    
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_OK(cudaFree(partial_max_d));
    CUDA_OK(cudaFree(partial_sum_d));
}

__global__ void batched_divide_softmax_kernel(float* data, float divisor, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    float* row_data = data + row * N;
    int tid = threadIdx.x;
    
    // Step 1: Divide and find max in one pass
    float thread_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        row_data[i] /= divisor;
        thread_max = fmaxf(thread_max, row_data[i]);
    }
    
    // Reduce max across block using shared memory
    __shared__ float smem_max[256];
    smem_max[tid] = thread_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        }
        __syncthreads();
    }
    
    float row_max = smem_max[0];
    
    // Step 2: Compute exp(x - max) and accumulate sum
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    
    // Reduce sum across block
    __shared__ float smem_sum[256];
    smem_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = smem_sum[0];
    
    // Step 3: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}