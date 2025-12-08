#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <chrono>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDNN_OK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void exp_subtract_max_kernel(const float* input, float* output, float max_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = expf(input[idx] - max_val);
    }
}

__global__ void normalize_kernel(float* output, float sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] /= sum;
    }
}

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

// ~50% pytorch
void softmax_slow(const float* input_d, float* output_d, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1: find max
    thrust::device_ptr<const float> dev_ptr_input(input_d);
    float h_max = thrust::reduce(dev_ptr_input, dev_ptr_input + N, -INFINITY, thrust::maximum<float>());

    // step 2: compute exp(input - max) for numerical stability
    // probar ambas a ver que va mejor
    // thrust::device_ptr<float> dev_ptr_output(output_d);
    // thrust::transform(dev_ptr_output, dev_ptr_output + N, dev_ptr_output, [h_max](float x) { return exp(x - h_max); });
    exp_subtract_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, output_d, h_max, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Step 3: Sum all exponentiated values using Thrust
    thrust::device_ptr<float> dev_ptr_output(output_d);
    float h_sum = thrust::reduce(dev_ptr_output, dev_ptr_output + N, 0.0f, thrust::plus<float>());

    // Step 4: Normalize by dividing each element by sum
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output_d, h_sum, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
}

void softmax_cudnn(cudnnHandle_t cudnn, cudnnTensorDescriptor_t input_desc, 
                   cudnnTensorDescriptor_t output_desc,
                   const float* input_d, float* output_d) {
    float alpha = 1.0f, beta = 0.0f;
    
    // Apply softmax across channels
    CUDNN_OK(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, input_desc, input_d,
                                 &beta, output_desc, output_d));
}

bool check_results(float* result1, float* result2, int N, float eps = 1e-4f) {
    for (int i = 0; i < N; ++i) {
        if (fabsf(result1[i] - result2[i]) > eps) {
            printf("Mismatch at index %d: %.8f vs %.8f (diff: %.8f)\n", 
                   i, result1[i], result2[i], fabsf(result1[i] - result2[i]));
            return false;
        }
    }
    return true;
}

void benchmark() {
    printf("\n=== Softmax Benchmark ===\n\n");
    
    // Test multiple sizes
    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 100;
    int warmup_iterations = 10;
    
    printf("%-12s %-15s %-15s %-12s %-12s %-12s %-12s\n", 
           "Size", "Custom (ms)", "cuDNN (ms)", "Custom GFLOPS", "cuDNN GFLOPS", "Custom GB/s", "cuDNN GB/s");
    printf("%-12s %-15s %-15s %-12s %-12s %-12s %-12s\n", 
           "--------", "------------", "------------", "------------", "------------", "-----------", "-----------");
    
    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        
        // Allocate memory
        float *input_h = (float*)malloc(N * sizeof(float));
        float *output_custom_h = (float*)malloc(N * sizeof(float));
        float *output_cudnn_h = (float*)malloc(N * sizeof(float));
        
        // Initialize with random values
        for (int i = 0; i < N; ++i) {
            input_h[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        }
        
        float *input_d, *output_custom_d, *output_cudnn_d;
        CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
        CUDA_OK(cudaMalloc((void**)&output_custom_d, N * sizeof(float)));
        CUDA_OK(cudaMalloc((void**)&output_cudnn_d, N * sizeof(float)));
        CUDA_OK(cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Setup cuDNN (outside timing loop)
        cudnnHandle_t cudnn;
        CUDNN_OK(cudnnCreate(&cudnn));
        
        cudnnTensorDescriptor_t input_desc, output_desc;
        CUDNN_OK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_OK(cudnnCreateTensorDescriptor(&output_desc));
        
        // Set descriptors for 4D tensor: [batch=1, channels=N, height=1, width=1]
        CUDNN_OK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1));
        CUDNN_OK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1));
        
        // Warmup
        for (int i = 0; i < warmup_iterations; ++i) {
            softmax(input_d, output_custom_d, N);
            softmax_cudnn(cudnn, input_desc, output_desc, input_d, output_cudnn_d);
        }
        CUDA_OK(cudaDeviceSynchronize());
        
        // Benchmark custom implementation
        auto start_custom = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            softmax(input_d, output_custom_d, N);
        }
        CUDA_OK(cudaDeviceSynchronize());
        auto end_custom = std::chrono::high_resolution_clock::now();
        
        // Benchmark cuDNN implementation
        auto start_cudnn = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            softmax_cudnn(cudnn, input_desc, output_desc, input_d, output_cudnn_d);
        }
        CUDA_OK(cudaDeviceSynchronize());
        auto end_cudnn = std::chrono::high_resolution_clock::now();
        
        // Calculate timings
        double time_custom_ms = std::chrono::duration<double, std::milli>(end_custom - start_custom).count() / num_iterations;
        double time_cudnn_ms = std::chrono::duration<double, std::milli>(end_cudnn - start_cudnn).count() / num_iterations;
        
        // Calculate metrics
        // FLOPs: approximately 3N (exp, sum, divide) 
        double flops = 3.0 * N;
        double gflops_custom = (flops / 1e9) / (time_custom_ms / 1000.0);
        double gflops_cudnn = (flops / 1e9) / (time_cudnn_ms / 1000.0);
        
        // Bandwidth: read input + write output
        double bytes = 2.0 * N * sizeof(float);
        double gb_s_custom = (bytes / 1e9) / (time_custom_ms / 1000.0);
        double gb_s_cudnn = (bytes / 1e9) / (time_cudnn_ms / 1000.0);
        
        // Copy results back and check correctness
        CUDA_OK(cudaMemcpy(output_custom_h, output_custom_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(output_cudnn_h, output_cudnn_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool match = check_results(output_custom_h, output_cudnn_h, N);
        
        printf("%-12d %-15.6f %-15.6f %-12.2f %-12.2f %-12.2f %-12.2f %s\n", 
               N, time_custom_ms, time_cudnn_ms, 
               gflops_custom, gflops_cudnn, 
               gb_s_custom, gb_s_cudnn,
               match ? "✓" : "✗");
        
        // Cleanup
        CUDNN_OK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_OK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_OK(cudnnDestroy(cudnn));
        CUDA_OK(cudaFree(input_d));
        CUDA_OK(cudaFree(output_custom_d));
        CUDA_OK(cudaFree(output_cudnn_d));
        free(input_h);
        free(output_custom_h);
        free(output_cudnn_h);
    }
    
    printf("\nNotes:\n");
    printf("- GFLOPS calculated as (3*N FLOPs) / time\n");
    printf("- GB/s calculated as (2*N*4 bytes) / time (read input + write output)\n");
    printf("- Softmax is typically memory-bandwidth bound, not compute-bound\n");
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    float input[N] = {1.0, 2.0, 3.0};
    float expected[N] = {0.090, 0.244, 0.665}; // approx

    float *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));

    float* result = (float*)malloc(N * sizeof(float));
    softmax(input_d, output_d, N);
    CUDA_OK(cudaMemcpy(result, output_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(output_d));

    float eps = 1e-3f;
    int failed = 0;

    for (int i = 0; i < N; ++i) {
        if (fabsf(result[i] - expected[i]) > eps) {
            printf("Error at index %d: expected %.8f, got %.8f\n", i, expected[i], result[i]);
            failed = 1;
        }
    }

    if (failed) return EXIT_FAILURE;
    printf("✓ Result is correct: ");
    for (int i = 0; i < N; ++i) printf("%.6f ", result[i]);
    printf("\n");
    free(result);
    benchmark();
    return 0;
}
