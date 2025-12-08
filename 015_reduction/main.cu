#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// no podemos sincronizar a nivel global
// cada block se calcula su valor
// lanzamos kernels de manera recursiva hasta que solo quede un block
__global__ void reduction_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input element or 0 if out-of-bounds
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Thrust reduction
float thrust_reduce(const float* d_input, int N) {
    thrust::device_ptr<const float> dev_ptr(d_input);
    return thrust::reduce(dev_ptr, dev_ptr + N, 0.0f, thrust::plus<float>());
}

// CUB reduction
float cub_reduce(const float* d_input, int N) {
    float* d_output;
    CUDA_OK(cudaMalloc(&d_output, sizeof(float)));
    
    // Determine temporary storage size
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    
    // Allocate temporary storage
    CUDA_OK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    
    float result;
    CUDA_OK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_OK(cudaFree(d_temp_storage));
    CUDA_OK(cudaFree(d_output));
    return result;
}

float reduction_recursive(const float* input, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float* output_d;
    CUDA_OK(cudaMalloc((void**)&output_d, blocksPerGrid * sizeof(float)));
    
    // Launch kernel with shared memory
    reduction_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        input, output_d, N
    );
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    
    // Base case: if only one block, we're done
    if (blocksPerGrid == 1) {
        float result;
        CUDA_OK(cudaMemcpy(&result, output_d, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(output_d));
        return result;
    }
    
    // Recursive case: reduce the partial sums
    float result = reduction_recursive(output_d, blocksPerGrid);
    CUDA_OK(cudaFree(output_d));
    return result;
}

void benchmark() {
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_runs = 10;

    printf("\n=== Reduction Benchmark: Custom vs Thrust vs CUB ===\n");
    printf("%-12s %-14s %-14s %-14s %-14s %-14s %-14s\n", 
           "N", "Custom (ms)", "Thrust (ms)", "CUB (ms)", 
           "Custom GB/s", "Thrust GB/s", "CUB GB/s");

    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t bytes = N * sizeof(float);

        // Allocate and initialize input array with random floats
        float *input = (float*)malloc(bytes);
        for (int i = 0; i < N; ++i) {
            input[i] = ((float)rand()/(float)RAND_MAX) * 2000.0f - 1000.0f;
        }

        float *input_d;
        CUDA_OK(cudaMalloc((void**)&input_d, bytes));
        CUDA_OK(cudaMemcpy(input_d, input, bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));

        // --- Warm-up ---
        reduction_recursive(input_d, N);
        CUDA_OK(cudaDeviceSynchronize());
        thrust_reduce(input_d, N);
        CUDA_OK(cudaDeviceSynchronize());
        cub_reduce(input_d, N);
        CUDA_OK(cudaDeviceSynchronize());

        // --- Benchmark Custom ---
        float ms_custom = 0.0f;
        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaEventRecord(start));
            float result = reduction_recursive(input_d, N);
            CUDA_OK(cudaEventRecord(stop));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
            ms_custom += ms;
        }
        ms_custom /= num_runs;

        // --- Benchmark Thrust ---
        float ms_thrust = 0.0f;
        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaEventRecord(start));
            float result = thrust_reduce(input_d, N);
            CUDA_OK(cudaEventRecord(stop));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
            ms_thrust += ms;
        }
        ms_thrust /= num_runs;

        // --- Benchmark CUB ---
        float ms_cub = 0.0f;
        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaEventRecord(start));
            float result = cub_reduce(input_d, N);
            CUDA_OK(cudaEventRecord(stop));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
            ms_cub += ms;
        }
        ms_cub /= num_runs;

        // Calculate bandwidth (bytes read only for reduction)
        double bandwidth_custom = bytes / (ms_custom * 1e6); // GB/s
        double bandwidth_thrust = bytes / (ms_thrust * 1e6);
        double bandwidth_cub = bytes / (ms_cub * 1e6);

        printf("%-12d %-14.4f %-14.4f %-14.4f %-14.2f %.2f (%.2fx) %.2f (%.2fx)\n",
               N, ms_custom, ms_thrust, ms_cub,
               bandwidth_custom,
               bandwidth_thrust, ms_thrust / ms_custom,
               bandwidth_cub, ms_cub / ms_custom);

        free(input);
        CUDA_OK(cudaFree(input_d));
        CUDA_OK(cudaEventDestroy(start));
        CUDA_OK(cudaEventDestroy(stop));
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 8;
    float input[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float expected = 36.0;

    float *input_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    float result = reduction_recursive(input_d, N);

    // Verify the result
    float eps = 1e-5;
    if (fabsf(result - expected) > eps) {
        printf("Error: expected %.6f, got %.6f\n", expected, result);
        return EXIT_FAILURE;
    }
    printf("✓ Result is correct: %.6f\n", result);
    
    CUDA_OK(cudaFree(input_d));
    
    benchmark();
    return 0;
}
