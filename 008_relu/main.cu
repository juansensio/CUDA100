#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// #define ALPHA 0.01f
#define ALPHA 0.0f

__global__ void relu_kernel(const float* input, float* output, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // output[idx] = max(0.0f, input[idx]);
        output[idx] = input[idx] > 0 ? input[idx] : ALPHA * input[idx]; // leaky ReLU
    }
};

void benchmark() {
    printf("\nBenchmarking reverse_array kernel throughput for various 1D array sizes...\n");
    printf("%-10s %-14s %-10s %-7s\n", "N", "Time (ms)", "GB/s", "%Max");
    int sizes[] = {1000, 100000, 1000000, 100000000};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 8;
    const double max_gb_per_s = 936.2; // Theoretical max bandwidth (3090 RTX)

    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t bytes = N * sizeof(float);

        float *A = (float*)malloc(bytes);
        for (int i = 0; i < N; ++i) A[i] = 1.0f * i;
        float *input_d, *output_d;
        CUDA_OK(cudaMalloc((void**)&input_d, bytes));
        CUDA_OK(cudaMalloc((void**)&output_d, bytes));
        CUDA_OK(cudaMemcpy(input_d, A, bytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(256);
        dim3 gridDim((N + threadsPerBlock.x  - 1) / threadsPerBlock.x);

        // --- kernel timing ---
        float ms_total = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        relu_kernel<<<gridDim, threadsPerBlock>>>(input_d, output_d, N);
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaMemcpy(input_d, A, bytes, cudaMemcpyHostToDevice)); // reset input each run
            cudaEventRecord(start);
            relu_kernel<<<gridDim, threadsPerBlock>>>(input_d, output_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            ms_total += ms;
        }
        ms_total /= num_runs;

        // Compute throughput (read + write, so *2)
        double num_bytes = (double)N * sizeof(float) * 2;
        double gbps = num_bytes / (ms_total * 1e6);
        double pct = 100.0 * gbps / max_gb_per_s;

        printf("%-10d %-14.4f %-10.2f %-7.2f\n", N, ms_total, gbps, pct);

        free(A);
        CUDA_OK(cudaFree(input_d));
        CUDA_OK(cudaFree(output_d));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 5;
    float input[N] = {-2.0, -1.0, 0.0, 1.0, 2.0}; 
    float expected[N] = {0.0, 0.0, 0.0, 1.0, 2.0};
    float output[N];

    float *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    relu_kernel<<<grid, block>>>(input_d, output_d, N);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (output[i] != expected[i]) {
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], output[i]);
            CUDA_OK(cudaFree(input_d));
            CUDA_OK(cudaFree(output_d));
            return EXIT_FAILURE;
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(output_d));
    benchmark();
    return 0;
}
