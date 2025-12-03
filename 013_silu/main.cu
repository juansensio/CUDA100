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

__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // float x = __ldg(&input[idx]);
        float sigmoid = 1.0f / (1.0f + exp(-x));
        // float sigmoid = 1.0f / (1.0f + __expf(-x));
        output[idx] = x * sigmoid;
    }
}

void benchmark() {
    // Range for N: 1 to 10,000 (see constraint)
    int sizes[] = {1, 10, 100, 1000, 5000, 10000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_runs = 8;

    printf("\nBenchmarking silu_kernel for various N (1 to 10,000)...\n");
    printf("%-11s %-14s %-14s %-14s\n", "N", "Time (ms)", "BW (GB/s)", "TFLOPs");

    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t bytes = N * sizeof(float);

        // Allocate and initialize input array with floats in [-100.0, 100.0]
        float *input = (float*)malloc(bytes);
        for (int i = 0; i < N; ++i) {
            input[i] = ((float)rand()/(float)RAND_MAX) * 200.0f - 100.0f;
        }

        float *input_d = nullptr, *output_d = nullptr;
        CUDA_OK(cudaMalloc((void**)&input_d, bytes));
        CUDA_OK(cudaMalloc((void**)&output_d, bytes));
        CUDA_OK(cudaMemcpy(input_d, input, bytes, cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        // --- kernel timing ---
        float ms_total = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        silu_kernel<<<grid, block>>>(input_d, output_d, N);
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start);
            silu_kernel<<<grid, block>>>(input_d, output_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            ms_total += ms;
        }
        ms_total /= num_runs;

        // Calculate memory bandwidth (read + write = 2 * N * sizeof(float))
        double total_bytes = 2.0 * N * sizeof(float);
        double bandwidth_gbps = total_bytes / (ms_total * 1e6); // GB/s

        // Compute FLOPs for SiLU operation
        // Each output: 1 mul (*x), 2 add/sub, 1 exp, 1 div, and 1 more mul (x*sigmoid) = about 5 FLOP
        // Realistically, exp and div counted as one each, so a rough estimate is 5 FLOPs per element
        double flops_per_element = 5.0; 
        double total_flops = N * flops_per_element;
        double tflops = (total_flops / (ms_total * 1e-3)) / 1e12; // TFLOPs

        printf("%-11d %-14.3f %-14.2f %-14.4f\n", N, ms_total, bandwidth_gbps, tflops);

        free(input);
        CUDA_OK(cudaFree(input_d));
        CUDA_OK(cudaFree(output_d));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}


int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    float input[N] = {0.5, 1.0, -0.5};
    float expected[N] = {0.3112295, 0.731059, -0.1887705};
    float output[N];

    float *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    silu_kernel<<<grid, block>>>(input_d, output_d, N);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result (relaxed comparison due to float differences)
    float eps = 1e-5;
    for (int i = 0; i < N; ++i) {
        if (fabsf(output[i] - expected[i]) > eps) {
            printf("Error at index %d: expected %.6f, got %.6f\n", i, expected[i], output[i]);
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
