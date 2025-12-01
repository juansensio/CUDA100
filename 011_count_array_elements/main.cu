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

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (input[idx] == K) {
            atomicAdd(output, 1);
        }
    }
}

void benchmark() {
    // Range for N: 1 to 100,000,000
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 50000000, 100000000};
    // If you want a denser sweep, adjust the above array.
    int k_values[] = {1, 50000, 100000}; // Try varying k, here are three cases
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_ks = sizeof(k_values) / sizeof(k_values[0]);
    const int num_runs = 8;

    printf("\nBenchmarking count_equal_kernel for various N, measuring memory bandwidth...\n");
    printf("%-11s %-7s %-14s %-10s\n", "N", "k", "Time (ms)", "BW (GB/s)");

    for (int ki = 0; ki < num_ks; ++ki) {
        int k = k_values[ki];
        for (int idx = 0; idx < num_sizes; ++idx) {
            int N = sizes[idx];
            size_t bytes = N * sizeof(int);

            // Allocate and initialize input array
            int *input = (int*)malloc(bytes);
            // For a fair test: sprinkle some hits and random values
            for (int i = 0; i < N; ++i) {
                // Make about 1 in 30,000 ==k
                input[i] = ((i % 30000 == 0) ? k : (rand() % 100000) + 1);
            }

            int *input_d = nullptr, *output_d = nullptr;
            CUDA_OK(cudaMalloc((void**)&input_d, bytes));
            CUDA_OK(cudaMalloc((void**)&output_d, sizeof(int)));
            CUDA_OK(cudaMemcpy(input_d, input, bytes, cudaMemcpyHostToDevice));

            dim3 block(256);
            dim3 grid((N + block.x - 1) / block.x);

            // --- kernel timing ---
            float ms_total = 0.0f;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Warm-up
            CUDA_OK(cudaMemset(output_d, 0, sizeof(int)));
            count_equal_kernel<<<grid, block>>>(input_d, output_d, N, k);
            CUDA_OK(cudaDeviceSynchronize());

            for (int run = 0; run < num_runs; ++run) {
                CUDA_OK(cudaMemset(output_d, 0, sizeof(int)));
                cudaEventRecord(start);
                count_equal_kernel<<<grid, block>>>(input_d, output_d, N, k);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                ms_total += ms;
            }
            ms_total /= num_runs;

            // Calculate memory bandwidth
            // Each element is 4 bytes (int), and we read N elements
            double total_bytes = (double)N * sizeof(int);
            // Bandwidth in GB/s: bytes / (time_in_seconds) / 1e9
            // Since ms_total is in milliseconds: bytes / (ms_total * 1e-3) / 1e9 = bytes / (ms_total * 1e6)
            double bandwidth_gbps = total_bytes / (ms_total * 1e6);

            printf("%-11d %-7d %-14.3f %-10.2f\n", N, k, ms_total, bandwidth_gbps);

            free(input);
            CUDA_OK(cudaFree(input_d));
            CUDA_OK(cudaFree(output_d));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 5;
    int input[N] = {1, 2, 3, 4, 1};
    int k = 1;
    int expected = 2;
    int output;

    int *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&output_d, sizeof(int)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    count_equal_kernel<<<grid, block>>>(input_d, output_d, N, k);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(&output, output_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify the result
    if (output != expected) {
        printf("Error: expected %d, got %d\n", expected, output);
        CUDA_OK(cudaFree(input_d));
        CUDA_OK(cudaFree(output_d));
        return EXIT_FAILURE;
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(output_d));
    benchmark();
    return 0;
}
