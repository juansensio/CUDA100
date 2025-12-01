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

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        if (input[row * M + col] == K) {
            atomicAdd(output, 1);
        }
    }

}

void benchmark() {
    // Range for N, M: 1 to 10,000
    int sizes[][2] = {{100, 100}, {500, 500}, {1000, 1000}, {2000, 2000}, {5000, 5000}, {10000, 10000}}; // Example sizes
    int k_values[] = {1, 50, 100}; // Values for k in range 1 to 100
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_ks = sizeof(k_values) / sizeof(k_values[0]);
    const int num_runs = 8;

    printf("\nBenchmarking count_2d_equal_kernel for various N, M, measuring memory bandwidth...\n");
    printf("%-11s %-11s %-7s %-14s %-10s\n", "N", "M", "k", "Time (ms)", "BW (GB/s)");

    for (int ki = 0; ki < num_ks; ++ki) {
        int k = k_values[ki];
        for (int idx = 0; idx < num_sizes; ++idx) {
            int N = sizes[idx][0];
            int M = sizes[idx][1];
            size_t bytes = N * M * sizeof(int);

            // Allocate and initialize input array
            int *input = (int*)malloc(bytes);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    input[i * M + j] = ((i * M + j) % 30000 == 0) ? k : (rand() % 100) + 1;
                }
            }

            int *input_d = nullptr, *output_d = nullptr;
            CUDA_OK(cudaMalloc((void**)&input_d, bytes));
            CUDA_OK(cudaMalloc((void**)&output_d, sizeof(int)));
            CUDA_OK(cudaMemcpy(input_d, input, bytes, cudaMemcpyHostToDevice));

            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

            // --- kernel timing ---
            float ms_total = 0.0f;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Warm-up
            CUDA_OK(cudaMemset(output_d, 0, sizeof(int)));
            count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, output_d, N, M, k);
            CUDA_OK(cudaDeviceSynchronize());

            for (int run = 0; run < num_runs; ++run) {
                CUDA_OK(cudaMemset(output_d, 0, sizeof(int)));
                cudaEventRecord(start);
                count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, output_d, N, M, k);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                ms_total += ms;
            }
            ms_total /= num_runs;

            // Calculate memory bandwidth
            double total_bytes = (double)N * M * sizeof(int);
            double bandwidth_gbps = total_bytes / (ms_total * 1e6);

            printf("%-11d %-11d %-7d %-14.3f %-10.2f\n", N, M, k, ms_total, bandwidth_gbps);

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
    int N = 2, M = 3;
    int input[N][M] = {{1, 2, 3}, {4, 5, 1}};
    int k = 1;
    int expected = 2;
    int output;

    int *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * M * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&output_d, sizeof(int)));
    CUDA_OK(cudaMemcpy(input_d, input, N * M * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, output_d, N, M, k);

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
