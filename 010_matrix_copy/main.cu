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

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
  
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      B[(y+j)*width + x] = A[(y+j)*width + x];
    // ~850 GB/s
}

void benchmark() {
    int sizes[] = {256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_runs = 8;
    const double max_gb_per_s = 936.2; // Theoretical max bandwidth (3090 RTX)

    printf("\nBenchmarking copy_matrix_kernel throughput for NxN FP32 matrices...\n");
    printf("%-7s %-14s %-10s %-7s\n", "N", "Time (ms)", "GB/s", "%Max");

    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t bytes = N * N * sizeof(float);

        float* A = (float*)malloc(bytes);
        for (int i = 0; i < N*N; ++i) A[i] = (float)i;

        float *A_d = nullptr, *B_d = nullptr;
        CUDA_OK(cudaMalloc((void**)&A_d, bytes));
        CUDA_OK(cudaMalloc((void**)&B_d, bytes));
        CUDA_OK(cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice));

        dim3 block(TILE_DIM, BLOCK_ROWS);  
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
        
        // --- kernel timing ---
        float ms_total = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        copy_matrix_kernel<<<grid, block>>>(A_d, B_d, N);
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start);
            copy_matrix_kernel<<<grid, block>>>(A_d, B_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            ms_total += ms;
        }
        ms_total /= num_runs;

        // Throughput calculation: total size read+written, divided by time in seconds
        double num_bytes = (double)bytes * 2; // A + B in bytes
        double gbps = num_bytes / (ms_total * 1e6); // ms to s (1e-3), bytes to GB (1e-9)
        double pct = 100.0 * gbps / max_gb_per_s;

        printf("%-7d %-14.2f %-10.2f %-7.2f\n", N, ms_total, gbps, pct);

        free(A);
        CUDA_OK(cudaFree(A_d));
        CUDA_OK(cudaFree(B_d));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    float input[N][N] = {
        {5.5f, 6.6f, 7.7f},
        {8.8f, 9.9f, 10.1f},
        {11.2f, 12.3f, 13.4f}
    };
    float expected[N][N] = {
        {5.5f, 6.6f, 7.7f},
        {8.8f, 9.9f, 10.1f},
        {11.2f, 12.3f, 13.4f}
    };
    float output[N][N];

    float *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    copy_matrix_kernel<<<grid, block>>>(input_d, output_d, N);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {   
        for (int j = 0; j < N; ++j) {
            if (output[i][j] != expected[i][j]) {
                printf("Error at index %d, %d: expected %f, got %f\n", i, j, expected[i][j], output[i][j]);
                CUDA_OK(cudaFree(input_d));
                CUDA_OK(cudaFree(output_d));
                return EXIT_FAILURE;
            }
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(output_d));
    benchmark();
    return 0;
}
