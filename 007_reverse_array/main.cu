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

#define TILE 256

__global__ void reverse_array(float* input, int N) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < N) {
    //     // input[idx] = input[N - idx - 1]; // data hazard: for large arrays some threads may read the same value before it is written by another thread
    // }

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t mirror = N - 1 - idx;
    if (idx < N / 2) {  // one thread swaps two elements
        float tmp = input[idx];
        input[idx] = input[mirror];
        input[mirror] = tmp;
    }

    // __shared__ float bufA[TILE];
    // __shared__ float bufB[TILE];
    // size_t tile = blockIdx.x;
    // size_t startA = tile * TILE;
    // size_t startB = N - (tile + 1) * TILE;
    // // If tiles cross or we passed the middle, stop
    // if (startA >= startB + TILE) return;
    // int t = threadIdx.x;
    // // Load front and back tiles
    // if (startA + t < N) bufA[t] = input[startA + t];
    // if (startB + t < N) bufB[t] = input[startB + t];
    // __syncthreads();
    // // Write swapped & reversed
    // if (startA + t < N) input[startA + t] = bufB[TILE - 1 - t];
    // if (startB + t < N) input[startB + t] = bufA[TILE - 1 - t];
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
        float *A_d;
        CUDA_OK(cudaMalloc((void**)&A_d, bytes));
        CUDA_OK(cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(TILE);
        dim3 gridDim(((N / 2) + threadsPerBlock.x  - 1) / threadsPerBlock.x);

        // --- kernel timing ---
        float ms_total = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        reverse_array<<<gridDim, threadsPerBlock>>>(A_d, N);
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice)); // reset input each run
            cudaEventRecord(start);
            reverse_array<<<gridDim, threadsPerBlock>>>(A_d, N);
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
        CUDA_OK(cudaFree(A_d));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    float A[N] = {1.0, 2.0, 3.0}; 
    float expected[N] = {3.0, 2.0, 1.0};

    float *A_d;
    CUDA_OK(cudaMalloc((void**)&A_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE);
    dim3 grid(((N / 2) + block.x - 1) / block.x);
    reverse_array<<<grid, block>>>(A_d, N);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (A[i] != expected[i]) {
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], A[i]);
            CUDA_OK(cudaFree(A_d));
            return EXIT_FAILURE;
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(A_d));
    benchmark();
    return 0;
}
