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

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        unsigned int hash = input[ix];
        for (int i = 0; i < R; ++i) {
            hash = fnv1a_hash(hash);
        }
        output[ix] = hash;
    }
}

void benchmark() {
    // Test for multiple R values
    const int R_list[] = {10, 50, 100};
    const int num_Rs = sizeof(R_list) / sizeof(R_list[0]);
    int sizes[] = {1000, 100000, 1000000, 10000000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int num_runs = 8;
    const double max_gb_per_s = 936.2; // Theoretical max bandwidth (3090 RTX)

    printf("\nBenchmarking fnv1a_hash_kernel throughput for various 1D array sizes and R rounds...\n");
    printf("%-10s %-6s %-14s %-10s %-7s\n", "N", "R", "Time (ms)", "GB/s", "%Max");

    for (int r_idx = 0; r_idx < num_Rs; ++r_idx) {
        int R = R_list[r_idx];
        for (int idx = 0; idx < num_sizes; ++idx) {
            int N = sizes[idx];
            size_t bytes_in = N * sizeof(int);
            size_t bytes_out = N * sizeof(unsigned int);

            int* A = (int*)malloc(bytes_in);
            for (int i = 0; i < N; ++i) A[i] = i;

            int *input_d = nullptr;
            unsigned int *output_d = nullptr;
            CUDA_OK(cudaMalloc((void**)&input_d, bytes_in));
            CUDA_OK(cudaMalloc((void**)&output_d, bytes_out));
            CUDA_OK(cudaMemcpy(input_d, A, bytes_in, cudaMemcpyHostToDevice));

            dim3 threadsPerBlock(256);
            dim3 gridDim((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

            // --- kernel timing ---
            float ms_total = 0.0f;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Warm-up
            fnv1a_hash_kernel<<<gridDim, threadsPerBlock>>>(input_d, output_d, N, R);
            CUDA_OK(cudaDeviceSynchronize());

            for (int run = 0; run < num_runs; ++run) {
                CUDA_OK(cudaMemcpy(input_d, A, bytes_in, cudaMemcpyHostToDevice)); // reset input each run
                cudaEventRecord(start);
                fnv1a_hash_kernel<<<gridDim, threadsPerBlock>>>(input_d, output_d, N, R);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                ms_total += ms;
            }
            ms_total /= num_runs;

            // Throughput calculation: total size read+written, divided by time in seconds
            double num_bytes = ((double)bytes_in + (double)bytes_out); // input + output (GBps)
            double gbps = num_bytes / (ms_total * 1e6); // ms to s (1e-3), then bytes to GB (1e-9)
            double pct = 100.0 * gbps / max_gb_per_s;

            printf("%-10d %-6d %-14.2f %-10.2f %-7.2f\n", N, R, ms_total, gbps, pct);

            free(A);
            CUDA_OK(cudaFree(input_d));
            CUDA_OK(cudaFree(output_d));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    int R = 3;
    int input[N] = {0, 1, 2147483647}; 
    unsigned int expected[N] = {96754810, 3571711400, 2006156166};
    unsigned int output[N];

    int *input_d;
    unsigned int *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * sizeof(unsigned int)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    fnv1a_hash_kernel<<<grid, block>>>(input_d, output_d, N, R);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {   
        if (output[i] != expected[i]) {
            printf("Error at index %d: expected %u, got %u\n", i, expected[i], output[i]);
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
