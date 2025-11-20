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

typedef unsigned int uint;

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
};

void benchmark() {
    printf("\nBenchmarking matrix addition kernel vs cuBLAS for various sizes...\n");
    printf("%-10s %-16s %-14s   %-14s %-14s %-14s\n", "N", "Custom (ms)", "TFLOPs", "cuBLAS (ms)", "TFLOPs", "Speedup");
    int sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 5;
    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t numel = (size_t)N * N;
        size_t bytes = numel * sizeof(float);

        float *A = (float*)malloc(bytes);
        float *B = (float*)malloc(bytes);
        float *C = (float*)malloc(bytes);
        float *C_cublas = (float*)malloc(bytes);
        if (!A || !B || !C || !C_cublas) {
            fprintf(stderr, "Host malloc failed\n");
            exit(EXIT_FAILURE);
        }

        // Fill input matrices with random data
        for (size_t i = 0; i < numel; ++i) {
            A[i] = (float)(rand()) / (float)(RAND_MAX);
            B[i] = (float)(rand()) / (float)(RAND_MAX);
        }
        float *A_d = NULL, *B_d = NULL, *C_d = NULL, *C_cublas_d = NULL;
        CUDA_OK(cudaMalloc((void**)&A_d, bytes));
        CUDA_OK(cudaMalloc((void**)&B_d, bytes));
        CUDA_OK(cudaMalloc((void**)&C_d, bytes));
        CUDA_OK(cudaMalloc((void**)&C_cublas_d, bytes));
        CUDA_OK(cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(C_d, 0, bytes));
        CUDA_OK(cudaMemset(C_cublas_d, 0, bytes));

        dim3 block(8, 8);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        // --- Time our matrix_add kernel ---
        float ms_kernel = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        matrix_add<<<grid, block>>>(A_d, B_d, C_d, N); // Warm-up
        CUDA_OK(cudaPeekAtLastError());
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start);
            matrix_add<<<grid, block>>>(A_d, B_d, C_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float t;
            cudaEventElapsedTime(&t, start, stop);
            ms_kernel += t;
        }
        ms_kernel /= num_runs;

        CUDA_OK(cudaMemcpy(C, C_d, bytes, cudaMemcpyDeviceToHost));

        // --- Time cuBLAS saxpy (C = A+B) ---
        float ms_cublas = 0.0f;
        cublasHandle_t handle;
        cublasCreate(&handle);

        // For cuBLAS: C_cublas = A; then C_cublas = C_cublas + B
        for (int run = 0; run < num_runs; ++run) {
            CUDA_OK(cudaMemcpy(C_cublas_d, A_d, bytes, cudaMemcpyDeviceToDevice));
            float alpha = 1.0f;
            cudaEventRecord(start);
            cublasSaxpy(handle, numel, &alpha, B_d, 1, C_cublas_d, 1);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float t;
            cudaEventElapsedTime(&t, start, stop);
            ms_cublas += t;
        }
        ms_cublas /= num_runs;

        cublasDestroy(handle);
        CUDA_OK(cudaMemcpy(C_cublas, C_cublas_d, bytes, cudaMemcpyDeviceToHost));

        // --- Optional: correctness check
        float max_err = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            float diff = fabsf(C[i] - C_cublas[i]);
            if (diff > max_err) max_err = diff;
        }
        if (max_err > 1e-4f) {
            printf("Warning: kernel and cuBLAS outputs differ (max abs diff = %g)\n", max_err);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // --- TFLOPs computations
        // For element-wise addition, 1 FLOP per element: total FLOP = numel (really, it's add, so 1 op/elt).
        // Throughput in TFLOP/s = (numel / (ms * 1e-3)) / 1e12 = numel / (ms * 1e9)
        double our_tflops = (ms_kernel > 0.0) ? ((double)numel / (ms_kernel * 1e6)) : 0.0;
        double cublas_tflops = (ms_cublas > 0.0) ? ((double)numel / (ms_cublas * 1e6)) : 0.0;
        // To convert to TFLOP/s: (numel) / (ms * 1e-3) / 1e12 = numel / (ms * 1e9)

        printf("%-10d %-16.4f %-14.6f   %-14.4f %-14.6f %-14.2f\n", N, ms_kernel, our_tflops, ms_cublas, cublas_tflops, our_tflops / cublas_tflops);

        CUDA_OK(cudaFree(A_d));
        CUDA_OK(cudaFree(B_d));
        CUDA_OK(cudaFree(C_d));
        CUDA_OK(cudaFree(C_cublas_d));
        free(A);
        free(B);
        free(C);
        free(C_cublas);
    }
}


int main() {
    printf("Testing element-wise matrix addition...\n");
    int N = 2;
    float A[N][N] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    float B[N][N] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    float expected[N][N] = {{6.0f, 8.0f}, {10.0f, 12.0f}};
    float C[N][N] = {0.0f};

    float* A_d, *B_d, *C_d;
    CUDA_OK(cudaMalloc((void**)&A_d, N * N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&B_d, N * N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&C_d, N * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(8, 8);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matrix_add<<<grid, block>>>(A_d, B_d, C_d, N);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (C[i][j] != expected[i][j]) {
                printf("Error at index %d, %d: expected %.1f, got %.1f\n", i, j, expected[i][j], C[i][j]);
                CUDA_OK(cudaFree(A_d));
                CUDA_OK(cudaFree(B_d));
                CUDA_OK(cudaFree(C_d));
                return EXIT_FAILURE;
            }
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    benchmark();
    return 0;
}
