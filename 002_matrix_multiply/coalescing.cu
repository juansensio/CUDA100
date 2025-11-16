#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline unsigned int CEIL_DIV(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__ void matrix_multiply_coalescing(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

torch::Tensor matrix_multiply_coalescing(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    auto output = torch::empty({m, n}, A.options());
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiply_coalescing<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void benchmark_against_cublas() {
    printf("\nBenchmarking against cuBLAS for various sizes...\n");
    printf("%-10s %-18s %-18s %-18s %-18s %-18s\n", "N", "Coalescing (ms)", "GFLOPs", "cuBLAS (ms)", "GFLOPs", "Speedup");
    int sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 5;
    for (int idx = 0; idx < num_sizes; ++idx) {
        int M = sizes[idx], N = sizes[idx], K = sizes[idx];
        size_t bytes_A = M * K * sizeof(float);
        size_t bytes_B = K * N * sizeof(float);
        size_t bytes_C = M * N * sizeof(float);

        float *A = (float*)malloc(bytes_A);
        float *B = (float*)malloc(bytes_B);
        float *C_coalescing = (float*)malloc(bytes_C);
        float *C_cublas = (float*)malloc(bytes_C);

        for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
        for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

        float *A_d, *B_d, *C_d;
        CUDA_OK(cudaMalloc((void**)&A_d, bytes_A));
        CUDA_OK(cudaMalloc((void**)&B_d, bytes_B));
        CUDA_OK(cudaMalloc((void**)&C_d, bytes_C));

        // --- Coalescing kernel timing ---
        CUDA_OK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(B_d, B, bytes_B, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(16, 16);
        dim3 gridDim((N + threadsPerBlock.x - 1)/threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1)/threadsPerBlock.y);

        float ms_coalescing = 0.0f;
        cudaEvent_t start_coalescing, stop_coalescing;
        cudaEventCreate(&start_coalescing);
        cudaEventCreate(&stop_coalescing);

        // Warm-up run
        matrix_multiply_coalescing<<<gridDim, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);
        CUDA_OK(cudaDeviceSynchronize());
        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start_coalescing);
            matrix_multiply_coalescing<<<gridDim, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);
            cudaEventRecord(stop_coalescing);
            cudaEventSynchronize(stop_coalescing);
            float tmp;
            cudaEventElapsedTime(&tmp, start_coalescing, stop_coalescing);
            ms_coalescing += tmp;
        }
        ms_coalescing /= num_runs;
        CUDA_OK(cudaMemcpy(C_coalescing, C_d, bytes_C, cudaMemcpyDeviceToHost));

        CUDA_OK(cudaFree(A_d));
        CUDA_OK(cudaFree(B_d));
        CUDA_OK(cudaFree(C_d));
        cudaEventDestroy(start_coalescing);
        cudaEventDestroy(stop_coalescing);

        // --- cuBLAS timing ---
        CUDA_OK(cudaMalloc((void**)&A_d, bytes_A));
        CUDA_OK(cudaMalloc((void**)&B_d, bytes_B));
        CUDA_OK(cudaMalloc((void**)&C_d, bytes_C));
        CUDA_OK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(B_d, B, bytes_B, cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f, beta = 0.0f;

        float ms_cublas = 0.0f;
        cudaEvent_t start_cublas, stop_cublas;
        cudaEventCreate(&start_cublas);
        cudaEventCreate(&stop_cublas);

        // Warm-up
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, B_d, N, A_d, K, &beta, C_d, N
        );
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start_cublas);
            cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, B_d, N, A_d, K, &beta, C_d, N
            );
            cudaEventRecord(stop_cublas);
            cudaEventSynchronize(stop_cublas);
            float tmp;
            cudaEventElapsedTime(&tmp, start_cublas, stop_cublas);
            ms_cublas += tmp;
        }
        ms_cublas /= num_runs;
        CUDA_OK(cudaMemcpy(C_cublas, C_d, bytes_C, cudaMemcpyDeviceToHost));

        cublasDestroy(handle);
        CUDA_OK(cudaFree(A_d));
        CUDA_OK(cudaFree(B_d));
        CUDA_OK(cudaFree(C_d));
        cudaEventDestroy(start_cublas);
        cudaEventDestroy(stop_cublas);

        // Output result
        double gflops_coalescing = (2.0 * M * N * K) / (ms_coalescing * 1e6);
        double gflops_cublas = (2.0 * M * N * K) / (ms_cublas * 1e6);
        double speedup = gflops_coalescing / gflops_cublas;  
        printf("%-10d %-18.4f %-18.4f %-18.4f %-18.4f %-18.4f\n", M, ms_coalescing, gflops_coalescing, ms_cublas, gflops_cublas, speedup);

        free(A);
        free(B);
        free(C_coalescing);
        free(C_cublas);
    }
}

int main() {
    printf("Testing correctness with small matrices...\n");
    int N = 2;
    int M = 2;
    int K = 3;
    float A[M][K] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; 
    float B[K][N] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}; 
    float expected[M][N] = {{58.0, 64.0}, {139.0, 154.0}};
    
    float* C = (float*)malloc(M * N * sizeof(float)); 
    if (C == NULL) {
        fprintf(stderr, "Failed to allocate host memory for C\n");
        return EXIT_FAILURE;
    }
    
    float* A_d, *B_d, *C_d;
    CUDA_OK(cudaMalloc((void**)&A_d, M * K * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&B_d, K * N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&C_d, M * N * sizeof(float)));
    
    CUDA_OK(cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 grid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiply_coalescing<<<grid, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabsf(C[i*N + j] - expected[i][j]) > 1e-6f) {
                printf("Error at index %d, %d: expected %.1f, got %.1f\n", i, j, expected[i][j], C[i*N + j]);
                free(C);
                CUDA_OK(cudaFree(A_d));
                CUDA_OK(cudaFree(B_d));
                CUDA_OK(cudaFree(C_d));
                return EXIT_FAILURE;
            }
        }
    }
    printf("✓ Result is correct!\n");
    free(C);
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    benchmark_against_cublas();
    return 0;
}
