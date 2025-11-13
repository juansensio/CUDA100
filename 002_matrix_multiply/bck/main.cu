#include "kernels.h"

// Structure to hold kernel configuration
struct KernelConfig {
    const char* name;
    void (*kernel_func)(const float*, const float*, float*, int, int, int);
    dim3 (*get_grid)(int, int, int);
    dim3 (*get_block)(int, int, int);
};

// Grid and block configuration functions for naive kernel
dim3 get_grid_naive(int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    return dim3((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
}

dim3 get_block_naive(int M, int N, int K) {
    return dim3(16, 16);
}

// Grid and block configuration functions for coalescing kernel
dim3 get_grid_coalescing(int M, int N, int K) {
    return dim3(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
}

dim3 get_block_coalescing(int M, int N, int K) {
    return dim3(32 * 32);
}

// Wrapper functions to launch kernels
void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 grid = get_grid_naive(M, N, K);
    dim3 block = get_block_naive(M, N, K);
    matrix_multiply_naive<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_coalescing(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 grid = get_grid_coalescing(M, N, K);
    dim3 block = get_block_coalescing(M, N, K);
    matrix_multiply_coalescing<<<grid, block>>>(A, B, C, M, N, K);
}

// Verify result correctness
int verify_result(const float* C, int M, int N, const float* expected, float eps) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(C[i*N + j] - expected[i*N + j]) > eps) {
                printf("Error at index %d, %d: expected %.1f, got %.1f\n", 
                       i, j, expected[i*N + j], C[i*N + j]);
                return 0;
            }
        }
    }
    return 1;
}

// Benchmark a single kernel
void benchmark_kernel(
    const char* kernel_name,
    void (*launch_func)(const float*, const float*, float*, int, int, int),
    int M, int N, int K,
    int num_runs
) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *A_test = (float*)malloc(sizeA);
    float *B_test = (float*)malloc(sizeB);
    float *C_test = (float*)malloc(sizeC);

    // Initialize A and B
    for (int i = 0; i < M * K; ++i) A_test[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B_test[i] = 1.0f;

    // Allocate device memory
    float *A_d_test, *B_d_test, *C_d_test;
    CUDA_OK(cudaMalloc((void**)&A_d_test, sizeA));
    CUDA_OK(cudaMalloc((void**)&B_d_test, sizeB));
    CUDA_OK(cudaMalloc((void**)&C_d_test, sizeC));

    CUDA_OK(cudaMemcpy(A_d_test, A_test, sizeA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d_test, B_test, sizeB, cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run (not measured)
    launch_func(A_d_test, B_d_test, C_d_test, M, N, K);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    float total_ms = 0.0f;

    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start, 0);
        launch_func(A_d_test, B_d_test, C_d_test, M, N, K);
        cudaEventRecord(stop, 0);
        CUDA_OK(cudaPeekAtLastError());
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float avg_ms = total_ms / num_runs;

    // Calculate GFLOPs: 2*M*N*K operations
    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);

    printf("%-30s %6d %6d %6d %12.3f %10.3f\n", kernel_name, M, N, K, gflops, avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_OK(cudaFree(A_d_test));
    CUDA_OK(cudaFree(B_d_test));
    CUDA_OK(cudaFree(C_d_test));
    free(A_test);
    free(B_test);
    free(C_test);
}

int main() {
    // Test correctness with small matrices
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
    
    CUDA_OK(cudaMemcpy(A_d, A, K * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Test naive kernel
    dim3 grid_naive = get_grid_naive(M, N, K);
    dim3 block_naive = get_block_naive(M, N, K);
    matrix_multiply_naive<<<grid_naive, block_naive>>>(A_d, B_d, C_d, M, N, K);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_result(C, M, N, (float*)expected, 1e-6f)) {
        printf("✓ Naive kernel result is correct!\n");
    } else {
        printf("✗ Naive kernel result is incorrect!\n");
    }

    // Test coalescing kernel
    dim3 grid_coal = get_grid_coalescing(M, N, K);
    dim3 block_coal = get_block_coalescing(M, N, K);
    matrix_multiply_coalescing<<<grid_coal, block_coal>>>(A_d, B_d, C_d, M, N, K);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_result(C, M, N, (float*)expected, 1e-6f)) {
        printf("✓ Coalescing kernel result is correct!\n");
    } else {
        printf("✗ Coalescing kernel result is incorrect!\n");
    }
    
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    free(C);

    // Benchmark all kernels
    printf("\nBenchmarking all kernels:\n");
    printf("%-30s %6s %6s %6s %12s %10s\n", "Kernel", "M", "N", "K", "GFLOPs", "Time (ms)");
    printf("%s\n", "----------------------------------------------------------------------------");

    int test_vals[] = {128, 256, 512, 1024, 2048, 4096};
    int num_tests = sizeof(test_vals) / sizeof(test_vals[0]);
    const int num_runs = 10;

    for (int ti = 0; ti < num_tests; ++ti) {
        int tm = test_vals[ti];
        int tn = test_vals[ti];
        int tk = test_vals[ti];

        printf("\nMatrix size: %dx%dx%d\n", tm, tn, tk);
        
        // Benchmark naive kernel
        benchmark_kernel("matrix_multiply_naive", launch_naive, tm, tn, tk, num_runs);
        
        // Benchmark coalescing kernel
        benchmark_kernel("matrix_multiply_coalescing", launch_coalescing, tm, tn, tk, num_runs);
    }
    
    return 0;
}

