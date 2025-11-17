#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
inline unsigned int CEIL_DIV(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor matrix_multiply_naive(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    auto output = torch::empty({m, n}, A.options());
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiply_naive<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor matrix_multiply_cublas(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    auto output = torch::empty({m, n}, A.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For row-major matrices, to compute C = A @ B using cuBLAS (column-major):
    // We compute C^T = B^T @ A^T, which means C = (B^T @ A^T)^T = A @ B
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // For row-major data interpreted as column-major by cuBLAS:
    // - A (m×k row-major) appears as (k×m column-major), lda = k
    // - B (k×n row-major) appears as (n×k column-major), ldb = n  
    // - C (m×n row-major) appears as (n×m column-major), ldc = n
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both to work with row-major data
        n, m, k,
        &alpha,
        B.data_ptr<float>(), n,  // B is k x n row-major, seen as n x k col-major, ldb = n
        A.data_ptr<float>(), k,  // A is m x k row-major, seen as k x m col-major, lda = k
        &beta,
        output.data_ptr<float>(), n  // C is m x n row-major, seen as n x m col-major, ldc = n
    );
    
    cublasDestroy(handle);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
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
