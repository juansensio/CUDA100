#include <torch/extension.h>

torch::Tensor matrix_multiply_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matrix_multiply_cublas(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply_naive", &matrix_multiply_naive, "Matrix multiplication (CUDA)");
    m.def("matrix_multiply_cublas", &matrix_multiply_cublas, "Matrix multiplication (cuBLAS)");
}


