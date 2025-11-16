#include <torch/extension.h>

torch::Tensor matrix_multiply_coalescing(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply_coalescing", &matrix_multiply_coalescing, "Matrix multiplication (CUDA with coalescing)");
}


