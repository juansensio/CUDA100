#include <torch/extension.h>

torch::Tensor matrix_add(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_add", &matrix_add, "Matrix addition (CUDA)");
}


