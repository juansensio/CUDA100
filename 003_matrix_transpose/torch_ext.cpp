#include <torch/extension.h>

torch::Tensor matrix_transpose(torch::Tensor A);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_transpose", &matrix_transpose, "Matrix transpose (CUDA)");
}


