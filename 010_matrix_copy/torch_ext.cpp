#include <torch/extension.h>

torch::Tensor copy_matrix_kernel(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_matrix_kernel", &copy_matrix_kernel, "Copy matrix kernel (CUDA)");
}


