#include <torch/extension.h>

torch::Tensor count_equal_kernel_torch(torch::Tensor input, int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_equal_kernel_torch", &count_equal_kernel_torch, "Count equal kernel (CUDA)");
}


