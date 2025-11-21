#include <torch/extension.h>

torch::Tensor relu_kernel(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_kernel", &relu_kernel, "ReLU kernel (CUDA)");
}


