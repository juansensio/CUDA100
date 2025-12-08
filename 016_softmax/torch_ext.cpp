#include <torch/extension.h>

torch::Tensor softmax(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax, "Softmax kernel (CUDA)");
}


