#include <torch/extension.h>

torch::Tensor vectorAdd(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vectorAdd", &vectorAdd, "Vector addition (CUDA)");
}


