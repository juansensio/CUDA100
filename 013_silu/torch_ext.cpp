#include <torch/extension.h>

torch::Tensor silu(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu", &silu, "SiLU kernel (CUDA)");
}


