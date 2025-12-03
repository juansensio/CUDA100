#include <torch/extension.h>

torch::Tensor swiglu(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu", &swiglu, "SWiGLU kernel (CUDA)");
}


