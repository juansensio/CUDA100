#include <torch/extension.h>

torch::Tensor conv1d(torch::Tensor input, torch::Tensor kernel);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d", &conv1d, "1D convolution (CUDA)");
}


