#include <torch/extension.h>

torch::Tensor reduction(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduction", &reduction, "Reduction kernel (CUDA)");
}


