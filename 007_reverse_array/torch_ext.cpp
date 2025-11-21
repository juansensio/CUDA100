#include <torch/extension.h>

torch::Tensor reverse_array(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reverse_array", &reverse_array, "Reverse array (CUDA)");
}


