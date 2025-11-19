#include <torch/extension.h>

torch::Tensor invert_image(torch::Tensor image);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invert_image", &invert_image, "Image inversion (CUDA)");
}


