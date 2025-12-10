#include <torch/extension.h>

torch::Tensor softmax_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor softmax_attention_cublas(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_attention", &softmax_attention, "Softmax attention kernel (CUDA)");
    m.def("softmax_attention_cublas", &softmax_attention_cublas, "Softmax attention kernel (cuBLAS)");
}


