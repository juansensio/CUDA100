#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor softmax_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q tensor must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K tensor must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V tensor must be float32");
    int M = Q.size(0);
    int N = K.size(0);
    int d = Q.size(1);
    auto output = torch::empty({M, d}, Q.options());
    softmax_attention(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), M, N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor softmax_attention_cublas(torch::Tensor Q, torch::Tensor K, torch::Tensor V) { 
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q tensor must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K tensor must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V tensor must be float32");
    int M = Q.size(0);
    int N = K.size(0);
    int d = Q.size(1);
    auto output = torch::empty({M, d}, Q.options());
    softmax_attention_cublas(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), M, N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}