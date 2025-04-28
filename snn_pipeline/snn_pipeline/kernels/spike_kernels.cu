#include <torch/extension.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256

// 前向 kernel：输入膜电位 u，输出 spike 和 reset_u
__global__ void spike_forward_kernel(
    const float* __restrict__ u,
    float* __restrict__ spike,
    float* __restrict__ reset_u,
    float threshold,
    int reset_mode,
    size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float val = u[idx];
  float s = val > threshold ? 1.f : 0.f;
  spike[idx] = s;
  if (reset_mode == 0)        // subtract
    reset_u[idx] = val - s * threshold;
  else                        // zero
    reset_u[idx] = val * (1.f - s);
}

// 反向 kernel：输入 grad_spike, grad_reset, u；输出 grad_u
__global__ void spike_backward_kernel(
    const float* __restrict__ grad_spike,
    const float* __restrict__ grad_reset,
    const float* __restrict__ u,
    float* __restrict__ grad_u,
    float threshold,
    int reset_mode,
    size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float val = u[idx];
  float s = val > threshold ? 1.f : 0.f;

  // surrogate gradient: α·exp(−β|u−θ|) / (exp(−β|u−θ|)+1)²
  float diff    = fabsf(val - threshold);
  float expval  = expf(-0.05f * diff);
  float surrogate = 0.3f * expval / ((expval + 1.f)*(expval + 1.f));

  float gu = 0.f;
  if (reset_mode == 0) {
    gu = grad_reset[idx] + grad_spike[idx] * surrogate;
  } else {
    gu = grad_reset[idx] * (1.f - s) + grad_spike[idx] * surrogate;
  }
  grad_u[idx] = gu;
}

// C++ 接口：forward
std::vector<at::Tensor> spike_forward(
    at::Tensor u,
    double threshold,
    std::string reset_mode) {
  auto u_cont = u.contiguous();
  auto N = u_cont.numel();
  auto spike    = torch::empty_like(u_cont);
  auto reset_u  = torch::empty_like(u_cont);
  int mode = (reset_mode == "zero") ? 1 : 0;

  int threads = THREADS_PER_BLOCK;
  int blocks  = (N + threads - 1) / threads;
  spike_forward_kernel<<<blocks, threads>>>(
    u_cont.data_ptr<float>(),
    spike.data_ptr<float>(),
    reset_u.data_ptr<float>(),
    (float)threshold,
    mode,
    N);
  cudaDeviceSynchronize();
  return {spike, reset_u};
}

// C++ 接口：backward
at::Tensor spike_backward(
    at::Tensor grad_spike,
    at::Tensor grad_reset,
    at::Tensor u,
    double threshold,
    std::string reset_mode) {
  auto gs = grad_spike.contiguous();
  auto gr = grad_reset.contiguous();
  auto u_cont = u.contiguous();
  auto N = u_cont.numel();
  auto grad_u = torch::empty_like(u_cont);
  int mode = (reset_mode == "zero") ? 1 : 0;

  int threads = THREADS_PER_BLOCK;
  int blocks  = (N + threads - 1) / threads;
  spike_backward_kernel<<<blocks, threads>>>(
    gs.data_ptr<float>(),
    gr.data_ptr<float>(),
    u_cont.data_ptr<float>(),
    grad_u.data_ptr<float>(),
    (float)threshold,
    mode,
    N);
  cudaDeviceSynchronize();
  return grad_u;
}

// PyBind 接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spike_forward,  "Spike forward (CUDA)");
  m.def("backward", &spike_backward, "Spike backward (CUDA)");
}