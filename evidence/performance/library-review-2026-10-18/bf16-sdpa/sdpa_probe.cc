#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Case {
  const char* name;
  int batch, queries, keys, query_heads, kv_heads, depth;
  bool causal;
  float scale;
  bool wide_value = false;
  bool wide_key = false;
};

at::Tensor input(const Case& c, int kind) {
  const int rows = kind == 0 ? c.queries : c.keys;
  const int heads = kind == 0 ? c.query_heads : c.kv_heads;
  std::vector<float> data(std::int64_t(c.batch) * rows * heads * c.depth);
  for (int b = 0; b < c.batch; ++b)
    for (int s = 0; s < rows; ++s)
      for (int h = 0; h < heads; ++h)
        for (int d = 0; d < c.depth; ++d) {
          const auto i = ((std::int64_t(b) * rows + s) * heads + h) * c.depth + d;
          float x = kind == 0 ? 0.3f * std::sin(.11f*d + .17f*s + .19f*h + .2f*b)
                    : kind == 1 ? 0.3f * std::cos(.07f*d + .23f*s + .13f*h + .1f*b)
                    : .2f*s + .1f*h + .05f*b + .5f*std::sin(.13f*d + .31f*s);
          if (kind == 2 && c.wide_value) x = 1.e8f + 1.e7f*x;
          if (kind == 1 && c.wide_key) x *= 1.e6f;
          data[i] = x;
        }
  return at::from_blob(data.data(), {c.batch, rows, heads, c.depth},
             at::TensorOptions().dtype(at::kFloat)).to(at::kBFloat16);
}

// Independent stable FP64 formula from the already BF16-rounded input ports.
// No Torch SDPA/math operator is used as the numerical oracle.
std::vector<double> reference(const Case& c, const at::Tensor& q_cpu,
                              const at::Tensor& k_cpu, const at::Tensor& v_cpu) {
  const auto q = q_cpu.to(at::kFloat), k = k_cpu.to(at::kFloat), v = v_cpu.to(at::kFloat);
  const auto* qp = q.const_data_ptr<float>();
  const auto* kp = k.const_data_ptr<float>();
  const auto* vp = v.const_data_ptr<float>();
  std::vector<double> out(std::int64_t(c.batch)*c.queries*c.query_heads*c.depth);
  std::vector<double> weights(c.keys);
  for (int b = 0; b < c.batch; ++b)
    for (int s = 0; s < c.queries; ++s)
      for (int h = 0; h < c.query_heads; ++h) {
        const int kh = h / (c.query_heads / c.kv_heads);
        const int active = c.causal ? std::min(c.keys, c.keys-c.queries+s+1) : c.keys;
        const auto qi = ((std::int64_t(b)*c.queries+s)*c.query_heads+h)*c.depth;
        double largest = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < active; ++t) {
          const auto ki = ((std::int64_t(b)*c.keys+t)*c.kv_heads+kh)*c.depth;
          double dot = 0;
          for (int d = 0; d < c.depth; ++d) dot += double(qp[qi+d])*kp[ki+d];
          weights[t] = dot*double(c.scale);
          largest = std::max(largest, weights[t]);
        }
        double total = 0;
        for (int t = 0; t < active; ++t) total += weights[t] = std::exp(weights[t]-largest);
        for (int d = 0; d < c.depth; ++d) {
          double sum = 0;
          for (int t = 0; t < active; ++t) {
            const auto vi = ((std::int64_t(b)*c.keys+t)*c.kv_heads+kh)*c.depth+d;
            sum += weights[t]*vp[vi];
          }
          out[qi+d] = sum/total;
        }
      }
  return out;
}

bool exercise(const Case& c) {
  auto q_cpu = input(c, 0), k_cpu = input(c, 1), v_cpu = input(c, 2);
  const auto expected = reference(c, q_cpu, k_cpu, v_cpu);
  const auto device = at::Device(at::kCUDA, 0);
  auto q = q_cpu.to(device), k = k_cpu.to(device), v = v_cpu.to(device);
  c10::cuda::device_synchronize();
  auto before = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  const int ratio = c.query_heads/c.kv_heads;
  const int64_t mask = c.causal
      ? int64_t(sdp::CustomMaskType::CausalFromBottomRight)
      : int64_t(sdp::CustomMaskType::NoCustomMask);
  bool zero_stride_alias = true;
  std::vector<at::Tensor> groups;
  // Each K/V group uses a zero-head-stride view; no repeated K/V allocation.
  // The current SDK native operator requires equal logical head counts.
  const auto fused = [&](const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs) {
    return std::get<0>(at::_efficient_attention_forward(qs, ks, vs, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        0.0, mask, false, double(c.scale)));
  };
  if (ratio == 1) {
    groups.push_back(fused(q, k, v));
  } else for (int h = 0; h < c.kv_heads; ++h) {
    const auto qs = q.narrow(2, h*ratio, ratio);
    const auto ks = k.narrow(2, h, 1).expand({c.batch, c.keys, ratio, c.depth});
    const auto vs = v.narrow(2, h, 1).expand({c.batch, c.keys, ratio, c.depth});
    if (ratio > 1)
      zero_stride_alias &= ks.stride(2) == 0 && vs.stride(2) == 0 &&
          ks.const_data_ptr() == k.narrow(2,h,1).const_data_ptr() &&
          vs.const_data_ptr() == v.narrow(2,h,1).const_data_ptr();
    groups.push_back(fused(qs, ks, vs));
  }
  const auto output = groups.size() == 1 ? groups[0] : at::cat(groups, 2);
  c10::cuda::device_synchronize();
  const auto after = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto actual = output.to(at::kCPU).to(at::kFloat).contiguous();
  const auto* ap = actual.const_data_ptr<float>();
  bool finite = true;
  double max_abs = 0, max_normalized = 0, square_error = 0, square_reference = 0;
  constexpr double atol = 0.01, rtol = 0.015;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    finite &= std::isfinite(ap[i]);
    const double error = std::abs(double(ap[i])-expected[i]);
    max_abs = std::max(max_abs, error);
    max_normalized = std::max(max_normalized, error/(atol+rtol*std::abs(expected[i])));
    square_error += error*error;
    square_reference += expected[i]*expected[i];
  }
  const bool pass = finite && zero_stride_alias && max_normalized <= 1 &&
      output.scalar_type() == at::kBFloat16;
  std::cout << std::setprecision(12)
      << "{\"case\":\"" << c.name << "\",\"pass\":" << (pass?"true":"false")
      << ",\"batch\":" << c.batch << ",\"q\":" << c.queries << ",\"kv\":" << c.keys
      << ",\"query_heads\":" << c.query_heads << ",\"kv_heads\":" << c.kv_heads
      << ",\"depth\":" << c.depth << ",\"causal_bottom_right\":" << (c.causal?"true":"false")
      << ",\"scale\":" << c.scale << ",\"bf16_output\":" << (output.scalar_type()==at::kBFloat16?"true":"false")
      << ",\"finite\":" << (finite?"true":"false") << ",\"zero_stride_alias\":" << (zero_stride_alias?"true":"false")
      << ",\"atol\":" << atol << ",\"rtol\":" << rtol
      << ",\"max_abs_error_vs_fp64\":" << max_abs << ",\"max_normalized_error\":" << max_normalized
      << ",\"relative_l2_error\":" << std::sqrt(square_error/std::max(square_reference,1.e-30))
      << ",\"torch_allocator_peak_extra_bytes\":" << after.allocated_bytes[0].peak-before.allocated_bytes[0].current
      << ",\"hypothetical_one_f32_scores_bytes\":" << std::int64_t(c.batch)*c.query_heads*c.queries*c.keys*4
      << "}\n";
  return pass;
}

int main() {
  try {
    at::NoGradGuard no_grad;
    at::globalContext().lazyInitDevice(at::kCUDA);
    c10::cuda::set_device(0);
    std::cout << "{\"torch\":\"2.8.0+cu129\",\"flash_available\":"
              << (sdp::is_flash_attention_available()?"true":"false")
              << ",\"operator\":\"_efficient_attention_forward\",\"timing\":false}\n";
    const Case cases[]{
      {"mha-square",2,17,17,4,4,64,true,.125f},
      {"mha-bottom-right",2,17,31,4,4,64,true,.125f},
      {"gqa-bottom-right-d256",2,17,31,8,2,256,true,.0625f},
      {"gqa-decode",1,1,513,8,2,256,true,.0625f},
      {"gqa-scale-zero",2,17,31,8,2,256,true,0.0f},
      {"gqa-scale-negative",2,17,31,8,2,256,true,-.0625f},
      {"gqa-wide-values",2,17,31,8,2,256,true,.0625f,true},
      {"gqa-wide-keys",2,17,31,8,2,256,true,.0625f,false,true},
      {"gqa-noncausal-q-longer",1,5,3,8,2,64,false,-.125f},
      {"gqa-space-shape",1,128,257,8,2,64,true,.125f},
    };
    bool pass = true;
    for (const auto& c : cases) pass &= exercise(c);
    c10::cuda::device_synchronize();
    std::cout << "{\"all_passed\":" << (pass?"true":"false") << "}\n";
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
