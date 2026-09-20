#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace hyle::compute::flashinfer::native {

// Private C++17 CUDA compiler boundary. The owning C++23 adapter validates all
// fields and catches upstream exceptions; no vendor types enter public HYLE.
struct Attention final {
  std::uint32_t query_length = 0;
  std::uint32_t kv_length = 0;
  float scale = 1.0F;
  bool bfloat16 = false;
};

cudaError_t execute(const Attention &attention, const void *query,
                    const void *key, const void *value, void *output,
                    void *workspace, cudaStream_t stream);

} // namespace hyle::compute::flashinfer::native
