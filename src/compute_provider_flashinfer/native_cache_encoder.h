#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hyle::compute::flashinfer::native {
enum class EncoderInput : std::uint8_t { kFloat16,kBFloat16,kFloat32 };
struct EncoderSpec final {
  std::uint64_t elements=0;
  float scale=1.F;
  EncoderInput input=EncoderInput::kFloat16;
};
cudaError_t encode_cache(EncoderSpec,const void *,void *,unsigned *,cudaStream_t);
}
