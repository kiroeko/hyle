#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/device.h"
#include "hyle/compute/operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_MARLIN_API
#elif defined(HYLE_COMPUTE_PROVIDER_MARLIN_BUILD)
#define HYLE_COMPUTE_PROVIDER_MARLIN_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_MARLIN_API HYLE_IMPORT
#endif

namespace hyle::compute::marlin {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4d41524c494e0001ULL};
inline constexpr StorageFormatId kWeightFormat{0x48594c4543554441ULL,
                                               0x4d41524c494e0001ULL};
inline constexpr StorageFormatId kScaleFormat{0x48594c4543554441ULL,
                                              0x4d41524c53430001ULL};

#if defined(HYLE_COMPUTE_CUDA)
// Exact FP16 activation/output, signed symmetric Q4, group128, FP32 reduction.
// K is a positive multiple of 128, N a positive multiple of 64, M is 1..64.
// N*K must not exceed INT32_MAX (the native kernel's indexing domain).
// CUDA SM80+ is required. Each invocation has independent caller workspace;
// after a failed Completion discard that invocation and create a fresh one.
HYLE_COMPUTE_PROVIDER_MARLIN_API Provider provider() noexcept;

// Owns compressed CUDA payload; views borrow this owner. The supplied memory
// resource and its Context must outlive this object and all in-flight use.
// Immutable weights can be shared between independent invocations/Graph slots.
class QuantizedWeights final {
public:
  QuantizedWeights() noexcept = default;
  QuantizedWeights(const QuantizedWeights &) = delete;
  QuantizedWeights &operator=(const QuantizedWeights &) = delete;
  QuantizedWeights(QuantizedWeights &&) noexcept = default;
  QuantizedWeights &operator=(QuantizedWeights &&) noexcept = default;
  ~QuantizedWeights() = default;

  // Cold, synchronous packing. Values are row-major [N,K] integers in [-8,7];
  // scales are row-major [N,K/128], finite and nonnegative, rounded to FP16.
  // No model/file format is implied. Temporary host and GPU packing storage is
  // released before return; only Q4 weights and FP16 scales remain resident.
  // host_scratch is borrowed only for this call and must permit host mapping.
  HYLE_COMPUTE_PROVIDER_MARLIN_API static Result<QuantizedWeights>
  pack(const Context &context, mem::MemoryResourceRef resource,
       std::int64_t output_channels, std::int64_t input_channels,
       std::span<const std::int8_t> values, std::span<const float> scales,
       mem::MemoryResourceRef host_scratch = mem::system_resource()) noexcept;

  bool valid() const noexcept { return weight_storage_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  ConstTensorView weights() const noexcept {
    return valid() ? weights_ : ConstTensorView{};
  }
  ConstTensorView scales() const noexcept {
    return valid() ? scales_ : ConstTensorView{};
  }
  std::size_t resident_bytes() const noexcept {
    return weight_storage_.size() + scale_storage_.size();
  }
  static constexpr QuantizedMatMul operation() noexcept {
    return {.mode = QuantizedWeightMode::kSignedSymmetric4,
            .group_size = 128,
            .accumulation_type = DataType::kFloat32};
  }

private:
  mem::Buffer weight_storage_{};
  mem::Buffer scale_storage_{};
  ConstTensorView weights_{};
  ConstTensorView scales_{};
};
#endif

} // namespace hyle::compute::marlin
