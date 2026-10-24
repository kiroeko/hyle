#pragma once

#include <array>
#include <unordered_map>

#include <ggml.h>

#include "hyle/mem/std_allocator.h"

#include "program.h"
#include "packed_matmul_native.h"

namespace hyle::compute::ggml_cuda::internal {

// Build-local forms are keyed by native tensor identity, never payload address.
// The public values retain their original shape and ownership/reader edges.
class TensorBuildFrame final {
 public:
  TensorBuildFrame(const Recipe &, mem::AllocatorRef allocator);
  ggml_tensor *matrix_form(ggml_context *, ggml_tensor *, const TensorDesc &);
  void remember_grouped(std::uint32_t value, ggml_tensor *producer) noexcept {
    grouped_[value] = producer;
  }
  ggml_tensor *grouped(std::uint32_t value) const noexcept {
    return grouped_[value];
  }
  ggml_tensor *private_bfloat_widen_source(
      std::uint32_t value, std::span<ggml_tensor *const> tensors) const noexcept;

 private:
  struct ShapeKey final {
    ggml_tensor *source = nullptr;
    std::array<std::int64_t, 3> shape{};
    std::uint8_t rank = 0;
    bool operator==(const ShapeKey &) const = default;
  };
  struct ShapeHash final {
    std::size_t operator()(const ShapeKey &) const noexcept;
  };
  using ViewEntry = std::pair<const ShapeKey, ggml_tensor *>;
  std::unordered_map<ShapeKey, ggml_tensor *, ShapeHash,
      std::equal_to<ShapeKey>, mem::StdAllocator<ViewEntry>> views_;
  std::vector<ggml_tensor *, mem::StdAllocator<ggml_tensor *>> grouped_;
  struct WidenSource final {
    std::uint32_t source = UINT32_MAX;
    std::uint8_t readers = 0;
    bool published = false;
  };
  std::vector<WidenSource, mem::StdAllocator<WidenSource>> widen_sources_;
};

struct ScalarConstant final {
  ggml_tensor *tensor;
  float value;
};
struct DenseConstant final {
  ggml_tensor *tensor;
  const void *data;
  std::size_t bytes;
};

struct HadamardFactors final {
  std::int64_t low, high;
  constexpr std::size_t constant_elements() const noexcept {
    return static_cast<std::size_t>(
        low * low + (high != 1 && high != low ? high * high : 0));
  }
};
constexpr HadamardFactors hadamard_factors(std::int64_t width) noexcept {
  return width > 512 ? HadamardFactors{64, width / 64}
                     : HadamardFactors{width, 1};
}

// Native device is nominated by the actual provider Context. Unknown callers
// decline the large packed exception; ordinary bounded MatMul is unchanged.
bool tensor_operation_supports(Operation, std::span<const TensorDesc>,
                               std::span<const TensorDesc>,
                               int native_device = -1) noexcept;
ggml_tensor *create_tensor_input(ggml_context *, const TensorDesc &);
std::vector<ggml_tensor *> lower_tensor_operation(
    ggml_context *, const Recipe &, const Node &, std::span<ggml_tensor *const>,
    TensorBuildFrame &,
    std::vector<ScalarConstant> &, std::vector<DenseConstant> &,
    std::span<const float> hadamard,
    std::span<const std::int32_t> snapshot_order);

} // namespace hyle::compute::ggml_cuda::internal
