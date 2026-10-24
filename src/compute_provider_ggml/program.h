#pragma once

#include <cstdint>
#include <vector>

#include "hyle/compute/providers/ggml_cpu.h"
#include "provider_internal.h"

namespace hyle::compute::ggml_cpu::internal {

struct Extension final {
  ProviderExtensionId id{};
  const void *table = nullptr;
  const Extension *next = nullptr;
};

struct Node final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};
// Immutable, recipe-local descriptions. No Graph objects or payload addresses.
struct Recipe final {
  struct Constant final {
    std::uint32_t value;
    ConstTensorView binding;
  };
  std::vector<TensorDesc> values;
  std::vector<Node> nodes;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  // Shared immutable Graph artifacts; ordinary inputs remain rebindable.
  std::vector<Constant> constants;
};

HYLE_COMPUTE_PROVIDER_GGML_API Status
widen_bfloat_constant(void *, ConstTensorView, TensorView,
                       mem::AllocatorRef) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API Status
pack_matrix_constant(void *, ConstTensorView, TensorView,
                     mem::AllocatorRef) noexcept;

HYLE_COMPUTE_PROVIDER_GGML_API ProviderRuntimeProperties
runtime(void *) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API bool
supports(void *, Operation, std::span<const TensorDesc>,
         std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API bool region_operation(Operation) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API ProviderPrepareOutcome
compile(void *, Recipe, mem::AllocatorRef, detail::ProviderPrepared *) noexcept;
// Actual compiled native input: no whole-bank import/cast and all reads are
// the original checked GET_ROWS. Used by both singleton and fused artifacts.
HYLE_COMPUTE_PROVIDER_GGML_API Status inspect_projected_read(
    const detail::ProviderPrepared &, std::uint32_t input_port,
    detail::OperationPreparedAccess *) noexcept;
// A static Graph extension adds region discovery to the same original provider
// instance. Its ordinary operation admission and execution remain available.
HYLE_COMPUTE_PROVIDER_GGML_API Provider
with_extension(Options, const Extension *) noexcept;

} // namespace hyle::compute::ggml_cpu::internal
