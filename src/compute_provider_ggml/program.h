#pragma once

#include <cstdint>
#include <vector>

#include "hyle/compute/providers/ggml_cpu.h"
#include "provider_internal.h"

namespace hyle::compute::ggml_cpu::internal {

struct Extension final {
  ProviderExtensionId id{};
  const void *table = nullptr;
};

struct Node final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};
// Immutable, recipe-local descriptions. No Graph objects or payload addresses.
struct Recipe final {
  std::vector<TensorDesc> values;
  std::vector<Node> nodes;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};

HYLE_COMPUTE_PROVIDER_GGML_API ProviderRuntimeProperties
runtime(void *) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API bool
supports(void *, Operation, std::span<const TensorDesc>,
         std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API bool region_operation(Operation) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_API ProviderPrepareOutcome
compile(void *, Recipe, mem::AllocatorRef, detail::ProviderPrepared *) noexcept;
// A static Graph extension supplies region-only discovery. Leaf configuration
// remains separate so the ordinary factory retains its mature leaf rules.
HYLE_COMPUTE_PROVIDER_GGML_API Provider
with_extension(Options, const Extension *) noexcept;

} // namespace hyle::compute::ggml_cpu::internal
