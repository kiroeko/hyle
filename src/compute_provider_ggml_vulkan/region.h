#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "hyle/compute/operation.h"
#include "native_device.h"

namespace hyle::compute::ggml_vulkan::detail {

struct RegionNode final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};
struct RegionRecipe final {
  std::vector<TensorDesc> values;
  std::vector<RegionNode> nodes;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};
// Descriptor-based ready resident port; capacity is measured after offset.
// Null handle/owner is permitted only while an idle slot remains unbound.
struct RegionBinding final {
  VkBuffer handle = VK_NULL_HANDLE;
  std::size_t offset = 0;
  std::size_t capacity = 0;
  std::shared_ptr<const void> owner;
};

HYLE_GGML_VULKAN_INTERNAL_API bool
region_supports(const Operation &, std::span<const TensorDesc>,
                std::span<const TensorDesc>) noexcept;

// One native graph/work/storage owner per invocation. Its caller serializes
// host encoding, binding and retirement on the route admission mutex. The
// completion fence may be waited outside that mutex while the owner is held.
class Region final {
public:
  Region() noexcept = default;
  Region(const Region &) = delete;
  Region &operator=(const Region &) = delete;
  Region(Region &&) noexcept = default;
  Region &operator=(Region &&) noexcept = default;
  HYLE_GGML_VULKAN_INTERNAL_API static Result<Region>
  create(NativeDevice, const RegionRecipe &) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API
      Status validate(std::span<const RegionBinding>) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API
      Status bind(std::span<const RegionBinding>) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API WorkState submit() noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API WorkState poll() noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API WorkState wait() noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API VkFence fence() const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API std::size_t storage_bytes() const noexcept;

private:
  struct State;
  std::shared_ptr<State> state_;
};
} // namespace hyle::compute::ggml_vulkan::detail
