#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/compute/vulkan.h"
#include <memory>
#include <vector>

namespace hyle::compute::vulkan::detail { struct Run; struct Runtime; }
namespace hyle::graph::vulkan::detail {
// Cold provenance of an original native region port. The enclosing Graph
// Invocation owns both slots. This adds neither ownership nor ready authority;
// consumers keep their original dependency and actual retirement contracts.
struct NativeValue final {
  const compute::vulkan::Tensor *tensor = nullptr;
  const std::shared_ptr<compute::vulkan::detail::Run> *run = nullptr;
  const std::vector<compute::vulkan::Tensor> *checks = nullptr;
  const bool *complete = nullptr;
};
const void *native_value_type() noexcept;
const NativeValue *native_value(const RegionValue &) noexcept;
Status append_native_checks(const NativeValue &,
    const std::shared_ptr<compute::vulkan::detail::Runtime> &,
    std::vector<compute::vulkan::Tensor> &) noexcept;
} // namespace hyle::graph::vulkan::detail
