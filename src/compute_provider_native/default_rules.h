#pragma once

#include "hyle/compute/operation.h"
#include "hyle/compute/providers/native.h"

namespace hyle::compute::native::internal {

// Private facade assembly seam. Expert Native has no default preference.
enum class DefaultRuleScope : std::uint8_t { kExpert, kDirect, kGraph };
HYLE_COMPUTE_PROVIDER_NATIVE_API Provider
default_provider(Options options, DefaultRuleScope scope,
                 bool single_native_thread) noexcept;

bool qualified_default_environment() noexcept;
bool retired_default_domain(Operation, std::span<const TensorDesc>,
                            std::span<const TensorDesc>) noexcept;
RoutePolicyId qualified_default_policy(Operation, std::span<const TensorDesc>,
                                       std::span<const TensorDesc>) noexcept;

} // namespace hyle::compute::native::internal
