#pragma once

#include <array>
#include <cstddef>

namespace hyle::compute::detail {

using NativeParameters = std::array<std::size_t, 3>;
using NativeRoute = void (*)(const float *, const float *, float *,
                             const NativeParameters &, float) noexcept;

} // namespace hyle::compute::detail
