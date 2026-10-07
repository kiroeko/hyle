#pragma once

#include <array>

namespace hyle::compute::cuda::detail {

// Optional cold fact only. Failure leaves an empty string, never a CUDA API
// number or an error that disables the native compute route.
std::array<char, 80> read_driver_release() noexcept;

} // namespace hyle::compute::cuda::detail
