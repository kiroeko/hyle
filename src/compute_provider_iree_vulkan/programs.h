#pragma once

#include <span>

namespace hyle::compute::iree_vulkan::detail {
std::span<const unsigned char> checked_scan_program() noexcept;
std::span<const unsigned char> fp8_program() noexcept;
std::span<const unsigned char> precision_program() noexcept;
std::span<const unsigned char> artifacts_program() noexcept;
std::span<const unsigned char> mlx_program() noexcept;
std::span<const unsigned char> official_program() noexcept;
std::span<const unsigned char> iq1_program() noexcept;
} // namespace hyle::compute::iree_vulkan::detail
