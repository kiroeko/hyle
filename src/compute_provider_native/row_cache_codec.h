#pragma once

#include <cstddef>

namespace hyle::compute::native::internal {

// One HYLE row-inline 68-byte FP4/UE8M0 row to 128 native BF16 values.
// Shares the checked codec's exact nibble, scale and BF16 rounding policy.
void decode_row_cache_float4(const std::byte *source,
                            std::byte *output) noexcept;

} // namespace hyle::compute::native::internal
