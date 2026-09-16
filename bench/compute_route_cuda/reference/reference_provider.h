#pragma once

#include "hyle/compute/provider.h"

namespace hyle::compute::sglang_reference {
inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x53474c5245460001ULL};
// Benchmark-only packed precise SiLU; requires 16-byte tensor alignment.
Provider provider() noexcept;
} // namespace hyle::compute::sglang_reference
