#pragma once

#include "hyle/compute/provider.h"

namespace hyle::compute::native {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4e41544956450001ULL};

struct Options final {
  ParallelExecutorRef executor{};
};

HYLE_COMPUTE_PROVIDER_NATIVE_API Provider
provider(Options options = {}) noexcept;

} // namespace hyle::compute::native

// Both factories are delivered by the same CPU provider factor and SDK seam.
// The independent identity keeps original ATen programs separately bindable.
namespace hyle::compute::aten_cpu {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4154454e43505501ULL};

// Uses the locked Torch CPU SDK; returns invalid when that feature is disabled.
// Ordinary callers use compute::cpu::create_context() for fixed rule binding.
HYLE_COMPUTE_PROVIDER_NATIVE_API Provider provider() noexcept;

} // namespace hyle::compute::aten_cpu
