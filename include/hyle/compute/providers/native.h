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
