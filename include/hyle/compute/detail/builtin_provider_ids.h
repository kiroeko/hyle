#pragma once

#include "hyle/compute/provider.h"

namespace hyle::compute::detail {

// Stable identities are available without the optional provider factories.
inline constexpr ProviderId kNativeProviderId{0x48594c4500000001ULL,
                                             0x4e41544956450001ULL};
inline constexpr ProviderId kCudaNativeProviderId{0x48594c4500000001ULL,
                                                 0x435544414b455201ULL};

} // namespace hyle::compute::detail
