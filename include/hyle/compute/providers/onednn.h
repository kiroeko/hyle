#pragma once

#include <string_view>

#include "hyle/compute/provider.h"
#include "hyle/compute/visibility.h"

namespace hyle::compute::onednn {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4f4e45444e4e0001ULL};

struct Options final {
  // Empty is an explicit caller-thread serial executor. A non-empty backing is
  // borrowed through every Context/artifact/invocation made from the provider.
  ParallelExecutorRef executor{};
};

#if defined(HYLE_COMPUTE_ONEDNN)
HYLE_COMPUTE_PROVIDER_ONEDNN_API Provider
provider(Options options = {}) noexcept;
HYLE_COMPUTE_PROVIDER_ONEDNN_API std::string_view version() noexcept;
#endif

} // namespace hyle::compute::onednn
