#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::graph::onednn {

inline constexpr compute::ProviderId kProviderId{0x48594c4500000001ULL,
                                                 0x474f4e45444e4e01ULL};

struct Options final {
  // Empty is an explicit caller-thread serial executor. A non-empty backing
  // is borrowed by the Context and every artifact/invocation made from it.
  compute::ParallelExecutorRef executor{};
};

#if defined(HYLE_GRAPH_ONEDNN)
// Graph-only provider: single-operation preparation deliberately returns
// Unimplemented. Put it before ordinary oneDNN/native providers in a Context
// when the four-node gated-MLP region should be considered at graph compile.
HYLE_GRAPH_PROVIDER_ONEDNN_API compute::Provider
provider(Options options = {}) noexcept;
#endif

} // namespace hyle::graph::onednn
