#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_FLASHINFER_API
#elif defined(HYLE_COMPUTE_PROVIDER_FLASHINFER_BUILD)
#define HYLE_COMPUTE_PROVIDER_FLASHINFER_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_FLASHINFER_API HYLE_IMPORT
#endif

namespace hyle::compute::flashinfer {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x464c415348490001ULL};

// Pinned upstream attention on qualified Linux CUDA SM89 devices. The initial
// domain is dense BSHD FP16/BF16, Q/KV heads 8/2, head depth 256, no extra
// mask, and bottom-right causal attention with KV length >= query length.
// Invocation borrows caller workspace and payload through real Completion.
// Replay is managed: upstream dispatch includes device/occupancy/attribute
// queries, but no runtime compilation, timing search or constant packing.
// Separate invocations have independent workspace and may run concurrently.
#if defined(HYLE_COMPUTE_FLASHINFER)
HYLE_COMPUTE_PROVIDER_FLASHINFER_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::flashinfer
