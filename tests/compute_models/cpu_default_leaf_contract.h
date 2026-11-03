#pragma once

#include "hyle/compute/operation.h"
#include "hyle/compute/provider.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/route_policy.h"

namespace cpu_default_leaf_contract {
namespace c = hyle::compute;

// These are the exact existing default-policy identities, not permission to
// substitute arbitrary native arithmetic for the independent model oracle.
inline bool selected(c::ProviderId provider, c::OperationKind kind,
                     c::ProviderImplementationId implementation,
                     c::RoutePolicyId policy) noexcept {
  if (provider != c::native::kProviderId || !policy.valid() ||
      implementation.value() != static_cast<std::uint64_t>(kind) + 1)
    return false;
  constexpr std::uint64_t engineering = 0x48594c4543505544ULL;
  if (kind == c::OperationKind::kGather)
    return policy.version == 1 &&
           ((policy.namespace_id == 0x48594c4553544f52ULL &&
             policy.policy_id == 1) ||
            (policy.namespace_id == engineering && policy.policy_id == 1 +
                (static_cast<std::uint64_t>(kind) << 16) + (2ULL << 8) + 4));
  constexpr auto native = std::uint64_t{4};
  return kind == c::OperationKind::kUnary && policy.version == 1 &&
         policy.namespace_id == engineering &&
         policy.policy_id == 1 +
             (static_cast<std::uint64_t>(c::OperationKind::kUnary) << 16) +
             (native << 8) + native;
}
} // namespace cpu_default_leaf_contract
