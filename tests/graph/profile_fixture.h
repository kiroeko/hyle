#pragma once

#include <vector>

#include "hyle/graph/executable.h"

namespace hyle::graph::test {

// Synthetic identity/timing fixtures test routing contracts, NOT qualified
// measurements or a claim about the performance of any real implementation.
inline compute::ProfileFingerprint profile_fingerprint(std::uint64_t word) {
  return {{word, 0, 0, 0}};
}

struct ProfileFixture final {
  explicit ProfileFixture(CompileEnvironment environment) {
    for (std::size_t t = 0; t < environment.targets.size(); ++t) {
      const auto &owner = *environment.targets[t].context;
      targets.push_back({owner.device().backend_id(), profile_fingerprint(1),
                         profile_fingerprint(2 + t), profile_fingerprint(3),
                         profile_fingerprint(4)});
      for (std::size_t p = 0; p < owner.provider_count(); ++p) {
        providers.push_back({ExecutionTargetId{static_cast<std::uint32_t>(t)},
                             p, owner.provider_instance_id(p).value(),
                             profile_fingerprint(10 + p),
                             profile_fingerprint(20 + p)});
      }
    }
    for (std::size_t i = 0; i < environment.storage_domains.size(); ++i)
      resources.push_back(profile_fingerprint(30 + i));
    context = {.hyle_build = profile_fingerprint(41),
               .semantics = profile_fingerprint(42),
               .workload_data = profile_fingerprint(43),
               .replay_configuration = profile_fingerprint(44),
               .targets = targets,
               .storage_domains = resources,
               .providers = providers};
  }
  ProfileFixture(const ProfileFixture &) = delete;
  ProfileFixture &operator=(const ProfileFixture &) = delete;
  std::vector<compute::ProfileTargetIdentity> targets;
  std::vector<compute::ProfileFingerprint> resources;
  std::vector<ProfileProviderBinding> providers;
  CompileProfileContext context;
};

} // namespace hyle::graph::test
