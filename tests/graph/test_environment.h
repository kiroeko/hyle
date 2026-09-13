#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "hyle/graph/executable.h"

namespace hyle::graph::test {

// Migration fixture for tests whose old "domain" ordinal named a
// (Context, MemoryResource) pair. It deliberately deduplicates the two axes
// and gives each distinct target one compute lane, so legacy scenarios cannot
// accidentally preserve the product bug that Graph v7 removes.
struct ExecutionSpec final {
  const compute::Context *context = nullptr;
  mem::MemoryResourceRef storage{};
};

class Environment final {
public:
  Environment(std::initializer_list<ExecutionSpec> specifications) {
    for (const ExecutionSpec specification : specifications) {
      std::size_t storage = 0;
      while (storage < storage_domains_.size() &&
             storage_domains_[storage].resource != specification.storage) {
        ++storage;
      }
      if (storage == storage_domains_.size()) {
        storage_domains_.push_back(StorageDomain{
            .resource = specification.storage,
        });
      }

      std::size_t target = 0;
      while (target < targets_.size() &&
             targets_[target].context != specification.context) {
        ++target;
      }
      if (target == targets_.size()) {
        targets_.push_back(ExecutionTarget{
            .context = specification.context,
            .default_storage =
                StorageDomainId{static_cast<std::uint32_t>(storage)},
        });
        lanes_.push_back(ExecutionLane{
            .target = ExecutionTargetId{static_cast<std::uint32_t>(target)},
            .role = ExecutionLaneRole::kCompute,
        });
      }
      target_ids_.push_back(
          ExecutionTargetId{static_cast<std::uint32_t>(target)});
      storage_ids_.push_back(
          StorageDomainId{static_cast<std::uint32_t>(storage)});
      lane_ids_.push_back(ExecutionLaneId{static_cast<std::uint32_t>(target)});
    }
  }

  CompileEnvironment environment() const noexcept {
    return CompileEnvironment{
        .storage_domains = storage_domains_,
        .targets = targets_,
        .lanes = lanes_,
    };
  }

  ExecutionTargetId target(std::size_t old_ordinal) const noexcept {
    return target_ids_[old_ordinal];
  }
  StorageDomainId storage(std::size_t old_ordinal) const noexcept {
    return storage_ids_[old_ordinal];
  }
  ExecutionLaneId lane(std::size_t old_ordinal) const noexcept {
    return lane_ids_[old_ordinal];
  }

  std::size_t target_count() const noexcept { return targets_.size(); }
  std::size_t storage_count() const noexcept {
    return storage_domains_.size();
  }
  std::size_t lane_count() const noexcept { return lanes_.size(); }

private:
  std::vector<StorageDomain> storage_domains_{};
  std::vector<ExecutionTarget> targets_{};
  std::vector<ExecutionLane> lanes_{};
  std::vector<ExecutionTargetId> target_ids_{};
  std::vector<StorageDomainId> storage_ids_{};
  std::vector<ExecutionLaneId> lane_ids_{};
};

} // namespace hyle::graph::test
