#pragma once

#include <utility>

#include "hyle/graph/detail/region_access.h"

namespace hyle::graph::detail {
struct InvocationState;

// Optional assembly over the original slot. Core knows neither its Scheduler
// nor its task types. This bridge owns no graph, bindings or result storage.
struct ProgressVTable final {
  void (*destroy)(void *) noexcept;
  Result<compute::Completion> (*submit)(void *, Invocation &,
      std::span<const QueueBinding>, OrderedGraphSubmission) noexcept;
  Status (*wait)(void *, InvocationState &) noexcept;
  void (*notify)(void *) noexcept;
};

class ProgressOwner final {
public:
  ProgressOwner() noexcept = default;
  ProgressOwner(void *state, const ProgressVTable *table) noexcept
      : state_(state), table_(table) {}
  ProgressOwner(const ProgressOwner &) = delete;
  ProgressOwner &operator=(const ProgressOwner &) = delete;
  ProgressOwner(ProgressOwner &&other) noexcept
      : state_(std::exchange(other.state_, nullptr)), table_(other.table_) {}
  ProgressOwner &operator=(ProgressOwner &&other) noexcept {
    if (this != &other) {
      reset();
      state_ = std::exchange(other.state_, nullptr);
      table_ = other.table_;
    }
    return *this;
  }
  ~ProgressOwner() { reset(); }
  explicit operator bool() const noexcept { return state_ != nullptr; }
  void *get() const noexcept { return state_; }
  const ProgressVTable *table() const noexcept { return table_; }
  Result<compute::Completion> submit(Invocation &invocation,
      std::span<const QueueBinding> queues, OrderedGraphSubmission ordered) const noexcept {
    return table_->submit(state_, invocation, queues, ordered);
  }
  Status wait(InvocationState &invocation) const noexcept {
    return table_->wait(state_, invocation);
  }
  void notify() const noexcept {
    if (state_) table_->notify(state_);
  }
  void reset() noexcept {
    if (auto *state = std::exchange(state_, nullptr)) table_->destroy(state);
    table_ = nullptr;
  }

private:
  void *state_ = nullptr;
  const ProgressVTable *table_ = nullptr;
};
} // namespace hyle::graph::detail
