#pragma once

// Private observation-status seam. All TUs of its instrumented executable use
// this definition; normal product builds contain neither hooks nor branches.
#if !defined(HYLE_TEST_INFERENCE_WARM_COST) || !HYLE_TEST_INFERENCE_WARM_COST
#error "The readback batch probe is only for its instrumented test target"
#endif

#include "hyle/compute/batch_transfer.h"

namespace hyle::inference::testing {

struct ReadbackBatchHooks final {
  void *state = nullptr;
  Status (*before_submit)(void *, compute::Queue &) noexcept = nullptr;
  Status (*wait)(void *, compute::CopyBatchUpdate &) noexcept = nullptr;
};

inline thread_local ReadbackBatchHooks readback_batch_hooks;

class ReadbackBatchHookScope final {
public:
  explicit ReadbackBatchHookScope(ReadbackBatchHooks hooks) noexcept
      : previous_(readback_batch_hooks) { readback_batch_hooks = hooks; }
  ~ReadbackBatchHookScope() { readback_batch_hooks = previous_; }
  ReadbackBatchHookScope(const ReadbackBatchHookScope &) = delete;
  ReadbackBatchHookScope &operator=(const ReadbackBatchHookScope &) = delete;
private:
  ReadbackBatchHooks previous_;
};

inline Status readback_batch_before_submit(compute::Queue &queue) noexcept {
  const auto hook = readback_batch_hooks;
  return hook.before_submit ? hook.before_submit(hook.state, queue) : Status::Ok();
}

inline Status readback_batch_wait(compute::CopyBatchUpdate &update) noexcept {
  const auto hook = readback_batch_hooks;
  return hook.wait ? hook.wait(hook.state, update) : update.wait();
}

} // namespace hyle::inference::testing
