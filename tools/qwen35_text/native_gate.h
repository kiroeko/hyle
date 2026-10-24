#pragma once

#include <memory>
#include "hyle/error/status.h"

namespace qwen35 {
class Backend;
namespace probe {

// Test instrument on the Backend's original native endpoint. The watchdog
// only prevents a failed test from hanging; opening it makes the test fail.
// It is not a Completion and never supplies retirement proof for Graph work.
class NativeGate final {
public:
  explicit NativeGate(Backend &);
  ~NativeGate();
  NativeGate(const NativeGate &) = delete;
  NativeGate &operator=(const NativeGate &) = delete;
  hyle::Status arm() noexcept;
  hyle::Status wait_until_entered() noexcept;
  // After original Graph submission, require its native accepted owner too.
  // Vulkan: original Runtime.pending Run; Apple: original FIFO accepted Job.
  hyle::Status wait_for_model_submission() noexcept;
  void release() noexcept;
  hyle::Status wait() noexcept;
  bool timed_out() const noexcept;
  static const char *name() noexcept;
private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace probe
} // namespace qwen35
