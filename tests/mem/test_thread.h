#pragma once

#include <optional>
#include <thread>
#include <utility>

namespace hyle::mem::test_support {

// Tests drive Mem's concurrent surfaces with the standard thread primitive so
// the Mem product does not acquire a product dependency merely from its test
// harness. Thread creation failure remains explicit to preserve the existing
// assertions on constrained validation hosts.
class Thread {
 public:
  Thread(Thread&&) noexcept = default;
  Thread& operator=(Thread&&) noexcept = default;
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;

  static std::optional<Thread> spawn(void (*entry)(void*), void* context) {
    try {
      return Thread(std::thread(entry, context));
    } catch (...) {
      return std::nullopt;
    }
  }

  void join() { thread_.join(); }

 private:
  explicit Thread(std::thread thread) noexcept
      : thread_(std::move(thread)) {}

  std::thread thread_;
};

}  // namespace hyle::mem::test_support
