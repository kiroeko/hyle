#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>

#include "hyle/compute/apple.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"

namespace hyle::compute::apple::detail {
struct Job {
  void (*execute)(Job *) noexcept = nullptr;
  Job *next = nullptr;
};
// The original Runtime/Job owner, shared by native math and F2 blit. No new
// thread, queue or progress engine is constructed by the copy capability.
struct Runtime {
  std::mutex queue_mutex;
  std::mutex close_mutex;
  platform::Parker wake;
  platform::Thread thread;
  std::atomic<std::uint64_t> owner_id{0};
  Job *head = nullptr;
  Job *tail = nullptr;
  bool accepting = true;
  Status failure;
  Environment environment;
  std::optional<mlx::core::Stream> stream;
  Status enqueue(Job &job) noexcept;
  void poison(Status terminal) noexcept;
  void run() noexcept;
  Status close() noexcept;
};
Status qualify_copy_dependency(const std::shared_ptr<Runtime> &runtime,
    compute::detail::CompletionDependency source) noexcept;
} // namespace hyle::compute::apple::detail
