#include "progress.h"

#include <chrono>
#include <utility>

#include "hyle/compute/backends/cuda.h"
#include "hyle/sched/sync.h"

namespace hyle::graph::cpu_cuda::detail {
namespace {

void CUDART_CB signal_ready(void *opaque) noexcept {
  static_cast<sched::Event *>(opaque)->signal();
}

} // namespace

Status wait_cuda_prefix(compute::Queue &queue, sched::Event &ready) noexcept {
  ready.reset();
  auto notification = compute::cuda::submit(
      queue, [&ready](cudaStream_t stream) noexcept {
        return cudaLaunchHostFunc(stream, &signal_ready, &ready);
      });
  if (!notification)
    return std::move(notification).error();
  // A CUDA error may suppress the host callback. The existing fiber-aware
  // timer wait provides a bounded error-observation fallback, not a busy loop.
  while (!ready.wait_for(std::chrono::milliseconds(1))) {
    auto state = notification->poll();
    if (!state)
      return std::move(state).error();
    if (*state == compute::CompletionPoll::kComplete)
      break;
  }
  // The prefix event is after the callback. Retire it before a slot can reset
  // or destroy ready, even if the fiber was woken before signal() returned.
  return notification->wait();
}

} // namespace hyle::graph::cpu_cuda::detail
