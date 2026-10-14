#define HYLE_TESTING 1
#include "../../src/compute_backend_cuda/cuda_internal.h"
#include <hyle/compute/detail/copy_stream_access.h>
#include "owned_outcome_completion_fixture.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <thread>

namespace {
namespace c = hyle::compute;
namespace n = c::cuda;
using hyle::Status;
Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
struct Gate final {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false, released = false;
  void wait() {
    std::unique_lock lock(mutex);
    entered = true; condition.notify_all();
    condition.wait(lock, [&] { return released; });
  }
  bool await() {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, std::chrono::seconds(3), [&] { return entered; });
  }
  void release() {
    std::lock_guard lock(mutex);
    released = true; condition.notify_all();
  }
};
thread_local bool fail_query = false;
cudaError_t query(cudaEvent_t event) noexcept {
  const auto actual = cudaEventQuery(event);
  if (std::exchange(fail_query, false) && actual == cudaErrorNotReady)
    return cudaErrorMemoryAllocation;
  return actual;
}
Status episode(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto id, n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(id));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), 4096));
  HYLE_ASSIGN_OR_RETURN(auto source, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto receiver, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto source_pointer, n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto receiver_pointer, n::pointer<std::byte>(receiver.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto uses, c::LeaseUsePool::create());
  HYLE_ASSIGN_OR_RETURN(auto writing, uses.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source));
  const std::array specs{c::CopyResourceSpec{device.ref(), 1024},
                         c::CopyResourceSpec{device.ref(), 1024}};
  HYLE_ASSIGN_OR_RETURN(auto program, c::CopyBatchProgram::prepare(context, specs));
  HYLE_ASSIGN_OR_RETURN(auto copy, c::CopyBatchInvocation::create(program, queue));
  HYLE_ASSIGN_OR_RETURN(auto result_slot, c::detail::ResultSlot<c::WriteLease>::create(hyle::mem::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(auto result_write, result_slot.begin());
  Gate gate;
  struct Release final {
    Gate &gate; c::Context &context;
    ~Release() { gate.release(); context.drain().IgnoreError(); }
  } release{gate, context};
  HYLE_ASSIGN_OR_RETURN(auto producer, n::submit(queue, [&](cudaStream_t stream) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream,
        [](void *opaque) { static_cast<Gate *>(opaque)->wait(); }, &gate)));
    return n::status(cudaMemsetAsync(source_pointer, 49, 1024, stream));
  }));
  HYLE_RETURN_IF_ERROR(writing.attach(producer));
  HYLE_RETURN_IF_ERROR(require(gate.await(), "CUDA prefix did not enter actual checkpoint"));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({0}, std::move(source)));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(destination)));
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0}, std::move(producer)));
  const std::array<c::CopyPhase, 2> phases{{{}, {}}};
  const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 1024}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
  HYLE_ASSIGN_OR_RETURN(auto submitted, copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update, c::CopyBatchUpdate::take(std::move(copy), std::move(submitted)));
  HYLE_ASSIGN_OR_RETURN(auto pending, c::detail::CopyStreamAccess::input(update, {1}));
  HYLE_ASSIGN_OR_RETURN(auto input_pointer, n::pointer<std::byte>(pending.view));
  HYLE_ASSIGN_OR_RETURN(auto reading, uses.begin());
  HYLE_RETURN_IF_ERROR(c::detail::CopyStreamAccess::admit_read(reading, update, {1}));
  HYLE_RETURN_IF_ERROR(reading.write(receiver));
  HYLE_RETURN_IF_ERROR(result_write.emplace(std::move(receiver)));
  // The same original ordered Queue accepts the suffix after F2, with exact
  // original source and destination guards. No intermediate host wait.
  HYLE_ASSIGN_OR_RETURN(auto suffix, n::submit(queue, [&](cudaStream_t stream) noexcept {
    return n::status(cudaMemcpyAsync(receiver_pointer, input_pointer, 1024,
                                    cudaMemcpyDeviceToDevice, stream));
  }));
  HYLE_RETURN_IF_ERROR(reading.attach(suffix));
  HYLE_ASSIGN_OR_RETURN(auto result, result_write.publish(std::move(suffix)));
  using Aggregate = hyle_g03_outcome_probe::Outcome<c::WriteLease, c::CopyBatchUpdate>;
  HYLE_ASSIGN_OR_RETURN(auto joined, Aggregate::take<c::detail::CopyStreamAccess>(
      std::move(result), std::move(update)));
  std::optional<Aggregate> aggregate(std::move(joined));
  auto watch = aggregate->watch();
  HYLE_RETURN_IF_ERROR(require(!aggregate->result() && !watch.expired() &&
      pool.stats().active_uses > 0, "pending aggregate fabricated ready result"));
  if (mode == 1)
    HYLE_RETURN_IF_ERROR(require(aggregate->cancel_source().code() == hyle::StatusCode::Unimplemented,
        "CUDA unsupported cancellation changed"));
  if (mode == 2 || mode == 3) {
    const auto previous = n::detail::set_completion_observation_hooks({&query, nullptr});
    fail_query = true;
    auto observed = aggregate->completion().poll();
    n::detail::set_completion_observation_hooks(previous);
    HYLE_RETURN_IF_ERROR(require(observed && *observed == c::CompletionPoll::kPending &&
        !fail_query && !watch.expired() && pool.stats().active_uses > 0,
        "unknown suffix error fabricated retirement"));
  }
  if (mode == 3) {
    aggregate.reset();
    HYLE_RETURN_IF_ERROR(require(!watch.expired(), "abandon released borrowed source early"));
    gate.release();
    HYLE_RETURN_IF_ERROR(context.drain());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!watch.expired() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HYLE_RETURN_IF_ERROR(require(watch.expired() && pool.stats().active_uses == 0,
        "original CUDA reaper did not release aggregate source/guards"));
  } else {
    gate.release();
    auto ready = aggregate->wait();
    if (mode == 2) {
      HYLE_RETURN_IF_ERROR(require(!ready && ready.error().IsResourceExhausted() &&
          aggregate->result().error().IsResourceExhausted(), "first native OOM verdict lost"));
    } else {
      if (!ready) return ready.error();
      HYLE_ASSIGN_OR_RETURN(auto pointer, n::pointer<std::byte>((*ready)->view().value()));
      std::array<std::byte, 1024> bytes{};
      HYLE_RETURN_IF_ERROR(n::status(cudaMemcpy(bytes.data(), pointer, bytes.size(), cudaMemcpyDeviceToHost)));
      for (auto byte : bytes)
        HYLE_RETURN_IF_ERROR(require(byte == std::byte{49}, "actual producer/copy/suffix byte oracle differs"));
      HYLE_RETURN_IF_ERROR(require(!result_slot.begin(), "old result permits slot overwrite"));
    }
  }
  std::fprintf(stderr, "CUDA original producer/F2/suffix aggregate mode=%u PASS:1024 bytes, actual retirement, original result/guards\n", mode);
  return Status::Ok();
}
Status run() {
  for (unsigned mode = 0; mode < 4; ++mode) HYLE_RETURN_IF_ERROR(episode(mode));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
