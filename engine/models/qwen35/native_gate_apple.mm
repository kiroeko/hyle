#include "native_gate.h"
#include "../../inference/backend.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <mlx/backend/metal/device.h>
#include "../../../src/compute_route_apple/apple_runtime_internal.h"

namespace qwen35::probe {
using hyle::inference::Backend;
namespace ad = hyle::compute::apple::detail;
namespace {
constexpr auto timeout = std::chrono::seconds(30);
}

struct NativeGate::State final : ad::Job {
  explicit State(Backend &backend) {
    auto context = backend.apple_context();
    if (!context) throw std::runtime_error(std::string(context.error().message()));
    runtime = ad::ContextAccess::runtime(**context);
    execute = &execute_gate;
  }
  std::shared_ptr<ad::Runtime> runtime;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  NS::SharedPtr<MTL::SharedEvent> event;
  NS::SharedPtr<MTL::CommandBuffer> command;
  bool accepted = false, entered = false, released = false, expired = false, finished = false;
  hyle::Status terminal;
  void release() noexcept {
    std::lock_guard lock(mutex);
    if (released) return;
    released = true;
    if (event) event->setSignaledValue(1);
    changed.notify_all();
  }
  void start_watchdog() {
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (changed.wait_for(lock, timeout, [&] { return released; })) return;
      expired = true; lock.unlock(); release();
    });
  }
  static void execute_gate(ad::Job *opaque) noexcept {
    auto &state = *static_cast<State *>(opaque);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    hyle::Status terminal;
    const auto began = std::chrono::steady_clock::now();
    const auto diagnostic = [&](const char *stage) noexcept {
      std::lock_guard lock(state.mutex);
      std::cerr << "native_gate_apple stage=" << stage
                << " elapsed_seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count()
                << " command_status=" << static_cast<unsigned>(state.command->status())
                << " event_value=" << state.event->signaledValue()
                << " released=" << state.released << " expired=" << state.expired;
      if (const auto *error = state.command->error()) {
        const auto *domain = error->domain();
        const auto *description = error->localizedDescription();
        std::cerr << " error_domain=" << (domain ? domain->utf8String() : "<null>")
                  << " error_code=" << error->code()
                  << " error_description=" << (description ? description->utf8String() : "<null>");
      } else {
        std::cerr << " error=none";
      }
      std::cerr << '\n';
    };
    try {
      if (!state.runtime->stream) throw std::runtime_error("original Apple Runtime has no stream");
      auto &encoder = mlx::core::metal::get_command_encoder(*state.runtime->stream);
      encoder.end_encoding();
      // Both native owners come from this Runtime's existing MLX stream/device.
      // No MTLCommandQueue, Context or replacement evaluator is constructed.
      const auto device = mlx::core::metal::device(state.runtime->stream->device).mtl_device();
      {
        std::lock_guard lock(state.mutex);
        state.event = NS::TransferPtr(device->newSharedEvent());
        state.command = NS::RetainPtr(encoder.get_command_buffer());
        if (!state.event || !state.command) throw std::bad_alloc();
        if (state.released) state.event->setSignaledValue(1);
      }
      state.command->encodeWait(state.event.get(), 1);
      diagnostic("before_commit");
      encoder.commit(); // Original encoder commits on its original stream queue.
      diagnostic("after_commit");
      {
        std::lock_guard lock(state.mutex);
        const auto status = state.command->status();
        if (status == MTL::CommandBufferStatusError)
          terminal = hyle::Status::Internal("Metal gate command failed after commit");
        else if (!state.released &&
                 (status == MTL::CommandBufferStatusCommitted ||
                  status == MTL::CommandBufferStatusScheduled))
          state.entered = true;
        state.changed.notify_all();
      }
      // The original FIFO owner stays in this actual native wait. Later Model
      // Jobs may be accepted, but cannot retire until this command completes.
      state.command->waitUntilCompleted();
      if (state.command->status() == MTL::CommandBufferStatusError)
        terminal.Update(hyle::Status::Internal("Metal native event gate command failed"));
    } catch (...) {
      terminal = hyle::Status::Internal("Apple native event gate construction/commit failed");
      state.release();
      // If a native prefix was accepted, release the wait before synchronizing
      // that original encoder. A failure cannot make queued owners disappear.
      try { if (state.runtime->stream) mlx::core::metal::get_command_encoder(*state.runtime->stream).synchronize(); }
      catch (...) {}
      state.runtime->poison(terminal);
    }
    std::lock_guard lock(state.mutex);
    state.terminal.Update(terminal); state.finished = true; state.changed.notify_all();
  }
};

NativeGate::NativeGate(Backend &backend) : state_(std::make_unique<State>(backend)) {}
NativeGate::~NativeGate() {
  release();
  if (state_->watchdog.joinable()) state_->watchdog.join();
  const auto status = wait();
  bool unresolved;
  { std::lock_guard lock(state_->mutex); unresolved = state_->accepted && !state_->finished; }
  if (unresolved) {
    std::cerr << "native_gate_unresolved_owner_retained status=" << hyle::status_code_name(status.code())
              << ": " << status.message() << '\n';
    (void)state_.release(); // Original Job/device/event remain alive on unknown retirement.
  }
}
hyle::Status NativeGate::arm() noexcept {
  auto &state = *state_;
  if (state.accepted || state.watchdog.joinable()) return hyle::Status::FailedPrecondition("native gate is already armed");
  try {
    {
      std::lock_guard lock(state.runtime->queue_mutex);
      if (!state.runtime->accepting || !state.runtime->failure.ok() || state.runtime->head)
        return hyle::Status::FailedPrecondition("native gate requires the original healthy idle FIFO");
    }
    state.start_watchdog();
    const auto admitted = state.runtime->enqueue(state);
    if (!admitted.ok()) { state.release(); return admitted; }
    state.accepted = true;
    return hyle::Status::Ok();
  } catch (...) {
    state.release(); return hyle::Status::ResourceExhausted("Apple gate watchdog allocation failed");
  }
}
hyle::Status NativeGate::wait_until_entered() noexcept {
  auto &state = *state_;
  std::unique_lock lock(state.mutex);
  if (!state.changed.wait_for(lock, timeout, [&] { return state.entered || state.finished || state.expired; }))
    return hyle::Status::DeadlineExceeded("original Metal stream did not commit native event wait");
  HYLE_RETURN_IF_ERROR(state.terminal);
  return state.entered && !state.released && !state.expired ? hyle::Status::Ok()
      : hyle::Status::DeadlineExceeded("Metal native gate opened before assertion");
}
hyle::Status NativeGate::wait_for_model_submission() noexcept {
  auto &state = *state_;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock(state.mutex);
      if (state.expired || state.released || state.finished)
        return hyle::Status::DeadlineExceeded("Metal gate opened before original Model Job admission");
    }
    {
      std::lock_guard lock(state.runtime->queue_mutex);
      // All preparation was completed before this gate. The gate is executing
      // on the FIFO owner, so a later queued Job is actual Model native work.
      if (state.runtime->head != nullptr) {
        std::cerr << "native_gate_model_admitted backend=apple proof=original_fifo_accepted_job"
                  << " runtime=" << state.runtime.get() << " job=" << state.runtime->head << '\n';
        return hyle::Status::Ok();
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return hyle::Status::DeadlineExceeded("original Model did not enqueue a native Job behind Metal gate");
}
void NativeGate::release() noexcept { state_->release(); }
hyle::Status NativeGate::wait() noexcept {
  auto &state = *state_;
  if (!state.accepted) return hyle::Status::Ok();
  std::unique_lock lock(state.mutex);
  if (!state.changed.wait_for(lock, timeout, [&] { return state.finished; }))
    return hyle::Status::DeadlineExceeded("original Metal gate Job did not retire after release");
  return state.terminal;
}
bool NativeGate::timed_out() const noexcept { std::lock_guard lock(state_->mutex); return state_->expired; }
const char *NativeGate::name() noexcept { return "original_apple_fifo_and_stream_shared_event"; }
} // namespace qwen35::probe
