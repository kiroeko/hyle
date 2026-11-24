#include "native_gate.h"
#include "../../inference/backend.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include "hyle/compute/cuda.h"

namespace qwen35::probe {
using hyle::inference::Backend;
namespace {
constexpr auto timeout = std::chrono::seconds(30);
}

struct NativeGate::State final {
  explicit State(Backend &owner) : backend(owner) {}
  Backend &backend;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  bool entered = false, released = false, expired = false, accepted = false;
  hyle::compute::Completion completion;
  bool callback_finished = false;
  static void CUDART_CB callback(void *opaque) {
    auto &state = *static_cast<State *>(opaque);
    std::unique_lock lock(state.mutex);
    state.entered = true; state.changed.notify_all();
    state.changed.wait(lock, [&] { return state.released; });
    // The observer may fail without proving native retirement. Only this
    // callback can prove it will no longer access its borrowed State pointer.
    state.callback_finished = true;
  }
  void release() noexcept {
    std::lock_guard lock(mutex);
    if (released) return;
    released = true;
    changed.notify_all();
  }
  void start_watchdog() {
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (changed.wait_for(lock, timeout, [&] { return released; })) return;
      expired = true;
      lock.unlock(); release();
    });
  }
};

NativeGate::NativeGate(Backend &backend) : state_(std::make_unique<State>(backend)) {}
NativeGate::~NativeGate() {
  release();
  if (state_->watchdog.joinable()) state_->watchdog.join();
  const auto status = wait();
  if (!status.ok()) {
    bool callback_finished;
    { std::lock_guard lock(state_->mutex); callback_finished = state_->callback_finished; }
    if (!callback_finished) {
      std::cerr << "native_gate_unresolved_callback_owner_retained status="
                << hyle::status_code_name(status.code()) << ": " << status.message() << '\n';
      // Keep the Completion/EventSlot and its original Queue/Context owner as
      // well as the callback payload. A failed wait is not a native proof.
      (void)state_.release(); return;
    }
  }
  status.IgnoreError();
}

hyle::Status NativeGate::arm() noexcept {
  auto &state = *state_;
  if (state.accepted || state.watchdog.joinable())
    return hyle::Status::FailedPrecondition("native gate is already armed");
  try {
    HYLE_ASSIGN_OR_RETURN(auto queue, state.backend.common_queue());
    state.start_watchdog();
    auto submitted = hyle::compute::cuda::submit(*queue,
        [&](cudaStream_t stream) noexcept { return cudaLaunchHostFunc(stream, &State::callback, &state); });
    if (!submitted) { state.release(); return submitted.error(); }
    state.completion = std::move(*submitted); state.accepted = true;
    return hyle::Status::Ok();
  } catch (...) {
    state.release();
    return hyle::Status::ResourceExhausted("native gate host owner/watchdog allocation failed");
  }
}

hyle::Status NativeGate::wait_until_entered() noexcept {
  auto &state = *state_;
  if (!state.accepted) return hyle::Status::FailedPrecondition("native gate was not accepted");
  std::unique_lock lock(state.mutex);
  if (!state.changed.wait_for(lock, timeout, [&] { return state.entered || state.expired; }))
    return hyle::Status::DeadlineExceeded("original CUDA Queue did not enter native gate");
  return state.entered && !state.expired ? hyle::Status::Ok()
      : hyle::Status::DeadlineExceeded("native gate watchdog opened before assertion");
}

hyle::Status NativeGate::wait_for_model_submission() noexcept {
  return hyle::Status::Ok();
}

void NativeGate::release() noexcept { state_->release(); }
hyle::Status NativeGate::wait() noexcept {
  auto &state = *state_;
  if (!state.accepted) return hyle::Status::Ok();
  return state.completion.wait();
}
bool NativeGate::timed_out() const noexcept {
  std::lock_guard lock(state_->mutex); return state_->expired;
}
const char *NativeGate::name() noexcept {
  return "original_cuda_host_callback";
}
} // namespace qwen35::probe
