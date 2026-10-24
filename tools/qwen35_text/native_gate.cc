#include "native_gate.h"
#include "backend.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#if defined(HYLE_COMPUTE_CUDA) && !defined(__APPLE__)
#include "hyle/compute/cuda.h"
#elif defined(HYLE_COMPUTE_VULKAN) && !defined(__APPLE__)
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#else
#error "Apple native gate is implemented in native_gate_apple.mm"
#endif

namespace qwen35::probe {
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
  hyle::Status terminal;
#if defined(HYLE_COMPUTE_CUDA)
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
#else
  std::shared_ptr<hyle::compute::vulkan::detail::Runtime> runtime;
  PFN_vkSignalSemaphore signal = nullptr;
  VkSemaphore semaphore = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool native_retired = false;
  struct Submission final { State *state; const VkSubmitInfo *info; };
  static VkResult submit(void *opaque, VkQueue queue) noexcept {
    const auto &submission = *static_cast<Submission *>(opaque);
    auto &state = *submission.state;
    return state.runtime->device->fn.QueueSubmit(queue, 1, submission.info, state.fence);
  }
#endif
  void release() noexcept {
    std::lock_guard lock(mutex);
    if (released) return;
    released = true;
#if !defined(HYLE_COMPUTE_CUDA)
    if (accepted) {
      VkSemaphoreSignalInfo info{};
      info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
      info.semaphore = semaphore; info.value = 1;
      terminal.Update(hyle::compute::vulkan::detail::native_status(
          signal(runtime->device->native.device, &info), "native gate vkSignalSemaphore"));
    }
#endif
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
#if !defined(HYLE_COMPUTE_CUDA)
  if (state_->accepted && !state_->native_retired) {
    // An observation error does not authorize destroying a pending semaphore,
    // fence or its Device. Keep their original owners on this failed-test path.
    std::cerr << "native_gate_unresolved_owner_retained status=" << hyle::status_code_name(status.code())
              << ": " << status.message() << '\n';
    (void)state_.release(); return;
  }
  auto &state = *state_;
  if (state.runtime) {
    const auto &device = *state.runtime->device;
    if (state.fence) device.fn.DestroyFence(device.native.device, state.fence, nullptr);
    if (state.semaphore) device.fn.DestroySemaphore(device.native.device, state.semaphore, nullptr);
  }
#else
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
#endif
}

hyle::Status NativeGate::arm() noexcept {
  auto &state = *state_;
  if (state.accepted || state.watchdog.joinable())
    return hyle::Status::FailedPrecondition("native gate is already armed");
  try {
#if defined(HYLE_COMPUTE_CUDA)
    state.start_watchdog();
    auto submitted = hyle::compute::cuda::submit(state.backend.common_queue(),
        [&](cudaStream_t stream) noexcept { return cudaLaunchHostFunc(stream, &State::callback, &state); });
    if (!submitted) { state.release(); return submitted.error(); }
    state.completion = std::move(*submitted); state.accepted = true;
#else
    namespace vd = hyle::compute::vulkan::detail;
    state.runtime = vd::Access::runtime(state.backend.native_probe_context());
    if (!state.runtime || !state.runtime->device->provider.native().timeline_semaphore)
      return hyle::Status::Unimplemented("original Vulkan device did not enable timeline semaphores");
    {
      std::lock_guard lock(state.runtime->mutex);
      if (state.runtime->closed || !state.runtime->failure.ok() || state.runtime->pending)
        return hyle::Status::FailedPrecondition("native gate requires the original healthy idle Runtime");
    }
    const auto &device = *state.runtime->device;
    auto procedure = device.instance_fn.GetDeviceProcAddr(device.native.device, "vkSignalSemaphore");
    if (!procedure) procedure = device.instance_fn.GetDeviceProcAddr(device.native.device, "vkSignalSemaphoreKHR");
    state.signal = reinterpret_cast<PFN_vkSignalSemaphore>(procedure);
    if (!state.signal) return hyle::Status::Unimplemented("original Vulkan device has no host timeline signal");
    VkSemaphoreTypeCreateInfo type{}; type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; type.initialValue = 0;
    VkSemaphoreCreateInfo semaphore{}; semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; semaphore.pNext = &type;
    HYLE_RETURN_IF_ERROR(vd::native_status(device.fn.CreateSemaphore(device.native.device, &semaphore, nullptr,
        &state.semaphore), "native gate vkCreateSemaphore"));
    VkFenceCreateInfo fence{}; fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    HYLE_RETURN_IF_ERROR(vd::native_status(device.fn.CreateFence(device.native.device, &fence, nullptr,
        &state.fence), "native gate vkCreateFence"));
    // Only the existing native queue lock is held during vkQueueSubmit. No
    // wait/poll/callback into HYLE occurs under that lock.
    const std::uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo timeline{}; timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount = 1; timeline.pWaitSemaphoreValues = &value;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info{}; info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; info.pNext = &timeline;
    info.waitSemaphoreCount = 1; info.pWaitSemaphores = &state.semaphore; info.pWaitDstStageMask = &stage;
    State::Submission submission{&state, &info};
    HYLE_RETURN_IF_ERROR(device.provider.with_queue(&State::submit, &submission));
    state.accepted = true;
    state.start_watchdog();
#endif
    return hyle::Status::Ok();
  } catch (...) {
    state.release();
    return hyle::Status::ResourceExhausted("native gate host owner/watchdog allocation failed");
  }
}

hyle::Status NativeGate::wait_until_entered() noexcept {
  auto &state = *state_;
  if (!state.accepted) return hyle::Status::FailedPrecondition("native gate was not accepted");
#if defined(HYLE_COMPUTE_CUDA)
  std::unique_lock lock(state.mutex);
  if (!state.changed.wait_for(lock, timeout, [&] { return state.entered || state.expired; }))
    return hyle::Status::DeadlineExceeded("original CUDA Queue did not enter native gate");
  return state.entered && !state.expired ? hyle::Status::Ok()
      : hyle::Status::DeadlineExceeded("native gate watchdog opened before assertion");
#else
  const auto &device = *state.runtime->device;
  const auto observed = device.fn.GetFenceStatus(device.native.device, state.fence);
  if (observed != VK_NOT_READY)
    return hyle::Status::Internal("accepted Vulkan timeline gate was not actually pending");
  std::lock_guard lock(state.mutex);
  state.entered = true;
  return !state.expired && !state.released ? hyle::Status::Ok()
      : hyle::Status::DeadlineExceeded("native gate opened before assertion");
#endif
}

hyle::Status NativeGate::wait_for_model_submission() noexcept {
#if !defined(HYLE_COMPUTE_CUDA)
  auto &state = *state_;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock(state.mutex);
      if (state.expired || state.released)
        return hyle::Status::DeadlineExceeded("Vulkan gate opened before a Model native Run was accepted");
    }
    // A gate's raw submit is deliberately absent from this list. A member is
    // inserted only after the original Model Run accepted native work. First
    // scratch growth can hold this mutex while synchronizing: never block on it.
    std::unique_lock lock(state.runtime->mutex, std::try_to_lock);
    if (lock.owns_lock() && state.runtime->pending != nullptr) {
      std::cerr << "native_gate_model_admitted backend=vulkan proof=original_runtime_pending_run"
                << " runtime=" << state.runtime.get() << " run=" << state.runtime->pending << '\n';
      return hyle::Status::Ok();
    }
    if (lock.owns_lock()) lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return hyle::Status::DeadlineExceeded("no accepted original Model native Run behind Vulkan gate");
#else
  return hyle::Status::Ok();
#endif
}

void NativeGate::release() noexcept { state_->release(); }
hyle::Status NativeGate::wait() noexcept {
  auto &state = *state_;
  if (!state.accepted) return hyle::Status::Ok();
#if defined(HYLE_COMPUTE_CUDA)
  return state.completion.wait();
#else
  const auto &device = *state.runtime->device;
  const auto observed = device.fn.WaitForFences(device.native.device, 1, &state.fence, VK_TRUE, 30000000000ULL);
  if (observed == VK_SUCCESS || observed == VK_ERROR_DEVICE_LOST) state.native_retired = true;
  if (observed == VK_TIMEOUT) return hyle::Status::DeadlineExceeded("native Vulkan gate did not retire after release");
  HYLE_RETURN_IF_ERROR(hyle::compute::vulkan::detail::native_status(observed, "native gate vkWaitForFences"));
  std::lock_guard lock(state.mutex);
  return state.terminal;
#endif
}
bool NativeGate::timed_out() const noexcept {
  std::lock_guard lock(state_->mutex); return state_->expired;
}
const char *NativeGate::name() noexcept {
#if defined(HYLE_COMPUTE_CUDA)
  return "original_cuda_host_callback";
#else
  return "original_vulkan_queue_host_signaled_timeline";
#endif
}
} // namespace qwen35::probe
