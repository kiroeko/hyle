#include "hyle/io/signal.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <utility>

#include "hyle/core/assert.h"
#include "hyle/core/environment.h"
#include "hyle/io/operation.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/waker.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hyle::io {
namespace {

constexpr unsigned mask(SignalInterest interest) noexcept {
  return static_cast<unsigned>(interest);
}

bool take_pending(std::atomic<unsigned> &pending, SignalInterest interest,
                  ProcessSignal *output) noexcept {
  unsigned observed = pending.load(std::memory_order_acquire);
  for (;;) {
    const unsigned available = observed & mask(interest);
    if (available == 0)
      return false;
    const unsigned selected =
        (available & mask(SignalInterest::kInterrupt)) != 0
            ? mask(SignalInterest::kInterrupt)
            : mask(SignalInterest::kTerminate);
    if (pending.compare_exchange_weak(observed, observed & ~selected,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      *output = selected == mask(SignalInterest::kInterrupt)
                    ? ProcessSignal::kInterrupt
                    : ProcessSignal::kTerminate;
      return true;
    }
  }
}

Status validate_create(IoContext &io, unsigned owner,
                       SignalInterest interest) noexcept {
  constexpr unsigned kSupported =
      mask(SignalInterest::kInterrupt) | mask(SignalInterest::kTerminate);
  if (!io.valid())
    return Status::FailedPrecondition("SignalSet requires a valid IoContext");
  if (owner >= io.num_workers())
    return Status::OutOfRange("SignalSet owner worker is outside IoContext");
  if (interest == SignalInterest::kNone || (mask(interest) & ~kSupported) != 0)
    return Status::InvalidArgument(
        "SignalSet requires a supported non-empty interest set");
  return Status{};
}

#if !HYLE_OS_WINDOWS

static_assert(std::atomic<unsigned>::is_always_lock_free,
              "signal pending bits must be lock-free");

struct PosixSignalState {
  std::mutex mutex;
  std::atomic<unsigned> pending{0};
  std::atomic<bool> waiter{false};
  int pipe_read = -1;
  int pipe_write = -1;
  int interrupt_references = 0;
  int terminate_references = 0;
  struct sigaction old_interrupt{};
  struct sigaction old_terminate{};
  bool have_old_interrupt = false;
  bool have_old_terminate = false;
  bool active_set = false;
};

PosixSignalState &global_signals() noexcept {
  static PosixSignalState state;
  return state;
}

unsigned signal_mask(int signal_number) noexcept {
  if (signal_number == SIGINT)
    return mask(SignalInterest::kInterrupt);
  if (signal_number == SIGTERM)
    return mask(SignalInterest::kTerminate);
  return 0;
}

extern "C" void hyle_signal_handler(int signal_number) noexcept {
  PosixSignalState &state = global_signals();
  const unsigned selected = signal_mask(signal_number);
  if (selected == 0 || state.pipe_write < 0)
    return;
  state.pending.fetch_or(selected, std::memory_order_release);
  const unsigned char wake = 1;
  ssize_t result;
  do {
    result = ::write(state.pipe_write, &wake, 1);
  } while (result < 0 && errno == EINTR);
  // EAGAIN is safe: pending retains every distinct signal kind and an existing
  // byte already wakes the sole waiter.
  (void)result;
}

Status ensure_pipe(PosixSignalState &state) noexcept {
  if (state.pipe_read >= 0)
    return Status{};
  int descriptors[2];
  if (::pipe(descriptors) != 0)
    return status_from_errno(errno, "create SignalSet self-pipe");
  for (int descriptor : descriptors) {
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    const int descriptor_flags = ::fcntl(descriptor, F_GETFD, 0);
    if (flags < 0 || descriptor_flags < 0 ||
        ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0 ||
        ::fcntl(descriptor, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0) {
      const int error = errno;
      ::close(descriptors[0]);
      ::close(descriptors[1]);
      return status_from_errno(error, "configure SignalSet self-pipe");
    }
  }
  state.pipe_read = descriptors[0];
  state.pipe_write = descriptors[1];
  return Status{};
}

Status configure_signal(PosixSignalState &state, int signal_number,
                        bool install) noexcept {
  int *references = signal_number == SIGINT ? &state.interrupt_references
                                            : &state.terminate_references;
  struct sigaction *old_action =
      signal_number == SIGINT ? &state.old_interrupt : &state.old_terminate;
  bool *have_old = signal_number == SIGINT ? &state.have_old_interrupt
                                           : &state.have_old_terminate;
  if (install) {
    if (*references == 0) {
      struct sigaction action{};
      action.sa_handler = &hyle_signal_handler;
      action.sa_flags = SA_RESTART;
      sigemptyset(&action.sa_mask);
      if (::sigaction(signal_number, &action, old_action) != 0)
        return status_from_errno(errno, "install process signal handler");
      *have_old = true;
    }
    ++*references;
    return Status{};
  }

  if (*references > 0 && --(*references) == 0) {
    if (*have_old && ::sigaction(signal_number, old_action, nullptr) != 0)
      return status_from_errno(errno, "restore process signal handler");
    *have_old = false;
  }
  return Status{};
}

#else

struct WindowsSignalState {
  std::mutex mutex;
  std::atomic<unsigned> interest{0};
  std::atomic<unsigned> pending{0};
  std::atomic<Proactor *> ring{nullptr};
  std::atomic<std::uint64_t> token{0};
  std::atomic<bool> waiter{false};
  bool installed = false;
  bool active_set = false;
};

WindowsSignalState &global_signals() noexcept {
  static WindowsSignalState state;
  return state;
}

BOOL WINAPI hyle_console_handler(DWORD control) noexcept {
  WindowsSignalState &state = global_signals();
  unsigned selected = 0;
  bool terminal = false;
  switch (control) {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
    selected = mask(SignalInterest::kInterrupt);
    break;
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    selected = mask(SignalInterest::kTerminate);
    terminal = true;
    break;
  default:
    return FALSE;
  }
  if ((state.interest.load(std::memory_order_acquire) & selected) == 0)
    return FALSE;

  state.pending.fetch_or(selected, std::memory_order_release);
  const std::uint64_t token =
      state.token.exchange(0, std::memory_order_acq_rel);
  if (token != 0) {
    Proactor *ring = state.ring.load(std::memory_order_acquire);
    if (ring != nullptr)
      (void)ring->post(token, 1);
  }
  if (terminal)
    ::Sleep(INFINITE);
  return TRUE;
}

#endif

} // namespace

Result<SignalSet> SignalSet::create(IoContext &io, unsigned owner_worker,
                                    SignalInterest interest) noexcept {
  SignalSet result;
  Status status = result.initialize(io, owner_worker, interest);
  if (!status.ok())
    return make_unexpected(std::move(status));
  return result;
}

Status SignalSet::initialize(IoContext &io, unsigned owner_worker,
                             SignalInterest interest) noexcept {
  Status validation = validate_create(io, owner_worker, interest);
  if (!validation.ok())
    return validation;

#if HYLE_OS_WINDOWS
  WindowsSignalState &state = global_signals();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.active_set)
    return Status::AlreadyExists(
        "one SignalSet already owns the process handlers");
  if (!state.installed && !::SetConsoleCtrlHandler(&hyle_console_handler, TRUE))
    return status_from_win32(::GetLastError(),
                             "install console control handler");
  state.installed = true;
  state.pending.store(0, std::memory_order_release);
  state.interest.store(mask(interest), std::memory_order_release);
  state.active_set = true;
#else
  PosixSignalState &state = global_signals();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.active_set)
    return Status::AlreadyExists(
        "one SignalSet already owns the process handlers");
  Status status = ensure_pipe(state);
  if (!status.ok())
    return status;
  if (contains(interest, SignalInterest::kInterrupt)) {
    status = configure_signal(state, SIGINT, true);
    if (!status.ok())
      return status;
  }
  if (contains(interest, SignalInterest::kTerminate)) {
    status = configure_signal(state, SIGTERM, true);
    if (!status.ok()) {
      if (contains(interest, SignalInterest::kInterrupt))
        (void)configure_signal(state, SIGINT, false);
      return status;
    }
  }
  state.pending.fetch_and(~mask(interest), std::memory_order_acq_rel);
  state.active_set = true;
#endif

  io_ = &io;
  owner_ = owner_worker;
  interest_ = interest;
  valid_ = true;
  return Status{};
}

SignalSet::~SignalSet() { reset(); }

SignalSet::SignalSet(SignalSet &&other) noexcept
    : io_(std::exchange(other.io_, nullptr)), owner_(other.owner_),
      interest_(std::exchange(other.interest_, SignalInterest::kNone)),
      valid_(std::exchange(other.valid_, false)) {
  HYLE_FATAL_ASSERT(!other.waiting_);
}

SignalSet &SignalSet::operator=(SignalSet &&other) noexcept {
  HYLE_FATAL_ASSERT(!waiting_ && !other.waiting_);
  if (this != &other) {
    reset();
    io_ = std::exchange(other.io_, nullptr);
    owner_ = other.owner_;
    interest_ = std::exchange(other.interest_, SignalInterest::kNone);
    valid_ = std::exchange(other.valid_, false);
  }
  return *this;
}

void SignalSet::reset() noexcept {
  HYLE_FATAL_ASSERT(!waiting_);
  if (!valid_)
    return;

#if HYLE_OS_WINDOWS
  WindowsSignalState &state = global_signals();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.interest.store(0, std::memory_order_release);
  state.pending.store(0, std::memory_order_release);
  if (state.installed) {
    (void)::SetConsoleCtrlHandler(&hyle_console_handler, FALSE);
    state.installed = false;
  }
  state.active_set = false;
#else
  PosixSignalState &state = global_signals();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (contains(interest_, SignalInterest::kInterrupt))
    (void)configure_signal(state, SIGINT, false);
  if (contains(interest_, SignalInterest::kTerminate))
    (void)configure_signal(state, SIGTERM, false);
  state.pending.fetch_and(~mask(interest_), std::memory_order_acq_rel);
  state.active_set = false;
#endif

  io_ = nullptr;
  interest_ = SignalInterest::kNone;
  valid_ = false;
}

#if HYLE_OS_WINDOWS

bool SignalSet::WaitAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (self == nullptr || !self->valid_) {
    status = Status::FailedPrecondition("SignalSet is not active");
    return false;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (self->waiting_) {
    status =
        Status::FailedPrecondition("SignalSet already has an outstanding wait");
    return false;
  }

  WindowsSignalState &state = global_signals();
  if (take_pending(state.pending, self->interest_, &signal)) {
    has_signal = true;
    return false;
  }
  bool expected = false;
  if (!state.waiter.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
    status = Status::FailedPrecondition(
        "process signal channel already has a waiter");
    return false;
  }
  self->waiting_ = true;
  operation.waker = sched::current_waker();
  state.ring.store(&self->io_->proactor_for_current(),
                   std::memory_order_release);
  state.token.store(to_token(&operation), std::memory_order_release);

  if (take_pending(state.pending, self->interest_, &signal)) {
    has_signal = true;
    // If the handler already consumed the token, its IOCP post must be reaped
    // before this stack slot can die. Otherwise disarm without suspending.
    if (state.token.exchange(0, std::memory_order_acq_rel) == 0)
      sched::suspend();
  } else {
    sched::suspend();
    has_signal = take_pending(state.pending, self->interest_, &signal);
    if (!has_signal)
      status = Status::Internal(
          "console completion arrived without a pending signal");
  }

  state.token.store(0, std::memory_order_release);
  state.ring.store(nullptr, std::memory_order_release);
  state.waiter.store(false, std::memory_order_release);
  self->waiting_ = false;
  return false;
}

#else

bool SignalSet::WaitAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (self == nullptr || !self->valid_) {
    status = Status::FailedPrecondition("SignalSet is not active");
    return false;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (self->waiting_) {
    status =
        Status::FailedPrecondition("SignalSet already has an outstanding wait");
    return false;
  }

  PosixSignalState &state = global_signals();
  bool expected = false;
  if (!state.waiter.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
    status = Status::FailedPrecondition(
        "process signal channel already has a waiter");
    return false;
  }
  self->waiting_ = true;

  for (;;) {
    if (take_pending(state.pending, self->interest_, &signal)) {
      has_signal = true;
      break;
    }
    operation = IoOperation{};
    operation.waker = sched::current_waker();
    Proactor &proactor = self->io_->proactor_for_current();
    if (!proactor.submit_read(state.pipe_read, &wake_byte, 1, 0,
                              to_token(&operation))) {
      status = Status::ResourceExhausted(
          "process signal wait could not enter the bounded IO queue");
      break;
    }
    sched::suspend();
    if (operation.result < 0) {
      status = detail::status_from_native_result(
          operation.result, detail::kFileNativeDomain, "process signal wait");
      break;
    }
    if (operation.result == 0) {
      status = Status::DataLoss("process signal pipe reached EOF");
      break;
    }
    // A byte can be stale when a previously queued signal was consumed from
    // pending before its wake byte. Loop and re-arm until a current bit exists.
  }

  state.waiter.store(false, std::memory_order_release);
  self->waiting_ = false;
  return false;
}

#endif

Result<ProcessSignal> SignalSet::WaitAwaitable::await_resume() noexcept {
  if (!status.ok())
    return make_unexpected(std::move(status));
  if (!has_signal)
    return make_unexpected(
        Status::Internal("signal wait completed without a signal"));
  return signal;
}

} // namespace hyle::io
