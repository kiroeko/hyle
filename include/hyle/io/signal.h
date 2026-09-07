#pragma once

#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/io/detail/net.h"
#include "hyle/io/io_context.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#define HYLE_IO_API HYLE_EXPORT
#else
#define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

enum class ProcessSignal : std::uint8_t {
  kInterrupt,
  kTerminate,
};

enum class SignalInterest : std::uint8_t {
  kNone = 0,
  kInterrupt = 1u << 0,
  kTerminate = 1u << 1,
};

constexpr SignalInterest operator|(SignalInterest left,
                                   SignalInterest right) noexcept {
  return static_cast<SignalInterest>(static_cast<unsigned>(left) |
                                     static_cast<unsigned>(right));
}

constexpr bool contains(SignalInterest set, SignalInterest value) noexcept {
  return (static_cast<unsigned>(set) & static_cast<unsigned>(value)) != 0;
}

// Process-wide signal-to-scheduler bridge. Exactly one SignalSet owns the
// process handlers at a time; fan-out belongs after wait() in Sched. Signals
// are coalesced by kind, so a burst consumes bounded storage while distinct
// interrupt/terminate requests cannot overwrite one another.
class SignalSet {
public:
  HYLE_IO_API static Result<SignalSet> create(IoContext &io,
                                              unsigned owner_worker,
                                              SignalInterest interest) noexcept;

  HYLE_IO_API ~SignalSet();
  HYLE_IO_API SignalSet(SignalSet &&other) noexcept;
  HYLE_IO_API SignalSet &operator=(SignalSet &&other) noexcept;
  SignalSet(const SignalSet &) = delete;
  SignalSet &operator=(const SignalSet &) = delete;

  bool valid() const noexcept { return valid_; }
  unsigned owner_worker() const noexcept { return owner_; }
  SignalInterest interest() const noexcept { return interest_; }

  struct WaitAwaitable {
    SignalSet *self = nullptr;
    detail::IoOperation operation{};
    unsigned char wake_byte = 0;
    Status status{};
    ProcessSignal signal = ProcessSignal::kInterrupt;
    bool has_signal = false;

    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Result<ProcessSignal> await_resume() noexcept;
  };

  // The SignalSet must outlive its outstanding wait. A second concurrent wait
  // returns FailedPrecondition. Repeated sequential waits are supported.
  WaitAwaitable wait() noexcept { return WaitAwaitable{this}; }

private:
  SignalSet() noexcept = default;
  HYLE_IO_API Status initialize(IoContext &io, unsigned owner_worker,
                                SignalInterest interest) noexcept;
  HYLE_IO_API void reset() noexcept;

  IoContext *io_ = nullptr;
  unsigned owner_ = 0;
  SignalInterest interest_ = SignalInterest::kNone;
  bool valid_ = false;
  bool waiting_ = false;
};

} // namespace hyle::io
