#pragma once

#include <cstdint>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"  // IoOperation, current_waker/suspend bridge

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-09 signal_set (capability C10). A co_await-able wait for a process signal --
// the trigger for C4 graceful shutdown (SIGTERM/SIGINT / Ctrl-C). Reference-lib
// consensus (tokio signal, libevent signal.c, read in full): do NOT do work in the
// signal handler; instead the handler does one async-signal-safe thing and the
// event loop turns it into a normal completion.
//
//  * POSIX: a process-global sigaction handler writes ONE byte (the signal number)
//    to a self-pipe (async-signal-safe; thread-agnostic -- the kernel may run the
//    handler on any thread, incl. a worker mid-fiber, and a bare write() is safe
//    there). The SignalSet arms an ordinary proactor READ on the pipe's read end;
//    when the byte arrives the read completes and the awaiting fiber wakes. This is
//    tokio's self-pipe, NOT signalfd -- signalfd would need the signal blocked
//    consistently in every thread, which fights the already-running scheduler
//    workers; the self-pipe needs no mask surgery.
//  * Windows: no POSIX signals. SetConsoleCtrlHandler catches Ctrl-C / Ctrl-Break /
//    close, running its handler on a SEPARATE thread (so no async-signal-safe
//    limits). That thread posts a completion to the waiting SignalSet's ring via
//    the thread-safe PostQueuedCompletionStatus (proactor.post) -- the same wake
//    the loop already understands.
//
// Signal handling is PROCESS-GLOBAL: installing a handler replaces the default
// disposition (SIGINT no longer terminates), and there is one handler per signal
// for the whole process. Registration goes through a process-global table (defined
// in signal.cc); construct a SignalSet on the worker that should service the wait.

namespace hyle::io {

// The signals HYLE exposes. Kept platform-neutral (mapped to SIGINT/SIGTERM on
// POSIX, to console control events on Windows) so callers need no <csignal>.
enum SignalBits : unsigned {
  kSigInt = 1u << 0,   // SIGINT  / Ctrl-C            (interactive interrupt)
  kSigTerm = 1u << 1,  // SIGTERM / Ctrl-Close/logoff (polite terminate request)
};

// A set of signals to await on one worker. Construct with the signals to catch;
// this installs the process-global handlers (idempotent/ref-counted per signal)
// and, on POSIX, creates the self-pipe read by co_await wait(). Move-only; the
// destructor deregisters + closes the pipe. Awaits MUST run on the fiber pinned to
// `owner_worker` (its ring reaps the pipe read / receives the posted completion).
//
// ONE-PROCESS-WAITER MODEL: at most one valid SignalSet, with one outstanding
// co_await wait(), exists per process. A second construction stays invalid and a
// concurrent second wait resolves with an error. The POSIX self-pipe and Windows
// console token are process-global single-consumer channels; pretending separate
// sets were independent lets one consume another's signal. This matches the
// canonical graceful-shutdown fiber. Fan out after one wait via a sched channel.
class SignalSet {
 public:
  SignalSet() noexcept = default;
  // `signals` is a bitmask of SignalBits. Registers handlers + (POSIX) opens the
  // self-pipe on `owner_worker`'s ring. valid() reports success; an invalid set's
  // wait() resolves immediately with a negative result.
  HYLE_IO_API SignalSet(IoContext& io, unsigned owner_worker,
                        unsigned signals) noexcept;
  HYLE_IO_API ~SignalSet();
  HYLE_IO_API SignalSet(SignalSet&& o) noexcept;
  HYLE_IO_API SignalSet& operator=(SignalSet&& o) noexcept;
  HYLE_DISABLE_COPY(SignalSet);

  bool valid() const noexcept { return valid_; }
  unsigned owner_worker() const noexcept { return owner_; }

  // The awaitable returned by wait(): arms the platform mechanism and yields the
  // fiber until a signal in the set fires. await_resume returns which SignalBit
  // fired (kSigInt / kSigTerm) or a negative value on error (invalid set / submit
  // failure). One-shot: co_await wait() again to catch the next signal.
  struct WaitAwaitable {
    SignalSet* self;
    IoOperation op_{};
    unsigned char byte_ = 0;  // POSIX: the signo the pipe read lands here
    bool submitted_ = false;

    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API int await_resume() noexcept;
  };

  // co_await sigs.wait() -> the SignalBit that fired, or negative on error.
  WaitAwaitable wait() noexcept { return WaitAwaitable{this}; }

 private:
  void reset() noexcept;

  IoContext* io_ = nullptr;
  unsigned owner_ = 0;
  unsigned signals_ = 0;
  // POSIX self-pipe (read end armed on the ring, write end written by the handler).
  // Unused on Windows (stays -1); the console handler posts to the ring instead.
  std::intptr_t pipe_rd_ = -1;
  std::intptr_t pipe_wr_ = -1;
  bool valid_ = false;
};

}  // namespace hyle::io
