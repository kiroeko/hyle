#include "hyle/io/signal.h"

#include <atomic>
#include <mutex>
#include <utility>

#include "hyle/macros.h"
#include "hyle/platform.h"
#include "hyle/sched/coro.h"   // sched::suspend
#include "hyle/sched/waker.h"  // sched::current_waker

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <csignal>
#  include <fcntl.h>
#  include <unistd.h>
#endif

// io-09 signal_set. See signal.h for the design; both platforms turn a process-
// global signal/console handler into an ordinary proactor completion the awaiting
// fiber wakes on. Process-global state lives here (one handler per signal for the
// whole process). The self-pipe (POSIX) is created ONCE and kept for the process
// lifetime -- tokio's pattern, which sidesteps the fd-reuse race a close-on-
// deregister would open (a handler mid-flight on another thread could write a
// stale fd). Deregistration only restores the default disposition; the pipe stays.

namespace hyle::io {

#if !defined(HYLE_OS_WINDOWS)

namespace {

// Process-global self-pipe + per-signal install refcounts. The handler writes the
// signo to `pipe_wr` (async-signal-safe: a bare write on a nonblocking fd). It is
// created before any sigaction installs the handler, so the handler is guaranteed
// to see it (sigaction is a barrier); hence plain ints, not atomics.
struct SigGlobal {
  std::mutex mu;
  int pipe_rd = -1;
  int pipe_wr = -1;
  int refcount_int = 0;   // SIGINT installs
  int refcount_term = 0;  // SIGTERM installs
  bool pipe_made = false;
};
SigGlobal& sig_global() noexcept {
  static SigGlobal g;
  return g;
}

// The one signal handler for every signal HYLE catches. Writes the signo as a
// single byte to the self-pipe; the awaiting fiber's proactor read delivers it.
// Async-signal-safe: only write() on a nonblocking fd (EAGAIN when the pipe is
// full is fine -- a pending byte already means "signal arrived"; we lose nothing).
extern "C" void hyle_signal_handler(int signo) noexcept {
  SigGlobal& g = sig_global();
  if (g.pipe_wr < 0) return;
  unsigned char b = static_cast<unsigned char>(signo);
  ssize_t n;
  do {
    n = ::write(g.pipe_wr, &b, 1);
  } while (n < 0 && errno == EINTR);
  (void)n;
}

// Lazily create the process-global self-pipe. Under g.mu. Returns false on failure.
bool ensure_pipe(SigGlobal& g) noexcept {
  if (g.pipe_made) return true;
  int fds[2];
  if (::pipe2(fds, O_CLOEXEC) != 0) return false;
  // Write end nonblocking so the handler never blocks (drops the byte on a full
  // pipe -- harmless, a queued byte already signals "arrived").
  int fl = ::fcntl(fds[1], F_GETFL, 0);
  ::fcntl(fds[1], F_SETFL, fl | O_NONBLOCK);
  g.pipe_rd = fds[0];
  g.pipe_wr = fds[1];
  g.pipe_made = true;
  return true;
}

int bit_to_signo(unsigned bit) noexcept {
  if (bit == kSigInt) return SIGINT;
  if (bit == kSigTerm) return SIGTERM;
  return 0;
}
unsigned signo_to_bit(int signo) noexcept {
  if (signo == SIGINT) return kSigInt;
  if (signo == SIGTERM) return kSigTerm;
  return 0;
}

// Install/uninstall our handler for one signal, refcounted so the last SignalSet
// to drop it restores SIG_DFL. `install`=true adds a ref; false drops one.
void set_signal(SigGlobal& g, int signo, bool install) noexcept {
  int* rc = (signo == SIGINT) ? &g.refcount_int : &g.refcount_term;
  if (install) {
    if ((*rc)++ == 0) {
      struct sigaction sa;
      std::memset(&sa, 0, sizeof(sa));
      sa.sa_handler = &hyle_signal_handler;
      sa.sa_flags = SA_RESTART;  // don't abort restartable syscalls elsewhere
      ::sigemptyset(&sa.sa_mask);
      ::sigaction(signo, &sa, nullptr);
    }
  } else {
    if (*rc > 0 && --(*rc) == 0) {
      struct sigaction sa;
      std::memset(&sa, 0, sizeof(sa));
      sa.sa_handler = SIG_DFL;
      ::sigemptyset(&sa.sa_mask);
      ::sigaction(signo, &sa, nullptr);
    }
  }
}

}  // namespace

SignalSet::SignalSet(IoContext& io, unsigned owner_worker,
                     unsigned signals) noexcept
    : io_(&io), owner_(owner_worker), signals_(signals) {
  SigGlobal& g = sig_global();
  std::lock_guard<std::mutex> lk(g.mu);
  if (!ensure_pipe(g)) return;
  pipe_rd_ = static_cast<std::intptr_t>(g.pipe_rd);
  pipe_wr_ = static_cast<std::intptr_t>(g.pipe_wr);
  if (signals_ & kSigInt) set_signal(g, SIGINT, /*install=*/true);
  if (signals_ & kSigTerm) set_signal(g, SIGTERM, /*install=*/true);
  valid_ = true;
}

SignalSet::~SignalSet() {
  if (!valid_) return;
  SigGlobal& g = sig_global();
  std::lock_guard<std::mutex> lk(g.mu);
  if (signals_ & kSigInt) set_signal(g, SIGINT, /*install=*/false);
  if (signals_ & kSigTerm) set_signal(g, SIGTERM, /*install=*/false);
  // The pipe stays open for the process lifetime (tokio pattern); nothing to close.
}

bool SignalSet::WaitAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (!self->valid_) {
    op_.result = -1;
    return false;  // invalid set: resume at once with an error
  }
  op_.waker = sched::current_waker();
  // Arm a 1-byte read on the global self-pipe via the owner worker's ring. The
  // handler's write completes it; the byte is the signo (read into byte_).
  Proactor& p = self->io_->proactor(self->owner_);
  submitted_ = p.submit_read(self->pipe_rd_, &byte_, 1, 0, to_token(&op_));
  if (!submitted_) {
    op_.result = -1;
    return false;
  }
  sched::suspend();
  return false;
}

int SignalSet::WaitAwaitable::await_resume() noexcept {
  if (op_.result < 1) return -1;  // error / EOF -> no signal
  unsigned bit = signo_to_bit(static_cast<int>(byte_));
  return bit != 0 ? static_cast<int>(bit) : -1;
}

#else  // HYLE_OS_WINDOWS

namespace {

// Process-global console-handler target. The console handler runs on a SEPARATE
// thread (no async-signal-safe limit), so it does the full post itself. `token` is
// exchanged to 0 when consumed so exactly ONE post happens per armed wait (the op_
// lives on the fiber stack -- a second post after resume would be a UAF). `ring`
// is the owner worker's proactor (stable for the SignalSet's life).
struct WinSigGlobal {
  std::mutex mu;
  std::atomic<Proactor*> ring{nullptr};
  std::atomic<std::uint64_t> token{0};
  std::atomic<unsigned> want{0};  // which SignalBits are being awaited
  int refcount = 0;
  bool installed = false;
};
WinSigGlobal& win_sig_global() noexcept {
  static WinSigGlobal g;
  return g;
}

BOOL WINAPI hyle_console_handler(DWORD ctrl_type) noexcept {
  WinSigGlobal& g = win_sig_global();
  unsigned bit = 0;
  bool terminal = false;  // close/logoff/shutdown: must NOT return or the OS kills us
  switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
      bit = kSigInt;
      break;
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      bit = kSigTerm;
      terminal = true;
      break;
    default:
      return FALSE;  // not ours -> let the next handler (default) run
  }
  if ((g.want.load(std::memory_order_acquire) & bit) == 0) return FALSE;
  // Deliver EXACTLY ONCE: exchange the token to 0 so a second event doesn't post to
  // an already-resumed (freed) op_. ring is stable while a wait is armed.
  std::uint64_t tok = g.token.exchange(0, std::memory_order_acq_rel);
  if (tok != 0) {
    Proactor* p = g.ring.load(std::memory_order_acquire);
    if (p != nullptr) p->post(tok, static_cast<std::int64_t>(bit));
  }
  if (terminal) {
    // Returning from a close/logoff/shutdown handler lets the OS terminate the
    // process immediately -- before the app can drain. Block here so the graceful
    // shutdown the posted signal triggers has the OS-granted window to run (tokio
    // parks its handler thread for the same reason). The OS still hard-kills after
    // its timeout, so this cannot wedge the machine.
    ::Sleep(INFINITE);
  }
  return TRUE;  // handled -> don't run the default (terminate) handler
}

unsigned all_supported() noexcept { return kSigInt | kSigTerm; }

}  // namespace

SignalSet::SignalSet(IoContext& io, unsigned owner_worker,
                     unsigned signals) noexcept
    : io_(&io), owner_(owner_worker), signals_(signals) {
  WinSigGlobal& g = win_sig_global();
  std::lock_guard<std::mutex> lk(g.mu);
  if (g.refcount++ == 0) {
    if (::SetConsoleCtrlHandler(&hyle_console_handler, TRUE)) g.installed = true;
  }
  (void)all_supported();
  valid_ = true;
}

SignalSet::~SignalSet() {
  if (!valid_) return;
  WinSigGlobal& g = win_sig_global();
  std::lock_guard<std::mutex> lk(g.mu);
  if (g.refcount > 0 && --g.refcount == 0 && g.installed) {
    ::SetConsoleCtrlHandler(&hyle_console_handler, FALSE);
    g.installed = false;
  }
}

bool SignalSet::WaitAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (!self->valid_) {
    op_.result = -1;
    return false;
  }
  op_.waker = sched::current_waker();
  WinSigGlobal& g = win_sig_global();
  // Publish the target BEFORE arming so a signal racing in is delivered: set ring +
  // wanted bits, then the token last (release) -- the handler reads token last.
  g.ring.store(&self->io_->proactor(self->owner_), std::memory_order_release);
  g.want.store(self->signals_, std::memory_order_release);
  g.token.store(to_token(&op_), std::memory_order_release);
  submitted_ = true;
  sched::suspend();  // resumed by the console handler's post() -> net::complete
  return false;
}

int SignalSet::WaitAwaitable::await_resume() noexcept {
  // Windows carries the SignalBit directly in the posted result (see the handler).
  win_sig_global().token.store(0, std::memory_order_release);  // disarm if woken otherwise
  if (op_.result <= 0) return -1;
  return static_cast<int>(op_.result);
}

#endif  // HYLE_OS_WINDOWS

// ---- move/dtor plumbing shared by both platforms ---------------------------

SignalSet::SignalSet(SignalSet&& o) noexcept
    : io_(o.io_),
      owner_(o.owner_),
      signals_(std::exchange(o.signals_, 0)),
      pipe_rd_(o.pipe_rd_),
      pipe_wr_(o.pipe_wr_),
      valid_(std::exchange(o.valid_, false)) {}

SignalSet& SignalSet::operator=(SignalSet&& o) noexcept {
  if (this != &o) {
    // Release our own registration first (dtor logic), then steal o's.
    this->~SignalSet();
    io_ = o.io_;
    owner_ = o.owner_;
    signals_ = std::exchange(o.signals_, 0);
    pipe_rd_ = o.pipe_rd_;
    pipe_wr_ = o.pipe_wr_;
    valid_ = std::exchange(o.valid_, false);
  }
  return *this;
}

}  // namespace hyle::io
