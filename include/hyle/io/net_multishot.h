#pragma once

#include <cstdint>
#include <new>

#include "hyle/io/net.h"
#include "hyle/io/net_timeout.h"  // reuse kCancelToken (cancel op's own-CQE drop)
#include "hyle/io/proactor.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/waker.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-07b multishot recv (kBuffer/kMore half): the co_await surface for io_uring's
// zero-copy multishot receive. ONE submit_recv_multishot keeps delivering a
// completion per arriving segment (kMore SET while armed); the kernel picks a buffer
// from a registered ring per arrival (kBuffer + buffer_id), so there is NO caller
// buffer and NO per-recv submit -- the two costs an ordinary RecvOp pays every call.
// This is the read path proxies/echo servers want (liburing proxy.c recv_multishot).
//
// TWO HARD PARTS (same family as net_timeout.h's timed op, verified against tokio-
// uring Lifecycle + liburing proxy.c, 2026-07-05):
//
//  1. ONE ARM, MANY COMPLETIONS, ONE FIBER. co_await resumes a fiber exactly once,
//     but multishot yields a stream. Model it as a MultishotRecvStream you drive with
//     `co_await stream.next()`: the FIRST next() arms the recv; each completion wakes
//     the fiber, which reads the buffer then co_awaits next() AGAIN -- WITHOUT
//     re-submitting (the multishot stays armed kernel-side). A completion with kMore
//     CLEAR is terminal (peer closed / error / ring exhausted) -> next() returns an
//     empty final chunk.
//
//  2. LATE-COMPLETION LIFETIME (the cancel-safety trap). A multishot has completions
//     in flight until its TERMINAL (kMore-clear) CQE. If the fiber abandons the stream
//     early (drops it, or a higher layer cancels), a proactor cancel is async: in-
//     flight + a final -ECANCELED CQE still land. A stack slot would be a UAF. So the
//     stream state is HEAP-allocated with a refcount {kernel, fiber}, EXACTLY like
//     TimedSlot -- but the KERNEL ref is not spent per completion; it is spent only on
//     the kMore-CLEAR terminal completion (that is when the kernel is truly done with
//     the token). The fiber ref is dropped when the stream is destroyed. Ordinary one-
//     shot RecvOp keeps its zero-alloc stack slot; only multishot pays the heap slot.
//
// TOKEN TAG: net_timeout uses bit0=1 for a timed slot. Multishot uses bits 0b11
// (bit0=1 so io_context's is_timed_token catches it, bit1=1 so complete_timed
// redirects here). kCancelToken/kWakeToken keep bit1 clear, so they never collide.

namespace hyle::io {

// A kernel-selected received buffer, handed to the consumer for one chunk. Non-owning
// VIEW into the provided-buffer ring; its DESTRUCTOR returns the buffer to the ring
// (RAII: the consumer can't forget buf_ring_return, which would starve the ring). A
// terminal/empty chunk (more==false, data==nullptr) owns nothing -> dtor is a no-op.
class RecvBuf {
 public:
  RecvBuf() noexcept = default;
  RecvBuf(Proactor* p, int bgid, const void* data, std::uint32_t len,
          std::uint32_t buffer_id, bool has_buf, bool more) noexcept
      : p_(p), bgid_(bgid), data_(data), len_(len), buffer_id_(buffer_id),
        has_buf_(has_buf), more_(more) {}
  ~RecvBuf() {
    if (has_buf_ && p_ != nullptr) p_->buf_ring_return(bgid_, buffer_id_);
  }
  RecvBuf(RecvBuf&& o) noexcept
      : p_(o.p_), bgid_(o.bgid_), data_(o.data_), len_(o.len_),
        buffer_id_(o.buffer_id_), has_buf_(o.has_buf_), more_(o.more_) {
    o.has_buf_ = false;  // moved-from returns nothing
    o.p_ = nullptr;
  }
  RecvBuf& operator=(RecvBuf&& o) noexcept {
    if (this != &o) {
      if (has_buf_ && p_ != nullptr) p_->buf_ring_return(bgid_, buffer_id_);
      p_ = o.p_; bgid_ = o.bgid_; data_ = o.data_; len_ = o.len_;
      buffer_id_ = o.buffer_id_; has_buf_ = o.has_buf_; more_ = o.more_;
      o.has_buf_ = false; o.p_ = nullptr;
    }
    return *this;
  }
  HYLE_DISABLE_COPY(RecvBuf);

  const void* data() const noexcept { return data_; }
  std::uint32_t size() const noexcept { return len_; }
  // The recv result if negative (errno) -- a terminal error chunk carries it in size
  // via valid(): valid() false + err() gives the reason. >=0 result => a data chunk.
  std::int64_t err() const noexcept { return err_; }
  void set_err(std::int64_t e) noexcept { err_ = e; }
  // false => this is the terminal chunk: the multishot ended (peer closed, error, or
  // ring exhausted). The consumer stops calling next() after seeing more()==false.
  bool more() const noexcept { return more_; }
  bool empty() const noexcept { return len_ == 0; }

 private:
  Proactor* p_ = nullptr;
  int bgid_ = -1;
  const void* data_ = nullptr;
  std::uint32_t len_ = 0;
  std::uint32_t buffer_id_ = 0;
  bool has_buf_ = false;   // owns a ring buffer to return on dtor
  bool more_ = false;
  std::int64_t err_ = 0;
};

// Heap state for one multishot recv stream. Refcount {kernel, fiber} = 2 (same shape
// as TimedSlot). Single-threaded per owner worker => plain (non-atomic) fields.
//
// A multishot delivers MANY completions on ONE token, and several can land in a
// SINGLE reap batch (kernel had multiple segments queued) BEFORE the fiber resumes to
// consume the first. A single result field would let completion N+1 clobber N (the
// bug that made a 6-byte data chunk vanish behind the following EOF). So the slot
// holds a small QUEUE of pending completions; complete_multishot pushes, next() pops.
// Capacity: the kernel can't have more buffers outstanding than the ring holds, plus
// one terminal -- kMaxPending covers the common ring sizes; if it ever overflowed we'd
// drop-newest (a correctness bug), so it's sized generously and asserted.
struct MultishotSlot {
  IoOperation io{};          // landing pad: waker (result/flags used transiently)
  mem::Allocator alloc;      // frees the slot at refcount 0
  Proactor* p = nullptr;
  int bgid = -1;
  int refcount = 2;          // {kernel op, fiber}
  bool armed = false;        // submit_recv_multishot issued
  bool kernel_done = false;  // terminal (kMore-clear) completion arrived -> kernel ref spent
  bool waiting = false;      // a next() call is currently suspended on io.waker
  bool abandoned = false;    // stream dtor ran: never wake its stale fiber again
  bool overflowed = false;   // pending queue filled; cancellation has been requested

  // Pending-completion ring (one producer = the reaper, one consumer = the fiber, same
  // thread -> no atomics). Each entry captures a delivered completion verbatim.
  static constexpr unsigned kMaxPending = 64;
  struct Pending {
    std::int64_t result;
    std::uint32_t flags;
    std::uint32_t buffer_id;
  };
  Pending q[kMaxPending];
  unsigned head = 0;  // pop cursor
  unsigned tail = 0;  // push cursor (tail - head = count, both monotonic mod kMaxPending)

  bool q_empty() const noexcept { return head == tail; }
  bool q_full() const noexcept { return tail - head >= kMaxPending; }
  bool q_push(std::int64_t r, std::uint32_t f, std::uint32_t bid) noexcept {
    if (q_full()) return false;
    q[tail % kMaxPending] = Pending{r, f, bid};
    ++tail;
    return true;
  }
  Pending q_pop() noexcept {
    Pending pd = q[head % kMaxPending];
    ++head;
    return pd;
  }

  // Return every kernel-selected buffer that the consumer never reached. This is
  // needed when a stream is dropped with completions already queued: those CQEs
  // transferred buffer ownership to userspace, so cancelling the arm does not put
  // them back into the provided-buffer ring.
  void return_pending_buffers() noexcept {
    while (!q_empty()) {
      Pending pd = q_pop();
      if ((pd.flags & Completion::kBuffer) != 0 && pd.result > 0 &&
          p != nullptr) {
        p->buf_ring_return(bgid, pd.buffer_id);
      }
    }
  }

  void release() noexcept {
    if (--refcount == 0) {
      mem::Allocator a = alloc;
      this->~MultishotSlot();
      a.deallocate(this, sizeof(MultishotSlot), alignof(MultishotSlot));
    }
  }
};

// Token tagging: bits 0b11 == multishot slot (bit0 set so io_context routes it to the
// timed/tagged path; bit1 set so complete_timed redirects to complete_multishot).
inline std::uint64_t to_multishot_token(MultishotSlot* s) noexcept {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(s)) | 3ull;
}
inline bool is_multishot_token(std::uint64_t ud) noexcept {
  return (ud & 3ull) == 3ull;
}
inline MultishotSlot* from_multishot_token(std::uint64_t ud) noexcept {
  return reinterpret_cast<MultishotSlot*>(static_cast<std::uintptr_t>(ud & ~3ull));
}

// Router entry for a MULTISHOT completion (io_context::poll -> complete_timed sees
// bit1 set and forwards here). Owner-thread only. A multishot delivers MANY
// completions on one token; QUEUE each (several can arrive in one reap batch), spend
// the kernel ref on the terminal (kMore clear) one, and wake the fiber so it drains
// the queue via next().
inline void complete_multishot(const Completion& c) noexcept {
  MultishotSlot* s = from_multishot_token(c.user_data);
  const bool terminal = (c.flags & Completion::kMore) == 0;

  // Once the stream object is gone, its saved Waker refers to a fiber that has
  // already continued (and may have been destroyed). Do not queue or wake it.
  // Return any delivered provided buffer immediately and retain the kernel ref
  // until the terminal CQE proves the multishot is really over.
  if (s->abandoned) {
    if ((c.flags & Completion::kBuffer) != 0 && c.result > 0 && s->p != nullptr)
      s->p->buf_ring_return(s->bgid, c.buffer_id);
    if (terminal) {
      s->kernel_done = true;
      s->release();  // terminal CQE spends the kernel ref
    }
    return;
  }

  bool was_empty = s->q_empty();
  if (!s->q_push(c.result, c.flags, c.buffer_id)) {
    // Never leak a provided buffer on overload. A queue overflow should be
    // unreachable with the runtime's 64-CQE reap batch, but it is a recoverable
    // resource-pressure condition, not permission to silently starve the ring.
    if ((c.flags & Completion::kBuffer) != 0 && c.result > 0 && s->p != nullptr)
      s->p->buf_ring_return(s->bgid, c.buffer_id);
    if (!s->overflowed && !terminal) {
      s->overflowed = true;
      s->p->submit_cancel(to_multishot_token(s), kCancelToken);
    }
  }
  // kMore CLEAR => the multishot is finished; this is the last completion for the
  // token, so the kernel ref is now spent. Mark it so resume can drop that ref.
  if (terminal) s->kernel_done = true;
  // Wake ONLY on the empty->non-empty transition: the fiber, once woken, drains the
  // WHOLE queue (await_ready stays true while non-empty, so it never re-suspends until
  // empty). If several completions land in one reap batch, only the first wakes; the
  // rest just enqueue. This gives EXACTLY ONE wake_rendezvous per suspend episode --
  // the same single-wake discipline net_timeout.h needs (a second wake on an already-
  // runnable fiber would leak wake_state_ into its next suspend).
  if (was_empty && s->waiting) {
    s->waiting = false;
    s->io.waker.wake();
  }
  if (terminal) s->release();  // terminal CQE spends the kernel ref
}

// co_await stream.next() -> RecvBuf. Drive a multishot recv: first next() arms it,
// each subsequent one waits for the next kernel-delivered segment. A RecvBuf with
// more()==false is terminal.
struct MultishotRecvStream {
  Proactor* p = nullptr;
  std::intptr_t fd = -1;
  int bgid = -1;
  mem::Allocator alloc{};
  MultishotSlot* slot_ = nullptr;
  bool ended_ = false;  // a terminal chunk was already returned

  MultishotRecvStream() noexcept = default;
  MultishotRecvStream(Proactor* p, std::intptr_t fd, int bgid,
                      mem::Allocator alloc) noexcept
      : p(p), fd(fd), bgid(bgid), alloc(alloc) {}
  HYLE_DISABLE_COPY(MultishotRecvStream);
  MultishotRecvStream(MultishotRecvStream&& o) noexcept
      : p(o.p), fd(o.fd), bgid(o.bgid), alloc(o.alloc),
        slot_(o.slot_), ended_(o.ended_) {
    o.slot_ = nullptr;
  }
  // Dtor: if the stream is dropped while the multishot may still be armed (fiber
  // abandons it early), cancel the kernel op and drop the fiber ref. The kernel ref
  // stays until the terminal -ECANCELED completion lands (complete_multishot spends
  // it), so a late CQE writes a still-alive slot -- no UAF. If the stream already saw
  // its terminal chunk (kernel_done), the kernel ref is gone; just drop the fiber ref.
  ~MultishotRecvStream() {
    if (slot_ == nullptr) return;
    slot_->abandoned = true;
    slot_->waiting = false;
    slot_->return_pending_buffers();
    if (slot_->armed && !slot_->kernel_done) {
      // Still armed: async-cancel it. The late terminal CQE drops the kernel ref.
      p->submit_cancel(to_multishot_token(slot_), kCancelToken);
    }
    slot_->release();  // fiber ref
  }

  struct NextAwaitable {
    MultishotRecvStream* self;

    // Ready without suspending IF a completion is already queued (a prior reap batch
    // delivered >=1 segment the fiber hasn't consumed yet -- common when the kernel
    // had several buffered). Only suspend when the queue is empty AND we still expect
    // more from the kernel. The first-call arm happens in await_suspend (we always
    // suspend the first time since nothing is queued yet).
    bool await_ready() noexcept {
      MultishotSlot* sl = self->slot_;
      // kernel_done + empty covers the defensive queue-overflow case: report a
      // terminal error instead of suspending for a CQE that can never arrive.
      return sl != nullptr && (!sl->q_empty() || sl->kernel_done);
    }
    bool await_suspend(std::coroutine_handle<>) noexcept {
      MultishotRecvStream* s = self;
      if (s->slot_ == nullptr) {
        // First call: allocate the heap slot + arm the multishot.
        auto* sl = static_cast<MultishotSlot*>(
            s->alloc.allocate(sizeof(MultishotSlot), alignof(MultishotSlot)));
        if (sl == nullptr) return false;  // OOM -> await_resume returns error chunk
        new (sl) MultishotSlot{};
        sl->alloc = s->alloc;
        sl->p = s->p;
        sl->bgid = s->bgid;
        s->slot_ = sl;
        sl->io.waker = sched::current_waker();
        if (!s->p->submit_recv_multishot(s->fd, s->bgid, to_multishot_token(sl))) {
          // submit failed: kernel never took it -> drop the kernel ref now (mirrors
          // net_timeout on_submit_failed). Fiber ref dropped in the stream dtor.
          sl->kernel_done = true;  // no completion coming
          (void)sl->q_push(kSubmitFailed, 0, 0);  // synthesize an error chunk for next()
          sl->release();           // kernel ref (never materialized)
          return false;            // resume immediately -> error chunk
        }
        sl->armed = true;
      } else {
        // Subsequent call, queue empty: multishot still armed kernel-side. Re-capture
        // the waker (the fiber may have migrated) and wait for the next completion. Do
        // NOT re-submit.
        s->slot_->io.waker = sched::current_waker();
      }
      s->slot_->waiting = true;
      sched::suspend();
      return false;
    }
    RecvBuf await_resume() noexcept {
      MultishotRecvStream* s = self;
      MultishotSlot* sl = s->slot_;
      if (sl == nullptr || sl->q_empty()) {  // OOM, queue overflow, or spurious wake
        RecvBuf b;
        b.set_err(sl != nullptr && sl->overflowed ? -105 : kSubmitFailed);
        s->ended_ = true;
        return b;
      }
      MultishotSlot::Pending pd = sl->q_pop();
      std::int64_t res = pd.result;
      bool more = (pd.flags & Completion::kMore) != 0;
      bool has_buf = (pd.flags & Completion::kBuffer) != 0 && res > 0;
      if (res <= 0) {
        // Error / peer close / -ECANCELED: terminal (a res<=0 completion always has
        // kMore clear). No buffer to return.
        s->ended_ = true;
        RecvBuf b(s->p, s->bgid, nullptr, 0, 0, false, false);
        b.set_err(res);
        return b;
      }
      const void* data = has_buf ? s->p->buf_ring_addr(s->bgid, pd.buffer_id) : nullptr;
      if (!more) s->ended_ = true;
      return RecvBuf(s->p, s->bgid, data, static_cast<std::uint32_t>(res),
                     pd.buffer_id, has_buf, more);
    }
  };

  // co_await stream.next() -> RecvBuf. Returns a terminal chunk (more()==false) once
  // the multishot ends; do not call again after that.
  NextAwaitable next() noexcept { return NextAwaitable{this}; }
  bool ended() const noexcept { return ended_; }
};

}  // namespace hyle::io
