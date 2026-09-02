#pragma once

#include <atomic>

#include "hyle/container/intrusive_hlist.h"
#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

namespace hyle::sched {

class CancelCallback;
class CancelScope;

namespace detail {
struct CancelNode;
struct CancelTree;

// Intrusive callback storage lives in CancelCallback, so registering an
// in-flight IO operation does not allocate. Tree topology and callback links
// are serialized by the tree control-path lock; the cancelled bit remains the
// only state read by the hot polling path.
struct CancelCallbackRecord {
  using Fn = void (*)(void* context) noexcept;

  CancelNode* node = nullptr;
  CancelTree* tree = nullptr;  // stable until owner unregisters
  Fn fn = nullptr;
  void* context = nullptr;
  container::SafeIntrusiveHListHook hook;
};
}  // namespace detail

// A shareable handle to one node in a cooperative-cancellation tree. Copies
// name the same node; derive() creates a child subtree. Cancelling a node marks
// it and all descendants and invokes their registered callbacks exactly once.
//
// Tree mutation is thread-safe. cancel(), derive(), copy/drop and callback
// registration are control-path operations serialized per tree. is_cancelled()
// is one O(1) acquire load and takes no lock, allocation or syscall.
class CancelToken {
 public:
  CancelToken() noexcept = default;

  HYLE_SCHED_API static Result<CancelToken> try_create_root(
      mem::AllocatorRef allocator) noexcept;
  HYLE_SCHED_API static CancelToken root(mem::AllocatorRef allocator) noexcept;

  HYLE_SCHED_API Result<CancelToken> try_derive() const noexcept;
  HYLE_SCHED_API CancelToken derive() const noexcept;

  HYLE_SCHED_API CancelToken(const CancelToken& other) noexcept;
  HYLE_SCHED_API CancelToken& operator=(const CancelToken& other) noexcept;
  HYLE_SCHED_API CancelToken(CancelToken&& other) noexcept;
  HYLE_SCHED_API CancelToken& operator=(CancelToken&& other) noexcept;
  HYLE_SCHED_API ~CancelToken();

  bool valid() const noexcept { return node_ != nullptr; }
  HYLE_SCHED_API void cancel() const noexcept;
  HYLE_SCHED_API bool is_cancelled() const noexcept;

 private:
  friend class CancelCallback;
  friend class CancelScope;

  explicit CancelToken(detail::CancelNode* node) noexcept : node_(node) {}
  HYLE_SCHED_API void retain_() noexcept;
  HYLE_SCHED_API void reset_() noexcept;
  HYLE_SCHED_API std::atomic<bool>* flag_() const noexcept;

  detail::CancelNode* node_ = nullptr;
};

// Adds a token to the active cancellation sources of the current sched
// execution context. On a Fiber the scope migrates with that Fiber; inside a
// worker-stack schedule_task callback it is thread-confined for the callback's
// lifetime. Nested scopes compose with JoinHandle::abort instead of replacing
// it. The token (or one of its copies) must outlive the scope.
class CancelScope {
 public:
  explicit CancelScope(const CancelToken& token) noexcept
      : scope_(token.flag_()) {}
  CancelScope(const CancelScope&) = delete;
  CancelScope& operator=(const CancelScope&) = delete;

 private:
  detail::ScopedCancelFlag scope_;
};

// Intrusive RAII callback for actively cancelling an already-started operation.
// Registration and destruction may occur on different threads. The callback is
// detached before invocation, may unregister sibling callbacks, and runs under
// the tree's recursive control lock. Keep a copy of the token alive until this
// object is destroyed; violating that lifetime contract fails fast.
class CancelCallback {
 public:
  using Fn = detail::CancelCallbackRecord::Fn;

  HYLE_SCHED_API CancelCallback(const CancelToken& token, Fn fn,
                                void* context) noexcept;
  HYLE_SCHED_API ~CancelCallback();
  CancelCallback(const CancelCallback&) = delete;
  CancelCallback& operator=(const CancelCallback&) = delete;
  CancelCallback(CancelCallback&&) = delete;
  CancelCallback& operator=(CancelCallback&&) = delete;

  HYLE_SCHED_API void unregister() noexcept;

 private:
  detail::CancelCallbackRecord record_{};
};

// True when any active cancellation source on the current Fiber or
// worker-stack task is set. False outside an active sched execution scope. Cost
// is O(k) atomic loads where k is the number of nested sources (normally one
// JoinHandle flag plus zero or one request token).
HYLE_SCHED_API bool is_cancel_requested() noexcept;

#define HYLE_CHECK_CANCEL()                              \
  do {                                                   \
    if (::hyle::sched::is_cancel_requested())            \
      return ::hyle::Status::Cancelled("cancelled");     \
  } while (0)

}  // namespace hyle::sched
