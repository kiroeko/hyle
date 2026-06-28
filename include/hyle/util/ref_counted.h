#pragma once

#include <atomic>
#include <compare>
#include <concepts>
#include <cstdint>
#include <utility>

#include "hyle/macros.h"

// Intrusive reference counting (clay ref_counted). The count lives INSIDE the
// object (CRTP base), so a Ref<T> is one pointer with no separate control block
// -- cheaper than std::shared_ptr and ABI-stable across the DLL seam (a control
// block's layout is not). incref is relaxed; decref is acq_rel (textbook
// correct: a new ref can only form from an existing one, so only the final
// release must synchronize-with the destroying thread).
//
// Divergences from clay (deliberate):
//  - clay's decref calls unqualified `destroy()`, which name-resolves to the
//    BASE's destroy -- a subclass override is never reached. We dispatch via
//    static_cast<Sub*>(this)->destroy(), so custom teardown actually works.
//  - dropped clay's process-global `total_refcounted_objects` atomic: it adds a
//    contended RMW to every construct/destruct (perf-hostile). Leaks are caught
//    by ASan instead.
//  - Ref has no implicit `operator T*` (clay had one): it invites `delete ref`
//    and silent decays. Matches std::unique_ptr/shared_ptr -- use get().

namespace hyle::util {

// CRTP base: `class Widget : public RefCounted<Widget>`. A fresh object starts
// at refcount 1 (the constructing reference). No vtable is added; subclasses may
// add their own. To customize teardown, give the subclass an accessible
// `destroy()` (public, or `friend class hyle::util::RefCounted<Sub>;`).
template <class Sub>
class RefCounted {
 public:
  RefCounted() noexcept : refcount_(1) {}
  HYLE_DISABLE_COPY_AND_MOVE(RefCounted);

  // New reference. relaxed: forming a ref requires already holding one, and
  // handing a ref across threads carries its own synchronization.
  void incref(std::uint32_t n = 1) noexcept {
    refcount_.fetch_add(n, std::memory_order_relaxed);
  }

  // Drop a reference; destroy on the last one. release on the decrement so a
  // thread's writes are visible to whoever frees; then a standalone acquire
  // fence ONLY on the final drop, so the destroying thread sees all prior
  // releases before running destroy(). This is the Boost/libc++ shared_ptr
  // ordering: non-final decrefs pay only release, not a full acq_rel.
  void decref(std::uint32_t n = 1) noexcept {
    if (refcount_.fetch_sub(n, std::memory_order_release) == n) {
      std::atomic_thread_fence(std::memory_order_acquire);
      static_cast<Sub*>(this)->destroy();  // CRTP dispatch (clay bug fixed)
    }
  }

  bool is_unique() const noexcept {
    return refcount_.load(std::memory_order_acquire) == 1;
  }
  std::uint32_t use_count() const noexcept {
    return refcount_.load(std::memory_order_relaxed);
  }

 protected:
  ~RefCounted() = default;  // non-virtual: destroy through the Sub type
  // Default teardown: delete as the concrete type. Subclasses that allocate
  // through a HYLE Allocator override this to free through the same allocator.
  void destroy() { delete static_cast<Sub*>(this); }

 private:
  std::atomic<std::uint32_t> refcount_;
};

// A type that can back a Ref<T>: exposes incref/decref (i.e. derives from
// RefCounted, or duck-types the same surface).
template <class T>
concept Refable = requires(T* p) {
  p->incref();
  p->decref();
};

// Intrusive owning smart pointer. Copy = share (incref); move = steal. Holds a
// strong reference for its lifetime. Construct via make_ref<T>()/adopt_ref()/
// retain_ref() -- the raw-pointer constructor is intentionally absent to keep
// "adopt vs retain" explicit at the call site (clay's implicit Ref(T*) adopted
// silently, a known footgun its own comments warned about).
template <Refable T>
class Ref {
 public:
  Ref() noexcept = default;
  Ref(std::nullptr_t) noexcept {}  // NOLINT(*-explicit): Ref r = nullptr;

  Ref(const Ref& o) noexcept : ptr_(o.ptr_) {
    if (ptr_) ptr_->incref();
  }
  Ref(Ref&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }

  // Upcast Ref<Derived> -> Ref<Base> (copy and move), enabled only when the
  // pointer conversion is valid.
  template <class U>
    requires std::convertible_to<U*, T*>
  Ref(const Ref<U>& o) noexcept : ptr_(o.get()) {  // NOLINT(*-explicit)
    if (ptr_) ptr_->incref();
  }
  template <class U>
    requires std::convertible_to<U*, T*>
  Ref(Ref<U>&& o) noexcept : ptr_(o.release()) {}  // NOLINT(*-explicit)

  ~Ref() {
    if (ptr_) ptr_->decref();
  }

  Ref& operator=(const Ref& o) noexcept {
    if (ptr_ != o.ptr_) {
      if (o.ptr_) o.ptr_->incref();
      if (ptr_) ptr_->decref();
      ptr_ = o.ptr_;
    }
    return *this;
  }
  Ref& operator=(Ref&& o) noexcept {
    if (this != &o) {
      if (ptr_) ptr_->decref();
      ptr_ = o.ptr_;
      o.ptr_ = nullptr;
    }
    return *this;
  }
  Ref& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept {
    HYLE_FATAL_ASSERT(ptr_ != nullptr);
    return *ptr_;
  }
  T* operator->() const noexcept {
    HYLE_FATAL_ASSERT(ptr_ != nullptr);
    return ptr_;
  }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  // Drop our reference (decref) and become null.
  void reset() noexcept {
    if (ptr_) {
      ptr_->decref();
      ptr_ = nullptr;
    }
  }
  // Relinquish ownership WITHOUT decref: caller now owns the +1.
  T* release() noexcept {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }
  void swap(Ref& o) noexcept { std::swap(ptr_, o.ptr_); }

  bool operator==(const Ref& o) const noexcept { return ptr_ == o.ptr_; }
  bool operator==(const T* p) const noexcept { return ptr_ == p; }
  std::strong_ordering operator<=>(const Ref& o) const noexcept {
    return ptr_ <=> o.ptr_;
  }

 private:
  template <Refable U>
  friend class Ref;
  template <Refable U>
  friend Ref<U> adopt_ref(U*) noexcept;

  explicit Ref(T* p) noexcept : ptr_(p) {}  // adopt; used by factories
  T* ptr_ = nullptr;
};

// Construct a T and adopt its initial +1 (RefCounted starts at 1, so NO extra
// incref). The common way to mint a reference-counted object.
template <class T, class... Args>
HYLE_FORCE_INLINE Ref<T> make_ref(Args&&... args) {
  return adopt_ref(new T(std::forward<Args>(args)...));
}

// Adopt an existing +1 reference (no incref): for a raw pointer that already
// carries a reference you own (e.g. returned across an API that did the incref).
template <Refable T>
HYLE_FORCE_INLINE Ref<T> adopt_ref(T* p) noexcept {
  return Ref<T>(p);
}

// Add a NEW reference to a borrowed pointer (incref): you do not own `p`'s
// count and want to keep it alive.
template <Refable T>
HYLE_FORCE_INLINE Ref<T> retain_ref(T* p) noexcept {
  if (p) p->incref();
  return adopt_ref(p);
}

template <Refable T>
HYLE_FORCE_INLINE void swap(Ref<T>& a, Ref<T>& b) noexcept {
  a.swap(b);
}

}  // namespace hyle::util
