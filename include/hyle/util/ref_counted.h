#pragma once

#include <atomic>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"

// Intrusive reference counting (clay ref_counted). The count lives INSIDE the
// object (CRTP base), so a Ref<T> is one pointer with no separate control block.
// This is a native same-toolchain C++ ownership type; its object layout is not
// a cross-runtime contract.
// The default thread-safe policy uses relaxed incref and release decref plus an
// acquire fence only on the final drop (a new ref can only form from an
// existing one). ThreadUnsafeRefCounted is an explicit type-level opt-in for a
// complete reference lifetime confined to one thread.
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

enum class RefCountMode : std::uint8_t {
  kThreadSafe,
  kThreadUnsafe,
};

namespace detail {

template <RefCountMode Mode>
class RefCounter;

template <>
class RefCounter<RefCountMode::kThreadSafe> {
 public:
  explicit RefCounter(std::uint32_t value) noexcept : value_(value) {}

  void increment() const noexcept {
    value_.fetch_add(1, std::memory_order_relaxed);
  }
  std::uint32_t fetch_add(std::uint32_t value) const noexcept {
    return value_.fetch_add(value, std::memory_order_relaxed);
  }
  std::uint32_t fetch_sub(std::uint32_t value) const noexcept {
    return value_.fetch_sub(value, std::memory_order_release);
  }
  bool is_unique() const noexcept {
    return value_.load(std::memory_order_acquire) == 1;
  }
  std::uint32_t load() const noexcept {
    return value_.load(std::memory_order_relaxed);
  }
  static void acquire_before_destroy() noexcept {
    std::atomic_thread_fence(std::memory_order_acquire);
  }

 private:
  mutable std::atomic<std::uint32_t> value_;
};

template <>
class RefCounter<RefCountMode::kThreadUnsafe> {
 public:
  explicit RefCounter(std::uint32_t value) noexcept : value_(value) {}

  void increment() const noexcept { ++value_; }
  std::uint32_t fetch_add(std::uint32_t value) const noexcept {
    const std::uint32_t old = value_;
    value_ += value;
    return old;
  }
  std::uint32_t fetch_sub(std::uint32_t value) const noexcept {
    const std::uint32_t old = value_;
    value_ -= value;
    return old;
  }
  bool is_unique() const noexcept { return value_ == 1; }
  std::uint32_t load() const noexcept { return value_; }
  static void acquire_before_destroy() noexcept {}

 private:
  mutable std::uint32_t value_;
};

}  // namespace detail

// CRTP base: `class Widget : public RefCounted<Widget>`. A fresh object starts
// at refcount 1 (the constructing reference). No vtable is added; subclasses may
// add their own. To customize teardown, give the subclass an accessible
// `destroy()` (public, or friend the exact RefCounted<Sub, Mode> base).
template <class Sub, RefCountMode Mode = RefCountMode::kThreadSafe>
class RefCounted {
 public:
  // Lets Ref<T> detect the otherwise-dangerous shape
  // `Derived : RefCounted<Base>` when Base is not polymorphically destructible.
  // In that shape Base::decref() necessarily destroys through Base.
  using hyle_refcounted_subtype = Sub;

  RefCounted() noexcept : refcount_(1) {}
  RefCounted(const RefCounted&) = delete;
  RefCounted& operator=(const RefCounted&) = delete;
  RefCounted(RefCounted&&) = delete;
  RefCounted& operator=(RefCounted&&) = delete;

  // New reference. relaxed: forming a ref requires already holding one, and
  // handing a ref across threads carries its own synchronization. The common
  // single-reference operation deliberately does not consume the old value;
  // MSVC can therefore emit `lock inc` instead of the slower value-returning
  // `lock xadd` used by a checked batch operation. As with mature intrusive
  // pointers, callers must already own a live reference and must not overflow
  // the count; ordinary Ref copies maintain that invariant by construction.
  void incref() const noexcept { refcount_.increment(); }

  // Explicit batch change is uncommon and fully validates its arithmetic.
  void incref(std::uint32_t n) const noexcept {
    HYLE_FATAL_ASSERT(n != 0);
    const std::uint32_t old = refcount_.fetch_add(n);
    HYLE_FATAL_ASSERT(old != 0);
    HYLE_FATAL_ASSERT(old <=
                      std::numeric_limits<std::uint32_t>::max() - n);
  }

  // Drop a reference; destroy on the last one. release on the decrement so a
  // thread's writes are visible to whoever frees; then a standalone acquire
  // fence ONLY on the final drop, so the destroying thread sees all prior
  // releases before running destroy(). This is the Boost/libc++ shared_ptr
  // ordering: non-final decrefs pay only release, not a full acq_rel.
  void decref() const noexcept {
    // A release without a live +1 is an ownership-contract violation. Keeping
    // that precondition out of this value-returning hot path is what lets Ref
    // match shared_ptr on MSVC; the explicit batch overload remains checked.
    const std::uint32_t old = refcount_.fetch_sub(1);
    if (old == 1) {
      decltype(refcount_)::acquire_before_destroy();
      const_cast<Sub*>(static_cast<const Sub*>(this))->destroy();
    }
  }

  // Explicit batch change is uncommon and fully validates its arithmetic.
  void decref(std::uint32_t n) const noexcept {
    HYLE_FATAL_ASSERT(n != 0);
    const std::uint32_t old = refcount_.fetch_sub(n);
    HYLE_FATAL_ASSERT(old >= n);
    if (old == n) {
      decltype(refcount_)::acquire_before_destroy();
      // Reference ownership is logically mutable even through Ref<const T>.
      // Teardown still runs on the concrete non-const object.
      const_cast<Sub*>(static_cast<const Sub*>(this))->destroy();
    }
  }

  bool is_unique() const noexcept { return refcount_.is_unique(); }
  std::uint32_t use_count() const noexcept { return refcount_.load(); }

 protected:
  ~RefCounted() = default;  // non-virtual: destroy through the Sub type
  // Default teardown: delete as the concrete type. Subclasses that allocate
  // through a HYLE AllocatorRef override this to free through the same allocator.
  // destroy() must not throw: it is reached from noexcept ownership release.
  void destroy() { delete static_cast<Sub*>(this); }

 private:
  detail::RefCounter<Mode> refcount_;
};

// Explicit opt-in for objects whose complete reference lifetime is confined to
// one thread. Ref<T> itself stays the same one-word owner, but copies/drops use
// plain integer operations. Moving any owner or borrowed pointer across threads
// violates this type's contract.
template <class Sub>
using ThreadUnsafeRefCounted =
    RefCounted<Sub, RefCountMode::kThreadUnsafe>;

// A type that can back a Ref<T>: exposes incref/decref (i.e. derives from
// RefCounted, or duck-types the same surface).
template <class T>
concept Refable = requires(T* p) {
  p->incref();
  p->decref();
};

namespace detail {

template <class T, class = void>
struct ref_destruction_is_safe : std::true_type {};

// RefCounted<Base>'s last release dispatches to Base::destroy(). A further
// Derived is safe only when Base has a virtual destructor; otherwise even a
// Ref<Derived> would eventually execute `delete Base*`.
template <class T>
struct ref_destruction_is_safe<
    T, std::void_t<
           typename std::remove_cv_t<T>::hyle_refcounted_subtype>>
    : std::bool_constant<
          std::same_as<std::remove_cv_t<T>,
                       typename std::remove_cv_t<T>::hyle_refcounted_subtype> ||
          std::has_virtual_destructor_v<
              typename std::remove_cv_t<T>::hyle_refcounted_subtype>> {};

}  // namespace detail

template <class T>
concept RefCompatible =
    Refable<T> && detail::ref_destruction_is_safe<T>::value;

// Intrusive owning smart pointer. Copy = share (incref); move = steal. Holds a
// strong reference for its lifetime. Construct via make_ref<T>()/adopt_ref()/
// retain_ref() -- the raw-pointer constructor is intentionally absent to keep
// "adopt vs retain" explicit at the call site (clay's implicit Ref(T*) adopted
// silently, a known footgun its own comments warned about).
template <RefCompatible T>
class Ref {
 public:
  using element_type = T;

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
  template <class U>
    requires std::convertible_to<U*, T*>
  Ref& operator=(const Ref<U>& o) noexcept {
    Ref tmp(o);
    swap(tmp);
    return *this;
  }
  template <class U>
    requires std::convertible_to<U*, T*>
  Ref& operator=(Ref<U>&& o) noexcept {
    Ref tmp(std::move(o));
    swap(tmp);
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

 private:
  template <RefCompatible U>
  friend class Ref;
  template <RefCompatible U>
  friend Ref<U> adopt_ref(U*) noexcept;

  explicit Ref(T* p) noexcept : ptr_(p) {}  // adopt; used by factories
  T* ptr_ = nullptr;
};

// Pointer-identity comparisons are heterogeneous like std::shared_ptr and
// Boost intrusive_ptr.  In particular Ref<Derived> vs Ref<Base> does not mint
// a temporary owner (and therefore does no atomic incref/decref) merely to
// answer an identity question.
template <RefCompatible T, RefCompatible U>
HYLE_FORCE_INLINE bool operator==(const Ref<T>& left,
                                  const Ref<U>& right) noexcept {
  return left.get() == right.get();
}

template <RefCompatible T, RefCompatible U>
HYLE_FORCE_INLINE std::strong_ordering operator<=>(
    const Ref<T>& left, const Ref<U>& right) noexcept {
  // std::compare_three_way gives object pointers the implementation-defined
  // strict total order required by associative containers. Built-in pointer
  // <=> leaves unrelated complete objects unordered by a portable contract.
  return std::compare_three_way{}(left.get(), right.get());
}

template <RefCompatible T>
HYLE_FORCE_INLINE bool operator==(const Ref<T>& ref,
                                  std::nullptr_t) noexcept {
  return ref.get() == nullptr;
}

template <RefCompatible T>
HYLE_FORCE_INLINE std::strong_ordering operator<=>(
    const Ref<T>& ref, std::nullptr_t) noexcept {
  return std::compare_three_way{}(ref.get(), static_cast<T*>(nullptr));
}

template <RefCompatible T, class U>
  requires requires(T* left, U* right) { left == right; }
HYLE_FORCE_INLINE bool operator==(const Ref<T>& ref, U* pointer) noexcept {
  return ref.get() == pointer;
}

// Construct a T and adopt its initial +1 (RefCounted starts at 1, so NO extra
// incref). The common way to mint a reference-counted object. This is the
// ordinary C++ `new` path, so allocation and T's constructor may throw.
template <RefCompatible T, class... Args>
HYLE_FORCE_INLINE Ref<T> make_ref(Args&&... args) {
  return adopt_ref(new T(std::forward<Args>(args)...));
}

// Adopt an existing +1 reference (no incref): for a raw pointer that already
// carries a reference you own (e.g. returned across an API that did the incref).
template <RefCompatible T>
HYLE_FORCE_INLINE Ref<T> adopt_ref(T* p) noexcept {
  return Ref<T>(p);
}

// Add a NEW reference to a borrowed pointer (incref): you do not own `p`'s
// count and want to keep it alive.
template <RefCompatible T>
HYLE_FORCE_INLINE Ref<T> retain_ref(T* p) noexcept {
  if (p) p->incref();
  return adopt_ref(p);
}

// Ownership-preserving casts mirror the standard smart-pointer surface. A
// successful rvalue cast transfers the existing +1 without an atomic round
// trip; a failed dynamic rvalue cast leaves the source owner unchanged.
template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { static_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> static_pointer_cast(const Ref<U>& source) noexcept {
  return retain_ref(static_cast<T*>(source.get()));
}

template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { static_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> static_pointer_cast(Ref<U>&& source) noexcept {
  T* pointer = static_cast<T*>(source.get());
  (void)source.release();
  return adopt_ref(pointer);
}

template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { dynamic_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> dynamic_pointer_cast(const Ref<U>& source) noexcept {
  return retain_ref(dynamic_cast<T*>(source.get()));
}

template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { dynamic_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> dynamic_pointer_cast(Ref<U>&& source) noexcept {
  T* pointer = dynamic_cast<T*>(source.get());
  if (pointer == nullptr) return {};
  (void)source.release();
  return adopt_ref(pointer);
}

template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { const_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> const_pointer_cast(const Ref<U>& source) noexcept {
  return retain_ref(const_cast<T*>(source.get()));
}

template <RefCompatible T, RefCompatible U>
  requires requires(U* pointer) { const_cast<T*>(pointer); }
HYLE_FORCE_INLINE Ref<T> const_pointer_cast(Ref<U>&& source) noexcept {
  T* pointer = const_cast<T*>(source.get());
  (void)source.release();
  return adopt_ref(pointer);
}

template <RefCompatible T>
HYLE_FORCE_INLINE void swap(Ref<T>& a, Ref<T>& b) noexcept {
  a.swap(b);
}

}  // namespace hyle::util

namespace std {

// Hashes pointee identity, matching Ref equality and the standard smart-pointer
// convention. Specializing std::hash for this user-defined template is the
// intended extension point and enables Ref keys in unordered containers.
template <hyle::util::RefCompatible T>
struct hash<hyle::util::Ref<T>> {
  size_t operator()(const hyle::util::Ref<T>& ref) const noexcept {
    return hash<T*>{}(ref.get());
  }
};

}  // namespace std
