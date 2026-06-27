#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_MEM_API
#elif defined(HYLE_MEM_BUILD)
#  define HYLE_MEM_API HYLE_EXPORT
#else
#  define HYLE_MEM_API HYLE_IMPORT
#endif

namespace hyle::mem {

// Default alignment when a caller does not ask for more (matches the strictest
// fundamental scalar; backends never return less than this).
inline constexpr std::size_t kDefaultAlignment = alignof(std::max_align_t);

// ---- C-ABI allocator vtable -------------------------------------------------
// ONE function carries alloc/free/realloc, plus alignment (clay's vtable dropped
// alignment -- the bug this milestone fixes). Dispatch by the args:
//   ptr==nullptr, new_size>0  -> allocate(new_size, alignment)
//   ptr!=nullptr, new_size==0 -> free(ptr)                 (returns nullptr)
//   ptr!=nullptr, new_size>0  -> reallocate(...)           (returns new ptr)
// old_size lets sized/aligned backends free/copy precisely. Returns nullptr on
// OOM (allocation is a hot path; failure is a value, not a Status).
using ReallocFn = void* (*)(void* state, void* ptr, std::size_t old_size,
                            std::size_t new_size, std::size_t alignment) noexcept;

// ---- injectable allocator handle --------------------------------------------
// {fn, state}: trivially copyable, ABI-clean (function pointer + void*). Passed
// by value through every allocating API (charter: explicit injection, no hidden
// global heap). Methods are inline pure-forwarders -- zero overhead, the wrapper
// never costs a DLL call; only the backing fn lives in a TU.
class Allocator {
 public:
  Allocator() noexcept = default;  // null/invalid allocator
  Allocator(ReallocFn fn, void* state) noexcept : fn_(fn), state_(state) {}

  bool valid() const noexcept { return fn_ != nullptr; }

  // These are logically const: they dispatch through fn_ without mutating the
  // handle itself (so allocating from a `const Allocator&` is valid).
  void* allocate(std::size_t size,
                 std::size_t alignment = kDefaultAlignment) const noexcept {
    return fn_(state_, nullptr, 0, size, alignment);
  }
  void deallocate(void* ptr, std::size_t size,
                  std::size_t alignment = kDefaultAlignment) const noexcept {
    fn_(state_, ptr, size, 0, alignment);
  }
  void* reallocate(void* ptr, std::size_t old_size, std::size_t new_size,
                   std::size_t alignment = kDefaultAlignment) const noexcept {
    return fn_(state_, ptr, old_size, new_size, alignment);
  }

  // Raw vtable accessors (for handing the seam across a C-ABI boundary).
  ReallocFn fn() const noexcept { return fn_; }
  void* state() const noexcept { return state_; }

  // ---- typed convenience layer (clay allocator.h parity) --------------------
  // Header-inline (hot path, no DLL hop). Mirrors clay's allocate<T>/callocate/
  // release/construct/destroy/make_unique.

  // Raw storage for `numel` T's, aligned for T. No construction.
  template <class T>
  T* allocate_n(std::size_t numel = 1, std::size_t alignment = alignof(T)) const noexcept {
    return static_cast<T*>(allocate(sizeof(T) * numel, alignment));
  }

  // Zero-initialized storage for trivially-constructible T.
  template <class T>
  T* callocate(std::size_t numel = 1) const noexcept {
    static_assert(std::is_trivially_constructible_v<T>,
                  "callocate requires trivially-constructible T");
    std::size_t sz = sizeof(T) * numel;
    void* p = allocate(sz, alignof(T));
    if (p != nullptr) std::memset(p, 0, sz);
    return static_cast<T*>(p);
  }

  // Free typed storage from allocate_n/callocate (no destructor call).
  template <class T>
  void release(T* ptr, std::size_t numel = 1) const noexcept {
    deallocate(ptr, sizeof(T) * numel, alignof(T));
  }

  // Allocate + placement-new a T. nullptr if allocation failed.
  template <class T, class... Args>
  T* construct(Args&&... args) const {
    void* p = allocate(sizeof(T), alignof(T));
    if (p == nullptr) return nullptr;
    return ::new (p) T(std::forward<Args>(args)...);
  }

  // ~T() then free through THIS allocator (never global delete).
  template <class T>
  void destroy(T* ptr) const noexcept {
    if (ptr != nullptr) {
      ptr->~T();
      release(ptr);
    }
  }

  // unique_ptr whose deleter destroys through this allocator. The deleter holds
  // the raw fn/state (NOT an Allocator by value): inside Allocator's own
  // definition the type is incomplete, so a by-value Allocator member is
  // ill-formed (GCC rejects it; MSVC let it slide).
  template <class T>
  struct Deleter {
    ReallocFn fn;
    void* state;
    void operator()(T* p) const noexcept {
      if (p != nullptr) {
        p->~T();
        fn(state, p, sizeof(T), 0, alignof(T));
      }
    }
  };
  template <class T, class... Args>
  std::unique_ptr<T, Deleter<T>> make_unique(Args&&... args) const {
    return std::unique_ptr<T, Deleter<T>>(
        construct<T>(std::forward<Args>(args)...), Deleter<T>{fn_, state_});
  }

 private:
  ReallocFn fn_ = nullptr;
  void* state_ = nullptr;
};

// Process-wide SystemRAM allocator. mem-01: bare aligned malloc/free; mem-02
// swaps mimalloc in behind this same handle with zero caller changes.
HYLE_MEM_API Allocator system_allocator() noexcept;

}  // namespace hyle::mem
