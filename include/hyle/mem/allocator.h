#pragma once

#include <cstddef>
#include <cstdint>

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

  void* allocate(std::size_t size,
                 std::size_t alignment = kDefaultAlignment) noexcept {
    return fn_(state_, nullptr, 0, size, alignment);
  }
  void deallocate(void* ptr, std::size_t size,
                  std::size_t alignment = kDefaultAlignment) noexcept {
    fn_(state_, ptr, size, 0, alignment);
  }
  void* reallocate(void* ptr, std::size_t old_size, std::size_t new_size,
                   std::size_t alignment = kDefaultAlignment) noexcept {
    return fn_(state_, ptr, old_size, new_size, alignment);
  }

  // Raw vtable accessors (for handing the seam across a C-ABI boundary).
  ReallocFn fn() const noexcept { return fn_; }
  void* state() const noexcept { return state_; }

 private:
  ReallocFn fn_ = nullptr;
  void* state_ = nullptr;
};

// Process-wide SystemRAM allocator. mem-01: bare aligned malloc/free; mem-02
// swaps mimalloc in behind this same handle with zero caller changes.
HYLE_MEM_API Allocator system_allocator() noexcept;

}  // namespace hyle::mem
