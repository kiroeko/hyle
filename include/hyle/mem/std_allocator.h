#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

// Traditional C++ Allocator named-requirement adapter over hyle::mem::Allocator.
// PMR covers std::pmr consumers; this adapter covers allocator-template APIs such
// as std::vector<T, A>, std::allocate_shared, and third-party containers that
// rebind an allocator type. Like Allocator and PmrResource, it borrows the
// backing state: every container/control block using it must be destroyed before
// that state.

namespace hyle::mem {

// Stateless named-requirement adapter over SystemAllocator.  This is the
// normal default-heap shape for allocator-template APIs: empty/EBO-friendly,
// always equal, and compile-time bound.  StdAllocator<T> below deliberately
// remains the runtime-injected shape and stores an Allocator handle.
template <class T>
class SystemStdAllocator {
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T> &&
                    !std::is_void_v<T>,
                "SystemStdAllocator requires an unqualified object type");

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind {
    using other = SystemStdAllocator<U>;
  };

  constexpr SystemStdAllocator() noexcept = default;
  template <class U>
  constexpr SystemStdAllocator(const SystemStdAllocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_type count) {
    if (count > max_size()) throw std::bad_array_new_length{};
    if (count == 0) return nullptr;
    void* pointer = SystemAllocator::allocate(count * sizeof(T), alignof(T));
    if (pointer == nullptr) throw std::bad_alloc{};
    return static_cast<T*>(pointer);
  }

  void deallocate(T* pointer, size_type count) noexcept {
    HYLE_FATAL_ASSERT(count <= max_size());
    SystemAllocator::deallocate(pointer, count * sizeof(T), alignof(T));
  }

  static constexpr size_type max_size() noexcept {
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  constexpr SystemStdAllocator
  select_on_container_copy_construction() const noexcept {
    return {};
  }
};

template <class T, class U>
constexpr bool operator==(SystemStdAllocator<T>,
                          SystemStdAllocator<U>) noexcept {
  return true;
}

template <class T>
class StdAllocator {
  static_assert(!std::is_const_v<T> && !std::is_volatile_v<T> &&
                    !std::is_void_v<T>,
                "StdAllocator requires an unqualified object type");

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  // A container copied/moved/swapped with unequal backing state must retain its
  // own allocator so deallocation always returns to the allocating backing.
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;
  using is_always_equal = std::false_type;

  template <class U>
  struct rebind {
    using other = StdAllocator<U>;
  };

  // Runtime policy injection must be explicit. A default constructor that
  // silently chose system_allocator() made the slower stateful adapter look
  // like the normal system-heap shape; use empty SystemStdAllocator<T> for
  // that case instead.
  StdAllocator() = delete;

  explicit StdAllocator(Allocator allocator) noexcept : allocator_(allocator) {
    HYLE_FATAL_ASSERT(allocator_.valid());
  }

  template <class U>
  StdAllocator(const StdAllocator<U>& other) noexcept
      : allocator_(other.allocator()) {}

  [[nodiscard]] T* allocate(size_type count) {
    if (count > max_size()) throw std::bad_array_new_length{};
    if (count == 0) return nullptr;
    void* pointer = allocator_.allocate(count * sizeof(T), alignof(T));
    if (pointer == nullptr) throw std::bad_alloc{};
    return static_cast<T*>(pointer);
  }

  void deallocate(T* pointer, size_type count) noexcept {
    HYLE_FATAL_ASSERT(count <= max_size());
    allocator_.deallocate(pointer, count * sizeof(T), alignof(T));
  }

  size_type max_size() const noexcept {
    return (std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  StdAllocator select_on_container_copy_construction() const noexcept {
    return *this;
  }

  Allocator allocator() const noexcept { return allocator_; }

 private:
  Allocator allocator_;
};

template <class T, class U>
bool operator==(const StdAllocator<T>& left,
                const StdAllocator<U>& right) noexcept {
  const Allocator a = left.allocator();
  const Allocator b = right.allocator();
  return a.alloc_fn() == b.alloc_fn() && a.free_fn() == b.free_fn() &&
         a.realloc_fn() == b.realloc_fn() && a.state() == b.state();
}

}  // namespace hyle::mem
