#pragma once

#include <utility>

#include "hyle/core/compiler.h"

namespace hyle::mem {

// Move-only scoped loan from an object pool. Owner must provide
// return_object(T*) noexcept. The pool must outlive every loan. `release()`
// deliberately detaches without returning the object; the caller then owns the
// obligation to pass it back to the originating pool.
template <class T, class Owner> class PoolLoan {
public:
  PoolLoan() noexcept = default;
  PoolLoan(Owner *owner, T *ptr) noexcept : owner_(owner), ptr_(ptr) {
    HYLE_FATAL_ASSERT(ptr_ == nullptr || owner_ != nullptr);
  }
  ~PoolLoan() { reset(); }

  PoolLoan(const PoolLoan&) = delete;
  PoolLoan& operator=(const PoolLoan&) = delete;
  PoolLoan(PoolLoan &&other) noexcept
      : owner_(std::exchange(other.owner_, nullptr)),
        ptr_(std::exchange(other.ptr_, nullptr)) {}
  PoolLoan &operator=(PoolLoan &&other) noexcept {
    if (this != &other) {
      reset();
      owner_ = std::exchange(other.owner_, nullptr);
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }
  T *get() const noexcept { return ptr_; }
  T &operator*() const noexcept { return *ptr_; }
  T *operator->() const noexcept { return ptr_; }

  T *release() noexcept {
    owner_ = nullptr;
    return std::exchange(ptr_, nullptr);
  }
  void reset() noexcept {
    if (ptr_ != nullptr)
      owner_->return_object(ptr_);
    owner_ = nullptr;
    ptr_ = nullptr;
  }

private:
  Owner *owner_ = nullptr;
  T *ptr_ = nullptr;
};

} // namespace hyle::mem
