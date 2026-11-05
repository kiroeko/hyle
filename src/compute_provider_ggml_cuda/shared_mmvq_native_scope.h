#pragma once
#include "shared_mmvq_state.h"
extern "C" bool hyle_shared_mmvq_begin(void *) noexcept;
extern "C" bool hyle_shared_mmvq_check(void *) noexcept;
extern "C" void hyle_shared_mmvq_end(void *) noexcept;
struct SharedMmvqNativeScope {
  explicit SharedMmvqNativeScope(hyle_shared_mmvq::Plan *value) : plan(value) {
    if (!hyle_shared_mmvq_begin(plan))
      throw std::logic_error("shared MMVQ invocation admission failed");
  }
  ~SharedMmvqNativeScope() { hyle_shared_mmvq_end(plan); }
  void check() const {
    if (!hyle_shared_mmvq_check(plan))
      throw std::logic_error("shared MMVQ encoder contract failed");
  }
  SharedMmvqNativeScope(const SharedMmvqNativeScope &) = delete;
  SharedMmvqNativeScope &operator=(const SharedMmvqNativeScope &) = delete;
  hyle_shared_mmvq::Plan *plan;
};
