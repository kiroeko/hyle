#pragma once

// Only these static, trace-OFF native Graph test executables replace new.
// The caller arms one failure immediately before cold report construction;
// it disarms before throwing, and never affects native owner/driver threads.
// Shared-library and profiler allocations are outside this injection seam.
#if defined(HYLE_STATIC) && !defined(HYLE_TRACE_TRACY)
#include <cstdlib>
#include <new>

namespace native_report_test {
inline thread_local bool fail_next = false;
inline thread_local bool failed = false;
} // namespace native_report_test

void *operator new(std::size_t bytes) {
  if (native_report_test::fail_next) {
    native_report_test::fail_next = false;
    native_report_test::failed = true;
    throw std::bad_alloc();
  }
  if (void *memory = std::malloc(bytes == 0 ? 1 : bytes))
    return memory;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}
#endif
