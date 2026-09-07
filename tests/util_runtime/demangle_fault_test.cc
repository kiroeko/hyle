// Compile the real implementation in an isolated test TU. Only its platform
// calls and C allocation are substituted; no production hook or copied logic.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/util/demangle.h"

#if defined(_WIN32)
// clang-format off: DbgHelp requires Windows declarations first.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
// clang-format on
#else
#include <cxxabi.h>
#endif

namespace fault {
enum class Mode { kSuccess, kError, kOom };
thread_local Mode mode = Mode::kSuccess;
thread_local int allocations = 0;
thread_local int releases = 0;
thread_local bool fail_cpp_allocation = false;
constexpr std::size_t kLength = 1500;
std::atomic<int> active{0};
std::atomic<int> peak{0};
}  // namespace fault

// Isolated executable: fail only the public owning overload's string
// allocation. Allocation of platform temporaries still uses C malloc and its
// normal owner.
void* operator new(std::size_t size) {
  if (fault::fail_cpp_allocation) throw std::bad_alloc();
  if (void* value = std::malloc(size == 0 ? 1 : size)) return value;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace fault_std {
using namespace std;  // Only inside this private implementation adapter.
void* malloc(std::size_t size) noexcept {
  if (fault::mode == fault::Mode::kOom) return nullptr;
  void* value = std::malloc(size);
  if (value != nullptr) ++fault::allocations;
  return value;
}
void free(void* value) noexcept {
  if (value != nullptr) ++fault::releases;
  std::free(value);
}
}  // namespace fault_std

#if defined(_WIN32)
DWORD fault_undecorate(PCSTR, PSTR out, DWORD capacity, DWORD) {
  const int active = fault::active.fetch_add(1) + 1;
  int previous = fault::peak.load();
  while (active > previous &&
         !fault::peak.compare_exchange_weak(previous, active)) {
  }
  std::this_thread::yield();
  DWORD result = 0;
  if (fault::mode == fault::Mode::kSuccess && capacity > 0) {
    const std::size_t length =
        (std::min)(fault::kLength, static_cast<std::size_t>(capacity - 1));
    std::memset(out, 'x', length);
    out[length] = '\0';
    result = static_cast<DWORD>(length);
  }
  fault::active.fetch_sub(1);
  return result;
}
#define UnDecorateSymbolName fault_undecorate
#else
namespace __cxxabiv1 {
char* fault_cxa_demangle(const char*, char*, std::size_t*, int* status) {
  *status = fault::mode == fault::Mode::kError ? -2 : -1;
  if (fault::mode == fault::Mode::kError) return nullptr;
  auto* result = static_cast<char*>(fault_std::malloc(fault::kLength + 1));
  if (result != nullptr) {
    std::memset(result, 'x', fault::kLength);
    result[fault::kLength] = '\0';
    *status = 0;
  }
  return result;
}
}  // namespace __cxxabiv1
#define __cxa_demangle fault_cxa_demangle
#endif

namespace hyle::util::detail {
bool fault_demangle_visit(const char*, void*, DemangleSink) noexcept;
}
#define demangle fault_demangle_buffer
#define demangle_visit_internal fault_demangle_visit
#define std fault_std
#include "../../src/util_runtime/demangle.cc"
#undef std
#undef demangle_visit_internal
#undef demangle
#if defined(_WIN32)
#undef UnDecorateSymbolName
#else
#undef __cxa_demangle
#endif

TEST(DemangleFault, PlatformFailureAndAllocationFailureFallBack) {
  for (auto mode : {fault::Mode::kError, fault::Mode::kOom}) {
    fault::mode = mode;
    fault::allocations = fault::releases = 0;
    char output[32];
    EXPECT_EQ(
        hyle::util::fault_demangle_buffer("?original", output, sizeof(output)),
        9u);
    EXPECT_STREQ(output, "?original");
    EXPECT_EQ(fault::allocations, fault::releases);
  }
  fault::mode = fault::Mode::kSuccess;
}

TEST(DemangleFault, FailedConsumerSinkReleasesPlatformTemporary) {
  fault::mode = fault::Mode::kSuccess;
  fault::allocations = fault::releases = 0;
  bool called = false;
  EXPECT_FALSE(hyle::util::detail::fault_demangle_visit(
      "?original", &called,
      +[](void* state, const char* text, std::size_t size) noexcept {
        *static_cast<bool*>(state) = true;
        EXPECT_EQ(size, fault::kLength);
        EXPECT_EQ(text[0], 'x');
        return false;  // Same compiled path as a failed owning-string sink.
      }));
  EXPECT_TRUE(called);
  EXPECT_GT(fault::allocations, 0);
  EXPECT_EQ(fault::allocations, fault::releases);
}

TEST(DemangleFault, PublicOwningStringPropagatesAllocationFailure) {
  const std::string name(1500, 'x');
  bool caught = false;
  fault::fail_cpp_allocation = true;
  try {
    (void)hyle::util::demangle(name.c_str());
  } catch (const std::bad_alloc&) {
    caught = true;
  }
  fault::fail_cpp_allocation = false;
  EXPECT_TRUE(caught);
  EXPECT_EQ(hyle::util::demangle(name.c_str()), name);
}

#if defined(_WIN32)
TEST(DemangleFault, DbgHelpCallsWithinOneRuntimeImageAreSerialized) {
  fault::peak.store(0);
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([] {
      char output[8];
      for (int j = 0; j < 32; ++j) {
        EXPECT_EQ(hyle::util::fault_demangle_buffer("?original", output,
                                                    sizeof(output)),
                  fault::kLength);
      }
      EXPECT_EQ(fault::allocations, fault::releases);
    });
  }
  for (auto& worker : workers) worker.join();
  EXPECT_EQ(fault::peak.load(), 1);
}
#endif
