#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/stream.h"

// compute-01~04 CONTRACT death tests. Every HYLE_FATAL_ASSERT in the dispatch layer
// (compute.cc) encodes a CALLER contract (contract-over-defense: a caller bug aborts
// loudly, it is NOT a recoverable Status). These tests prove the contracts actually
// fire -- the error-path coverage the positive tests never reach.
//
// MECHANISM: HYLE_FATAL_ASSERT (macros.h:104) is NOT compiled out by NDEBUG -- it
// fprintf's + HYLE_TRAP()s in Debug AND Release, so EXPECT_DEATH catches it either way
// (same as util/registry_test.cc:116, sched/fiber_test.cc:114). We force the THREADSAFE
// death-test style: on a GPU machine a prior test may have initialized CUDA, and the
// default "fast" style fork()s that state into the child (fork-after-CUDA-init is
// unsafe). "threadsafe" re-execs a fresh binary -> no inherited CUDA/driver state.
//
// EACH scenario is a NAMED FUNCTION (not an inline EXPECT_DEATH block): the death
// statement contains commas (KernelArgs{...}) the preprocessor would mis-split, and a
// function body is the clean form (mirrors util/registry_test.cc:109's lambda). The
// function does all setup then triggers the assert; the threadsafe re-exec reruns it.
//
// The death matcher is "HYLE fatal" (not "") -- HYLE_FATAL_ASSERT prints
// "HYLE fatal: <cond>\n  at <file>:<line>" before trapping, so matching it PROVES the
// process died from the intended contract assert, NOT an incidental crash (null deref,
// a different abort). Verified once with the full "offset + size <= size_" text at
// compute.cc:322; "HYLE fatal" is the element-char-free stable substring.
//
// Two tiers:
//   - HOST-runnable (Buffer bounds, launch_batch cap): the host fallback hits the same
//     assert, so these run on ANY platform.
//   - CROSS-BACKEND gated (record/copy/launch across backends): need TWO devices on
//     DIFFERENT backends (e.g. CUDA GPU + host CPU). Skip when only one backend exists.

using namespace hyle;
using namespace hyle::compute;

// Force the threadsafe death-test style on Linux (re-exec, not fork -- a GPU machine has
// CUDA initialized, and fork-after-CUDA-init then calling CUDA in the child is unsafe).
// On Windows there is NO fork(): gtest death tests are ALWAYS spawn/re-exec, so the flag
// is unnecessary -- and referencing testing::FLAGS_gtest_death_test_style fails to link
// against the gtest DLL (shared-CRT build doesn't export it). So gate it to non-Windows.
#if defined(_WIN32)
#  define HYLE_THREADSAFE_DEATH() ((void)0)
#else
#  define HYLE_THREADSAFE_DEATH() GTEST_FLAG_SET(death_test_style, "threadsafe")
#endif

namespace {

// Find two devices on DIFFERENT backends (distinct vtable => the cross-backend/
// cross-device contracts can be violated). Returns false when only one backend is
// enumerated (host-only CI) -- callers then GTEST_SKIP.
bool find_two_backends(Device* a, Device* b) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    for (std::size_t j = 0; j < n && j < 8; ++j) {
      if (std::strcmp(devs[i].backend(), devs[j].backend()) != 0) {
        *a = devs[i];
        *b = devs[j];
        return true;
      }
    }
  }
  return false;
}

// A trivial host "kernel" (used only to mint a host-backend Kernel for the cross-backend
// launch test; it is never actually run there -- the launch aborts before dispatch).
void noop_host_kernel(const LaunchConfig&, const KernelArgs&) noexcept {}

// ---- the scenarios (each does setup + triggers exactly one contract assert) ----

void buffer_set_oob() {
  Device d = default_device();
  auto buf = Buffer::create(device_buffer_type(d), 256);
  std::uint8_t src[64] = {};
  (void)buf->set(224, src, 64);  // 224 + 64 = 288 > 256 -> fatal (compute.cc:322)
}

void buffer_get_oob() {
  Device d = default_device();
  auto buf = Buffer::create(device_buffer_type(d), 256);
  std::uint8_t dst[64] = {};
  (void)buf->get(224, dst, 64);  // compute.cc:328
}

void buffer_copy_dst_oob() {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  auto src = Buffer::create(bt, 256);
  auto dst = Buffer::create(bt, 128);
  (void)dst->copy_from(*src, 96, 0, 64);  // dst 96+64=160 > 128 (compute.cc:338)
}

void buffer_copy_src_oob() {
  Device d = default_device();
  BufferType bt = device_buffer_type(d);
  auto src = Buffer::create(bt, 128);
  auto dst = Buffer::create(bt, 256);
  (void)dst->copy_from(*src, 0, 96, 64);  // src 96+64=160 > 128 (compute.cc:338)
}

void buffer_clear_oob() {
  Device d = default_device();
  auto buf = Buffer::create(device_buffer_type(d), 256);
  (void)buf->clear(224, 0, 64);  // compute.cc:345
}

void buffer_flush_host_oob() {
  Device d = default_device();
  auto buf = Buffer::create(host_buffer_type(d), 256);
  (void)buf->flush_host(224, 64);
}

void buffer_invalidate_host_oob() {
  Device d = default_device();
  auto buf = Buffer::create(host_buffer_type(d), 256);
  (void)buf->invalidate_host(224, 64);
}

// Integer-wraparound OOB: offset + size overflows size_t back to a SMALL value. The old
// `offset + size <= size_` check passed (SIZE_MAX-3 + 10 wraps to 6 <= 256), letting an
// out-of-range write through to the backend memcpy. The wrap-safe check (offset <= size_
// && size <= size_ - offset) must catch it and fatal-abort.
void buffer_set_wraparound() {
  Device d = default_device();
  auto buf = Buffer::create(device_buffer_type(d), 256);
  std::uint8_t src[8] = {};
  (void)buf->set(static_cast<std::size_t>(-3), src, 10);  // -3 + 10 wraps to 7
}

void launch_batch_over_cap() {
  Device d = default_device();
  auto s = Stream::create(d);
  BatchItem items[1] = {};            // non-null; the count check fires first
  (void)launch_batch(*s, items, 65);  // 65 > 64 -> fatal (compute.cc:588)
}

void launch_batch_null_items() {
  Device d = default_device();
  auto s = Stream::create(d);
  (void)launch_batch(*s, nullptr, 1);  // count!=0 && items==null (compute.cc:588)
}

void record_cross_backend() {
  Device da, db;
  find_two_backends(&da, &db);
  auto s = Stream::create(da);   // stream on backend A
  auto ev = Event::create(db);   // event on backend B
  s->record(*ev);                // vt_/local_ mismatch -> fatal (compute.cc:169)
}

void copy_cross_backend() {
  Device da, db;
  find_two_backends(&da, &db);
  auto src = Buffer::create(device_buffer_type(da), 256);
  auto dst = Buffer::create(device_buffer_type(db), 256);
  (void)dst->copy_from(*src, 0, 0, 256);  // vt_ != src.vt_ -> fatal (compute.cc:337)
}

void launch_cross_backend() {
  // host device (load_host works only there) + a different backend for the stream.
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  Device host_dev = default_device();
  Device other;
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].type() == DeviceType::kCpu) host_dev = devs[i];
  }
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (std::strcmp(devs[i].backend(), host_dev.backend()) != 0) other = devs[i];
  }
  auto k = Kernel::load_host(host_dev, &noop_host_kernel);  // host-backend kernel
  auto s = Stream::create(other);                           // other-backend stream
  LaunchConfig cfg;
  KernelArgs empty{nullptr, 0, nullptr, 0, nullptr, 0};
  (void)launch(*s, *k, cfg, empty);  // stream.vt_ != kernel.vt_ -> fatal (compute.cc:580)
}

}  // namespace

// ---- Buffer bounds (host-runnable): offset+size must be <= size() ------------

TEST(ComputeContractDeathTest, BufferSetOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_set_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferGetOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_get_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferCopyFromDstOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_copy_dst_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferCopyFromSrcOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_copy_src_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferClearOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_clear_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferFlushHostOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_flush_host_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferInvalidateHostOutOfBounds) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_invalidate_host_oob(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, BufferSetWraparound) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(buffer_set_wraparound(), "HYLE fatal");
}

// ---- launch_batch cap (host-runnable): count <= 64, items non-null -----------

TEST(ComputeContractDeathTest, LaunchBatchCountExceedsCap) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_batch_over_cap(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, LaunchBatchNullItems) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_batch_null_items(), "HYLE fatal");
}

// ---- cross-backend / cross-device (gated: need two backends) -----------------

TEST(ComputeContractDeathTest, RecordEventCrossBackend) {
  Device a, b;
  if (!find_two_backends(&a, &b)) {
    GTEST_SKIP() << "only one backend enumerated -- cross-backend record unreachable";
  }
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(record_cross_backend(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, CopyFromCrossBackend) {
  Device a, b;
  if (!find_two_backends(&a, &b)) {
    GTEST_SKIP() << "only one backend enumerated -- cross-backend copy unreachable";
  }
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(copy_cross_backend(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, LaunchCrossBackend) {
  Device a, b;
  if (!find_two_backends(&a, &b)) {
    GTEST_SKIP() << "only one backend enumerated -- cross-backend launch unreachable";
  }
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_cross_backend(), "HYLE fatal");
}
