#include "hyle/platform/virtual_memory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>  // std::abort in death tests
#include <cstring>
#include <limits>
#include <utility>

using hyle::platform::large_page_align_up;
using hyle::platform::large_page_size;
using hyle::platform::page_align_up;
using hyle::platform::page_size;
using hyle::platform::Protect;
using hyle::platform::vm_advise_hugepage;
using hyle::platform::vm_advise_no_hugepage;
using hyle::platform::vm_commit;
using hyle::platform::vm_decommit;
using hyle::platform::vm_protect;
using hyle::platform::vm_release;
using hyle::platform::vm_reserve;
using hyle::platform::VmRegion;

// cstring already included for memset in new tests

TEST(VirtualMemory, PageSizeIsSane) {
  std::size_t p = page_size();
  EXPECT_GE(p, 4096u);
  EXPECT_EQ(p & (p - 1), 0u);  // power of two
  EXPECT_EQ(page_align_up(0), 0u);
  EXPECT_EQ(page_align_up(1), p);
  EXPECT_EQ(page_align_up(p), p);
  EXPECT_EQ(page_align_up(p + 1), 2 * p);
}

TEST(VirtualMemory, ReserveCommitWriteRelease) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 2).value();
  ASSERT_TRUE(vm_commit(base, p * 2).ok());

  auto* bytes = static_cast<unsigned char*>(base);
  for (std::size_t i = 0; i < p * 2; ++i) bytes[i] = static_cast<unsigned char>(i & 0xff);
  for (std::size_t i = 0; i < p * 2; ++i) {
    EXPECT_EQ(bytes[i], static_cast<unsigned char>(i & 0xff));
  }

  ASSERT_TRUE(vm_release(base, p * 2).ok());
}

TEST(VirtualMemory, ProtectNoneThenRestore) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 2).value();
  ASSERT_TRUE(vm_commit(base, p * 2).ok());

  // Lock the high page; low page stays RW so we still have a writable area.
  auto* bytes = static_cast<unsigned char*>(base);
  ASSERT_TRUE(vm_protect(bytes + p, p, Protect::kNone).ok());

  // Low page still writable.
  bytes[0] = 0xAB;
  EXPECT_EQ(bytes[0], 0xAB);

  // Restore high page and write through it.
  ASSERT_TRUE(vm_protect(bytes + p, p, Protect::kReadWrite).ok());
  bytes[p] = 0xCD;
  EXPECT_EQ(bytes[p], 0xCD);

  ASSERT_TRUE(vm_release(base, p * 2).ok());
}

#if defined(_WIN32)
TEST(VirtualMemory, ProtectNoneDoesNotHideMixedCommitFailure) {
  const std::size_t p = page_size();
  void* base = vm_reserve(p * 2).value();
  auto* bytes = static_cast<unsigned char*>(base);
  ASSERT_TRUE(vm_commit(bytes + p, p).ok());

  // VirtualProtect rejects a span mixing reserved and committed pages. The
  // reserved first page must not make the wrapper report a false success.
  EXPECT_FALSE(vm_protect(base, p * 2, Protect::kNone).ok());
  bytes[p] = 0xA5;
  EXPECT_EQ(bytes[p], 0xA5);
  ASSERT_TRUE(vm_release(base, p * 2).ok());
}
#endif

TEST(VirtualMemory, DecommitAndRecommit) {
  std::size_t p = page_size();
  void* base = vm_reserve(p).value();
  ASSERT_TRUE(vm_commit(base, p).ok());
  std::memset(base, 0x11, p);

  ASSERT_TRUE(vm_decommit(base, p).ok());
  ASSERT_TRUE(vm_commit(base, p).ok());
  // After recommit contents are undefined; just prove we can write again.
  std::memset(base, 0x22, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x22);

  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(VirtualMemory, SizeRoundsUp) {
  std::size_t p = page_size();
  // Request 1 byte -> one page reservation.
  void* base = vm_reserve(1).value();
  ASSERT_TRUE(vm_commit(base, 1).ok());
  static_cast<unsigned char*>(base)[0] = 7;
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 7);
  // Release with the rounded size callers must use after reserve.
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(VirtualMemory, VmRegionRAII) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve(p * 3).value();
  EXPECT_EQ(r.size(), p * 3);
  EXPECT_FALSE(r.has_guard());
  ASSERT_TRUE(r.commit_usable().ok());
  std::memset(r.data(), 0x5A, r.size());
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x5A);

  VmRegion moved = std::move(r);
  EXPECT_FALSE(r.valid());
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(static_cast<unsigned char*>(moved.data())[1], 0x5A);
  // moved dtor releases
}

TEST(VirtualMemory, GuardedLayout) {
  std::size_t p = page_size();
  std::size_t usable_req = p * 2 + 17;  // forces round-up
  VmRegion r = VmRegion::reserve_guarded(usable_req).value();
  EXPECT_TRUE(r.has_guard());
  EXPECT_EQ(r.size(), p + page_align_up(usable_req));
  EXPECT_EQ(r.usable_size(), page_align_up(usable_req));
  EXPECT_EQ(static_cast<std::byte*>(r.usable()),
            static_cast<std::byte*>(r.data()) + p);

  // Usable range is writable end-to-end.
  auto* u = static_cast<unsigned char*>(r.usable());
  for (std::size_t i = 0; i < r.usable_size(); ++i) u[i] = static_cast<unsigned char>(i);
  EXPECT_EQ(u[0], 0);
  EXPECT_EQ(u[r.usable_size() - 1],
            static_cast<unsigned char>((r.usable_size() - 1) & 0xff));

  // Guard page is the first page -- we do not touch it (would SIGSEGV/AV).
  // Re-protect usable as read-only and verify write path is still mapped.
  ASSERT_TRUE(r.protect(p, r.usable_size(), Protect::kRead).ok());
  EXPECT_EQ(u[0], 0);
  ASSERT_TRUE(r.protect(p, r.usable_size(), Protect::kReadWrite).ok());
  u[0] = 0xEE;
  EXPECT_EQ(u[0], 0xEE);
}

TEST(VirtualMemory, OutOfRangeCommitFails) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve(p).value();
  ASSERT_TRUE(r.valid());
  EXPECT_FALSE(r.commit(p, p).ok());  // offset past end
  EXPECT_FALSE(r.commit(0, p * 2).ok());  // length past end
}

TEST(VirtualMemory, ReleaseNullIsOk) {
  EXPECT_TRUE(vm_release(nullptr, page_size()).ok());
  VmRegion empty;
  EXPECT_FALSE(empty.valid());
  EXPECT_TRUE(empty.release().ok());  // no-op
}

TEST(VirtualMemory, ReserveZeroFails) {
  auto result = vm_reserve(0);
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsInvalidArgument());
}

TEST(VirtualMemory, CommitUsableOnPlainReserve) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve(p * 2).value();
  ASSERT_TRUE(r.valid());
  EXPECT_FALSE(r.has_guard());
  ASSERT_TRUE(r.commit_usable().ok());
  std::memset(r.data(), 0x3C, r.size());
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x3C);
}

TEST(VirtualMemory, ProtectReadOnlyAllowsRead) {
  std::size_t p = page_size();
  void* base = vm_reserve(p).value();
  ASSERT_TRUE(vm_commit(base, p).ok());
  static_cast<unsigned char*>(base)[0] = 0x11;
  ASSERT_TRUE(vm_protect(base, p, Protect::kRead).ok());
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x11);  // readable
  // restore before free
  ASSERT_TRUE(vm_protect(base, p, Protect::kReadWrite).ok());
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(VirtualMemory, LargePageSizeIsSaneOrZero) {
  std::size_t lp = large_page_size();
  // 0 = unknown platform; else power-of-two and >> base page.
  if (lp != 0) {
    EXPECT_EQ(lp & (lp - 1), 0u);
    EXPECT_GE(lp, page_size());
    EXPECT_EQ(large_page_align_up(0), 0u);
    EXPECT_EQ(large_page_align_up(1), lp);
    EXPECT_EQ(large_page_align_up(lp), lp);
    EXPECT_EQ(large_page_align_up(lp + 1), 2 * lp);
  } else {
    // Fallback align uses base page.
    EXPECT_EQ(large_page_align_up(1), page_size());
  }
}

TEST(VirtualMemory, AdviseHugepageSoft) {
  std::size_t p = page_size();
  std::size_t n = p * 16;  // a few pages; THP may or may not promote
  void* base = vm_reserve(n).value();
  ASSERT_TRUE(vm_commit(base, n).ok());
  // Soft contract: true on Linux (even if THP=never, madvise may still succeed),
  // false on Windows/macOS. Must not crash either way; range stays writable.
  (void)vm_advise_hugepage(base, n);
  (void)vm_advise_no_hugepage(base, n);
  static_cast<unsigned char*>(base)[0] = 0x42;
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x42);
  EXPECT_TRUE(vm_advise_hugepage(nullptr, 0).ok());
  EXPECT_TRUE(vm_advise_no_hugepage(nullptr, 0).ok());
  ASSERT_TRUE(vm_release(base, n).ok());
}

TEST(VirtualMemory, DecommitOutOfRangeFails) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve(p).value();
  ASSERT_TRUE(r.valid());
  EXPECT_FALSE(r.decommit(p, p).ok());
  EXPECT_FALSE(r.protect(p, p, Protect::kNone).ok());
}

TEST(VirtualMemory, MoveAssignReleasesPrevious) {
  std::size_t p = page_size();
  VmRegion a = VmRegion::reserve(p).value();
  VmRegion b = VmRegion::reserve(p * 2).value();
  ASSERT_TRUE(a.valid());
  ASSERT_TRUE(b.valid());
  void* b_data = b.data();
  a = std::move(b);
  EXPECT_FALSE(b.valid());
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.data(), b_data);
  EXPECT_EQ(a.size(), p * 2);
}

TEST(VirtualMemory, GuardedCommitUsableIdempotent) {
  VmRegion r = VmRegion::reserve_guarded(page_size()).value();
  ASSERT_TRUE(r.valid());
  // usable already committed by reserve_guarded; commit_usable is a re-commit.
  ASSERT_TRUE(r.commit_usable().ok());
  auto* u = static_cast<unsigned char*>(r.usable());
  u[0] = 0x55;
  u[r.usable_size() - 1] = 0xAA;
  EXPECT_EQ(u[0], 0x55);
  EXPECT_EQ(u[r.usable_size() - 1], 0xAA);
}

TEST(VirtualMemory, EmptyRangeIsNoOpButNullNonemptyFails) {
  EXPECT_TRUE(vm_commit(nullptr, 0).ok());
  EXPECT_TRUE(vm_decommit(nullptr, 0).ok());
  EXPECT_TRUE(vm_protect(nullptr, 0, Protect::kNone).ok());
  EXPECT_TRUE(vm_commit(nullptr, page_size()).IsInvalidArgument());
  EXPECT_TRUE(vm_decommit(nullptr, page_size()).IsInvalidArgument());
  EXPECT_TRUE(
      vm_protect(nullptr, page_size(), Protect::kNone).IsInvalidArgument());
}

TEST(VirtualMemory, SizeRoundingOverflowFails) {
  void* raw = vm_reserve(page_size()).value();
  EXPECT_TRUE(vm_commit(raw, std::numeric_limits<std::size_t>::max())
                  .IsOutOfRange());
  EXPECT_TRUE(vm_release(raw, page_size()).ok());
}

// ---- VmRegion::adopt (C++ side) --------------------------------------------
// adopt() wraps an existing raw vm_reserve* reservation in a managed region that
// releases it on destruction. Only the C ABI exercised this before; these cover
// the C++ handle directly: take-ownership + use, the NULL guard, move transfer,
// and that release()/destruction free the adopted block exactly once.

TEST(VirtualMemory, AdoptTakesOwnershipAndReleasesOnDtor) {
  std::size_t p = page_size();
  void* raw = vm_reserve(p * 2).value();
  {
    VmRegion r = VmRegion::adopt(raw, p * 2).value();
    ASSERT_TRUE(r.valid());
    EXPECT_EQ(r.data(), raw);   // wraps the exact reservation, no copy/move of pages
    EXPECT_EQ(r.size(), p * 2);
    EXPECT_FALSE(r.has_guard());
    // Behaves like any region: commit + write through it.
    ASSERT_TRUE(r.commit_usable().ok());
    std::memset(r.data(), 0x77, r.size());
    EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x77);
  }  // r's dtor releases the adopted reservation (no manual vm_release needed)
}

TEST(VirtualMemory, AdoptNullIsInvalid) {
  auto r = VmRegion::adopt(nullptr, page_size());
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

TEST(VirtualMemory, AdoptRejectsMisalignedBaseWithoutTakingOwnership) {
  const std::size_t p = page_size();
  void* raw = vm_reserve(p * 2).value();
  auto* misaligned = static_cast<unsigned char*>(raw) + 1;
  auto r = VmRegion::adopt(misaligned, p);
  ASSERT_FALSE(r);
  EXPECT_TRUE(r.error().IsInvalidArgument());
  EXPECT_TRUE(vm_release(raw, p * 2).ok());
}

TEST(VirtualMemory, AdoptMoveTransfersOwnership) {
  std::size_t p = page_size();
  void* raw = vm_reserve(p).value();
  VmRegion a = VmRegion::adopt(raw, p).value();
  VmRegion b = std::move(a);
  EXPECT_FALSE(a.valid());       // moved-from is empty; its dtor is a no-op
  EXPECT_EQ(a.data(), nullptr);
  ASSERT_TRUE(b.valid());
  EXPECT_EQ(b.data(), raw);
  EXPECT_EQ(b.size(), p);
  // Only b owns the reservation now -- exactly one release happens (b's dtor).
}

TEST(VirtualMemory, AdoptReleaseThenDoubleReleaseIsSafe) {
  std::size_t p = page_size();
  void* raw = vm_reserve(p).value();
  VmRegion r = VmRegion::adopt(raw, p).value();
  EXPECT_TRUE(r.release().ok());  // frees the adopted reservation
  EXPECT_FALSE(r.valid());   // now empty
  EXPECT_TRUE(r.release().ok());  // second release is a safe no-op
  EXPECT_FALSE(r.valid());
  // r's dtor is also a no-op now (state is already empty).
}

// ---- death tests: prove the PROTECTION actually faults (not just "set ok") --
// These were the coverage gap: earlier tests set kNone / kRead / a guard page
// and checked the call returned true, but never confirmed an illegal access
// truly traps. Death tests run in a forked child, so the SIGSEGV/AV is captured
// as expected death rather than crashing the suite. ".*" matcher: a hardware
// fault produces no deterministic stderr.

TEST(VirtualMemoryDeathTest, GuardPageWriteFaults) {
  EXPECT_DEATH(
      {
        auto result = VmRegion::reserve_guarded(page_size());
        if (!result) std::abort();
        VmRegion r = *std::move(result);
        // data() is the permanent no-access guard page (below usable()).
        volatile auto* g = static_cast<volatile unsigned char*>(r.data());
        g[0] = 0x1;  // must trap
      },
      ".*");
}

TEST(VirtualMemoryDeathTest, ProtectNoneWriteFaults) {
  EXPECT_DEATH(
      {
        std::size_t p = page_size();
        auto reserved = vm_reserve(p);
        if (!reserved) std::abort();
        void* base = *reserved;
        if (!vm_commit(base, p).ok()) std::abort();
        if (!vm_protect(base, p, Protect::kNone).ok()) std::abort();
        volatile auto* b = static_cast<volatile unsigned char*>(base);
        b[0] = 0x1;  // committed then locked to no-access -> trap
      },
      ".*");
}

TEST(VirtualMemoryDeathTest, ProtectReadOnlyWriteFaults) {
  EXPECT_DEATH(
      {
        std::size_t p = page_size();
        auto reserved = vm_reserve(p);
        if (!reserved) std::abort();
        void* base = *reserved;
        if (!vm_commit(base, p).ok()) std::abort();
        if (!vm_protect(base, p, Protect::kRead).ok()) std::abort();
        volatile auto* b = static_cast<volatile unsigned char*>(base);
        (void)b[0];    // read is allowed
        b[0] = 0x1;    // write to read-only page -> trap
      },
      ".*");
}
