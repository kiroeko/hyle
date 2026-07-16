// LoadU / StoreU partial-shape boundary guardrails for the NEON backend.
// StoreU must not write past N*sizeof(T), and LoadU must not read past it.
// The load cases place the logical range at the end of a readable page so an
// accidental full-register access faults in the following guard page.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#define HYLE_TEST_HAS_GUARD_PAGES 1
#else
#define HYLE_TEST_HAS_GUARD_PAGES 0
#endif

#include "hyle/simd/vec.h"

namespace hs = hyle::simd;

namespace {

// Build a partial vector without LoadU/StoreU (those are under test and currently
// full-register). Set is C1 D-form and does not touch memory.
template <typename T, std::size_t N>
auto MakePartialSet(T value) {
  return hs::Set(hs::FixedTag<T, N>{}, value);
}

template <typename T>
T PatternValue() {
  if constexpr (std::is_same_v<T, float>) return 3.25f;
  else if constexpr (std::is_same_v<T, double>) return 7.5;
  else if constexpr (std::is_signed_v<T>) return static_cast<T>(-0x55);
  else return static_cast<T>(0xA3);
}

// StoreU tail guard: bytes [N*sizeof(T), N*sizeof(T)+16) stay sentinel 0xA5.
template <typename T, std::size_t N>
void CheckStoreUNoWritePastLogical() {
  static_assert(N >= 1 && (N & (N - 1)) == 0, "N power of two");
  static_assert(N * sizeof(T) <= 8, "partial / D-band shapes only");

  constexpr std::size_t kLogical = N * sizeof(T);
  constexpr std::size_t kPad = 16;  // one full Q spill past the logical width
  alignas(16) unsigned char buf[kLogical + kPad];
  std::memset(buf, 0xA5, sizeof(buf));

  const T value = PatternValue<T>();
  const auto d = hs::FixedTag<T, N>{};
  const auto v = MakePartialSet<T, N>(value);
  hs::StoreU(v, d, reinterpret_cast<T*>(buf));

  // Logical lanes must be written.
  for (std::size_t i = 0; i < N; ++i) {
    T got{};
    std::memcpy(&got, buf + i * sizeof(T), sizeof(T));
    EXPECT_EQ(got, value) << "logical lane " << i << " sizeof(T)=" << sizeof(T)
                           << " N=" << N;
  }

  // Pad past logical width must be untouched.
  for (std::size_t off = kLogical; off < kLogical + kPad; ++off) {
    EXPECT_EQ(buf[off], static_cast<unsigned char>(0xA5))
        << "StoreU OOB write at +" << (off - kLogical) << " past logical "
        << kLogical << " B (sizeof(T)=" << sizeof(T) << " N=" << N << ")";
  }
}

#if HYLE_TEST_HAS_GUARD_PAGES
// LoadU read boundary: place exactly N*sizeof(T) bytes at the end of a RW page;
// the next page is PROT_NONE.
template <typename T, std::size_t N>
void CheckLoadUPageEndNoFault() {
  static_assert(N * sizeof(T) <= 8, "partial / D-band shapes only");
  constexpr std::size_t kLogical = N * sizeof(T);

  const long page_l = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_l, 0);
  const std::size_t page = static_cast<std::size_t>(page_l);

  void* mem = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mem, MAP_FAILED);
  auto* base = static_cast<unsigned char*>(mem);
  ASSERT_EQ(mprotect(base + page, page, PROT_NONE), 0);

  // Exactly kLogical bytes ending at the first-page boundary.
  auto* p = reinterpret_cast<T*>(base + page - kLogical);
  const T value = PatternValue<T>();
  for (std::size_t i = 0; i < N; ++i) {
    std::memcpy(reinterpret_cast<unsigned char*>(p) + i * sizeof(T), &value, sizeof(T));
  }

  const auto d = hs::FixedTag<T, N>{};
  const auto v = hs::LoadU(d, p);

  // Read back via StoreU into a full-register-sized scratch (safe width for both
  // pre-C4 wide stores and post-C4 narrow stores).
  alignas(16) unsigned char out[16] = {};
  hs::StoreU(v, d, reinterpret_cast<T*>(out));
  for (std::size_t i = 0; i < N; ++i) {
    T got{};
    std::memcpy(&got, out + i * sizeof(T), sizeof(T));
    EXPECT_EQ(got, value) << "LoadU page-end lane " << i << " sizeof(T)=" << sizeof(T)
                           << " N=" << N;
  }

  munmap(mem, page * 2);
}
#endif

}  // namespace

// ---- StoreU sentinel cases --------------------------------------------------------

#define HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(T, N, Name) \
  TEST(PartialBoundary, StoreU_NoWritePast_##Name) { CheckStoreUNoWritePastLogical<T, N>(); }

HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int8_t, 1, i8x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int8_t, 2, i8x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int8_t, 4, i8x4)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int8_t, 8, i8x8)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint8_t, 1, u8x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint8_t, 2, u8x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint8_t, 4, u8x4)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint8_t, 8, u8x8)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int16_t, 1, i16x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int16_t, 2, i16x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int16_t, 4, i16x4)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint16_t, 1, u16x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint16_t, 2, u16x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint16_t, 4, u16x4)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int32_t, 1, i32x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int32_t, 2, i32x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint32_t, 1, u32x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint32_t, 2, u32x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::int64_t, 1, i64x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(std::uint64_t, 1, u64x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(float, 1, f32x1)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(float, 2, f32x2)
HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST(double, 1, f64x1)

#undef HYLE_INSTANTIATE_PARTIAL_STORE_BOUNDARY_TEST

// ---- LoadU page-end cases --------------------------------------------------------

#if HYLE_TEST_HAS_GUARD_PAGES

#define HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(T, N, Name) \
  TEST(PartialBoundary, LoadU_PageEnd_##Name) { CheckLoadUPageEndNoFault<T, N>(); }

HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int8_t, 1, i8x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int8_t, 2, i8x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int8_t, 4, i8x4)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int8_t, 8, i8x8)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint8_t, 1, u8x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint8_t, 2, u8x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint8_t, 4, u8x4)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint8_t, 8, u8x8)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int16_t, 1, i16x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int16_t, 2, i16x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int16_t, 4, i16x4)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint16_t, 1, u16x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint16_t, 2, u16x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint16_t, 4, u16x4)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int32_t, 1, i32x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int32_t, 2, i32x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint32_t, 1, u32x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint32_t, 2, u32x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::int64_t, 1, i64x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(std::uint64_t, 1, u64x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(float, 1, f32x1)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(float, 2, f32x2)
HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST(double, 1, f64x1)

#undef HYLE_INSTANTIATE_PARTIAL_LOAD_BOUNDARY_TEST

#endif  // HYLE_TEST_HAS_GUARD_PAGES
