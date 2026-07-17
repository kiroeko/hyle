// LoadU / StoreU partial-shape boundary guardrails for fixed-width backends.
// StoreU must not write past N*sizeof(T), and LoadU must not read past it.
// The load cases place the logical range at the end of a readable page so an
// accidental full-register access faults in the following guard page.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Guard pages via the platform VM primitives (platform-02b) -- one portable path
// instead of per-OS VirtualAlloc/mmap. reserve() gives a raw reservation; we
// commit the low page RW and protect the high page as kNone so an over-read
// faults there.
#include "hyle/platform/virtual_memory.h"
#define HYLE_SIMD_TEST_HAS_GUARD_PAGES 1

#include "hyle/simd/vec.h"

namespace hs = hyle::simd;

namespace {

// Build a partial vector without LoadU/StoreU (the operations under test). Set does
// not touch memory and preserves the logical N in the vector type.
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

// StoreU tail guard: bytes after N*sizeof(T) stay sentinel 0xA5. The padding is one
// full native register, enough to catch an accidental XMM/YMM/ZMM/Q-width store.
template <typename T, std::size_t N>
void CheckStoreUNoWritePastLogical() {
  static_assert(N >= 1 && (N & (N - 1)) == 0, "N power of two");
  static_assert(N * sizeof(T) < hs::kNativeVectorBytes, "partial shapes only");

  constexpr std::size_t kLogical = N * sizeof(T);
  constexpr std::size_t kPad = hs::kNativeVectorBytes;
  alignas(64) unsigned char buf[kLogical + kPad];
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

#if HYLE_SIMD_TEST_HAS_GUARD_PAGES
// LoadU read boundary: place exactly N*sizeof(T) bytes at the end of a RW page;
// the next page is PROT_NONE.
template <typename T, std::size_t N>
void CheckLoadUPageEndNoFault() {
  static_assert(N * sizeof(T) < hs::kNativeVectorBytes, "partial shapes only");
  constexpr std::size_t kLogical = N * sizeof(T);

  namespace pf = hyle::platform;
  const std::size_t page = pf::page_size();
  ASSERT_GT(page, 0u);

  // Reserve two pages: [0,page) becomes RW (holds the data), [page,2*page) stays
  // no-access so an over-read past the logical range faults in it.
  pf::VmRegion region = pf::VmRegion::reserve(page * 2);
  ASSERT_TRUE(region.valid()) << pf::vm_last_error();
  ASSERT_TRUE(region.commit(0, page)) << pf::vm_last_error();
  ASSERT_TRUE(region.protect(page, page, pf::Protect::kNone)) << pf::vm_last_error();
  auto* base = static_cast<unsigned char*>(region.data());

  // Exactly kLogical bytes ending at the first-page boundary.
  auto* p = reinterpret_cast<T*>(base + page - kLogical);
  const T value = PatternValue<T>();
  for (std::size_t i = 0; i < N; ++i) {
    std::memcpy(reinterpret_cast<unsigned char*>(p) + i * sizeof(T), &value, sizeof(T));
  }

  const auto d = hs::FixedTag<T, N>{};
  const auto v = hs::LoadU(d, p);

  // Read back through the same logical tag into a full-native-width scratch.
  alignas(64) unsigned char out[hs::kNativeVectorBytes] = {};
  hs::StoreU(v, d, reinterpret_cast<T*>(out));
  for (std::size_t i = 0; i < N; ++i) {
    T got{};
    std::memcpy(&got, out + i * sizeof(T), sizeof(T));
    EXPECT_EQ(got, value) << "LoadU page-end lane " << i << " sizeof(T)=" << sizeof(T)
                           << " N=" << N;
  }
  // region releases both pages on scope exit (RAII).
}
#endif

}  // namespace

// ---- StoreU sentinel cases --------------------------------------------------------

#define HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(T, N, Name) \
  TEST(PartialBoundary, StoreU_NoWritePast_##Name) { CheckStoreUNoWritePastLogical<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 1, i8x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 2, i8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 4, i8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 8, i8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 1, u8x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 2, u8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 4, u8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 8, u8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int16_t, 1, i16x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int16_t, 2, i16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int16_t, 4, i16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint16_t, 1, u16x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint16_t, 2, u16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint16_t, 4, u16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int32_t, 1, i32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int32_t, 2, i32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint32_t, 1, u32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint32_t, 2, u32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int64_t, 1, i64x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint64_t, 1, u64x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(float, 1, f32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(float, 2, f32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(double, 1, f64x1)

#if defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_TARGET_AVX512)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 16, i8x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 16, u8x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int16_t, 8, i16x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint16_t, 8, u16x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int32_t, 4, i32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint32_t, 4, u32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int64_t, 2, i64x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint64_t, 2, u64x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(float, 4, f32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(double, 2, f64x2)
#endif

#if defined(HYLE_SIMD_TARGET_AVX512)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int8_t, 32, i8x32)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint8_t, 32, u8x32)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int16_t, 16, i16x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint16_t, 16, u16x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int32_t, 8, i32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint32_t, 8, u32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::int64_t, 4, i64x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(std::uint64_t, 4, u64x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(float, 8, f32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY(double, 4, f64x4)
#endif

#undef HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_STORE_BOUNDARY

// ---- LoadU page-end cases --------------------------------------------------------

#if HYLE_SIMD_TEST_HAS_GUARD_PAGES

#define HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(T, N, Name) \
  TEST(PartialBoundary, LoadU_PageEnd_##Name) { CheckLoadUPageEndNoFault<T, N>(); }

HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 1, i8x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 2, i8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 4, i8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 8, i8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 1, u8x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 2, u8x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 4, u8x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 8, u8x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int16_t, 1, i16x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int16_t, 2, i16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int16_t, 4, i16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint16_t, 1, u16x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint16_t, 2, u16x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint16_t, 4, u16x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int32_t, 1, i32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int32_t, 2, i32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint32_t, 1, u32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint32_t, 2, u32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int64_t, 1, i64x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint64_t, 1, u64x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(float, 1, f32x1)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(float, 2, f32x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(double, 1, f64x1)

#if defined(HYLE_SIMD_TARGET_AVX2) || defined(HYLE_SIMD_TARGET_AVX512)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 16, i8x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 16, u8x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int16_t, 8, i16x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint16_t, 8, u16x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int32_t, 4, i32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint32_t, 4, u32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int64_t, 2, i64x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint64_t, 2, u64x2)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(float, 4, f32x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(double, 2, f64x2)
#endif

#if defined(HYLE_SIMD_TARGET_AVX512)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int8_t, 32, i8x32)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint8_t, 32, u8x32)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int16_t, 16, i16x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint16_t, 16, u16x16)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int32_t, 8, i32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint32_t, 8, u32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::int64_t, 4, i64x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(std::uint64_t, 4, u64x4)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(float, 8, f32x8)
HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY(double, 4, f64x4)
#endif

#undef HYLE_SIMD_TEST_INSTANTIATE_PARTIAL_LOAD_BOUNDARY

#endif  // HYLE_SIMD_TEST_HAS_GUARD_PAGES

#undef HYLE_SIMD_TEST_HAS_GUARD_PAGES
