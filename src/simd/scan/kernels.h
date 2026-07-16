#pragma once

#include <cstddef>
#include <cstdint>

// Internal per-ISA scan kernel symbols (suffix = scalar|sse4|avx2|avx512|neon).

namespace hyle::simd::detail {

#define HYLE_SCAN_DECLS(sfx)                                                                   \
  void inclusive_scan_u32_##sfx(std::uint32_t* data, std::size_t n) noexcept;                  \
  void inclusive_scan_i32_##sfx(std::int32_t* data, std::size_t n) noexcept;                   \
  void inclusive_scan_u64_##sfx(std::uint64_t* data, std::size_t n) noexcept;                  \
  void inclusive_scan_i64_##sfx(std::int64_t* data, std::size_t n) noexcept;                   \
  void inclusive_scan_f32_##sfx(float* data, std::size_t n) noexcept;                          \
  void inclusive_scan_f64_##sfx(double* data, std::size_t n) noexcept;                         \
  void exclusive_scan_u32_##sfx(std::uint32_t* data, std::size_t n) noexcept;                  \
  void exclusive_scan_i32_##sfx(std::int32_t* data, std::size_t n) noexcept;                   \
  void exclusive_scan_u64_##sfx(std::uint64_t* data, std::size_t n) noexcept;                  \
  void exclusive_scan_i64_##sfx(std::int64_t* data, std::size_t n) noexcept;                   \
  void exclusive_scan_f32_##sfx(float* data, std::size_t n) noexcept;                          \
  void exclusive_scan_f64_##sfx(double* data, std::size_t n) noexcept

HYLE_SCAN_DECLS(scalar);
HYLE_SCAN_DECLS(sse4);
HYLE_SCAN_DECLS(avx2);
HYLE_SCAN_DECLS(avx512);
HYLE_SCAN_DECLS(neon);
HYLE_SCAN_DECLS(sve);
HYLE_SCAN_DECLS(rvv);

#undef HYLE_SCAN_DECLS

}  // namespace hyle::simd::detail
