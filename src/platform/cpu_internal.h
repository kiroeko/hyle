#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "hyle/platform/cpu.h"

namespace hyle::platform::detail {

HYLE_HIDDEN constexpr bool feature_bit(std::uint32_t value,
                                       unsigned index) noexcept {
  return (value & (std::uint32_t{1} << index)) != 0;
}

HYLE_HIDDEN constexpr void set_if(CpuFeatureSet& features, CpuFeature feature,
                                  bool present) noexcept {
  if (present) features.set(feature);
}

// Pure input for x86 CPUID/XCR0 decoding. Unsupported leaves/subleaves are
// represented by zero registers, which is exactly what the baseline probe does.
struct X86FeatureRegisters final {
  std::uint32_t leaf1_ecx = 0;
  std::uint32_t leaf7_0_ebx = 0;
  std::uint32_t leaf7_0_ecx = 0;
  std::uint32_t leaf7_0_edx = 0;
  std::uint32_t leaf7_1_eax = 0;
  std::uint32_t leaf7_1_edx = 0;
  std::uint32_t extended1_ecx = 0;
  std::uint32_t leaf24_ebx = 0;
  std::uint32_t xcr0 = 0;
};

HYLE_HIDDEN constexpr CpuFeatureSet decode_x86_features(
    const X86FeatureRegisters& r) noexcept {
  CpuFeatureSet out{CpuFeature::kX86Sse2};  // x86-64 architectural baseline

  set_if(out, CpuFeature::kX86Sse3, feature_bit(r.leaf1_ecx, 0));
  set_if(out, CpuFeature::kX86Pclmulqdq, feature_bit(r.leaf1_ecx, 1));
  set_if(out, CpuFeature::kX86Ssse3, feature_bit(r.leaf1_ecx, 9));
  set_if(out, CpuFeature::kX86Sse41, feature_bit(r.leaf1_ecx, 19));
  set_if(out, CpuFeature::kX86Sse42, feature_bit(r.leaf1_ecx, 20));
  set_if(out, CpuFeature::kX86Aes, feature_bit(r.leaf1_ecx, 25));
  set_if(out, CpuFeature::kX86Lzcnt, feature_bit(r.extended1_ecx, 5));
  set_if(out, CpuFeature::kX86Bmi1, feature_bit(r.leaf7_0_ebx, 3));
  set_if(out, CpuFeature::kX86Bmi2, feature_bit(r.leaf7_0_ebx, 8));

  const bool osxsave = feature_bit(r.leaf1_ecx, 27);
  const bool os_ymm =
      osxsave && feature_bit(r.xcr0, 1) && feature_bit(r.xcr0, 2);
  const bool os_zmm = os_ymm && feature_bit(r.xcr0, 5) &&
                      feature_bit(r.xcr0, 6) && feature_bit(r.xcr0, 7);

  const bool avx = feature_bit(r.leaf1_ecx, 28) && os_ymm;
  set_if(out, CpuFeature::kX86Avx, avx);
  set_if(out, CpuFeature::kX86Fma, feature_bit(r.leaf1_ecx, 12) && avx);
  set_if(out, CpuFeature::kX86F16c, feature_bit(r.leaf1_ecx, 29) && avx);
  set_if(out, CpuFeature::kX86Avx2, feature_bit(r.leaf7_0_ebx, 5) && avx);
  set_if(out, CpuFeature::kX86AvxVnni, feature_bit(r.leaf7_1_eax, 4) && avx);
  set_if(out, CpuFeature::kX86Gfni, feature_bit(r.leaf7_0_ecx, 8) && avx);
  set_if(out, CpuFeature::kX86Vaes, feature_bit(r.leaf7_0_ecx, 9) && avx);
  set_if(out, CpuFeature::kX86Vpclmulqdq,
         feature_bit(r.leaf7_0_ecx, 10) && avx);

  const bool avx512f = feature_bit(r.leaf7_0_ebx, 16) && avx && os_zmm;
  set_if(out, CpuFeature::kX86Avx512F, avx512f);
  set_if(out, CpuFeature::kX86Avx512Dq,
         feature_bit(r.leaf7_0_ebx, 17) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Cd,
         feature_bit(r.leaf7_0_ebx, 28) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Bw,
         feature_bit(r.leaf7_0_ebx, 30) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Vl,
         feature_bit(r.leaf7_0_ebx, 31) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Vbmi,
         feature_bit(r.leaf7_0_ecx, 1) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Vbmi2,
         feature_bit(r.leaf7_0_ecx, 6) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Vnni,
         feature_bit(r.leaf7_0_ecx, 11) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Bitalg,
         feature_bit(r.leaf7_0_ecx, 12) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Vpopcntdq,
         feature_bit(r.leaf7_0_ecx, 14) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Fp16,
         feature_bit(r.leaf7_0_edx, 23) && avx512f);
  set_if(out, CpuFeature::kX86Avx512Bf16,
         feature_bit(r.leaf7_1_eax, 5) && avx512f);
  // HYLE currently exposes the precise 512-bit AVX10.2 execution fact used by
  // its -mavx10.2-512 specialty TU. Leaf 7 advertises AVX10; leaf 24 supplies
  // both the version and supported vector widths.
  set_if(out, CpuFeature::kX86Avx10_2_512,
         feature_bit(r.leaf7_1_edx, 19) && (r.leaf24_ebx & 0xffu) >= 2 &&
             feature_bit(r.leaf24_ebx, 18) && avx && os_zmm);

  // AMX CPUID bits are hardware facts only. Linux thread permission remains a
  // separate platform-thread operation, hence the explicit Hardware names.
  const bool amx_tile = feature_bit(r.leaf7_0_edx, 24);
  set_if(out, CpuFeature::kX86AmxTileHardware, amx_tile);
  set_if(out, CpuFeature::kX86AmxBf16Hardware,
         amx_tile && feature_bit(r.leaf7_0_edx, 22));
  set_if(out, CpuFeature::kX86AmxInt8Hardware,
         amx_tile && feature_bit(r.leaf7_0_edx, 25));
  set_if(out, CpuFeature::kX86AmxFp16Hardware,
         amx_tile && feature_bit(r.leaf7_1_eax, 21));
  return out;
}

inline constexpr std::uint64_t kArmHwcapAes = std::uint64_t{1} << 3;
inline constexpr std::uint64_t kArmHwcapPmull = std::uint64_t{1} << 4;
inline constexpr std::uint64_t kArmHwcapFphp = std::uint64_t{1} << 9;
inline constexpr std::uint64_t kArmHwcapAsimdhp = std::uint64_t{1} << 10;
inline constexpr std::uint64_t kArmHwcapAsimddp = std::uint64_t{1} << 20;
inline constexpr std::uint64_t kArmHwcapSve = std::uint64_t{1} << 22;
inline constexpr std::uint64_t kArmHwcap2Sve2 = std::uint64_t{1} << 1;
inline constexpr std::uint64_t kArmHwcap2SveAes = std::uint64_t{1} << 2;
inline constexpr std::uint64_t kArmHwcap2SvePmull = std::uint64_t{1} << 3;
inline constexpr std::uint64_t kArmHwcap2SveI8mm = std::uint64_t{1} << 9;
inline constexpr std::uint64_t kArmHwcap2SveF32mm = std::uint64_t{1} << 10;
inline constexpr std::uint64_t kArmHwcap2SveF64mm = std::uint64_t{1} << 11;
inline constexpr std::uint64_t kArmHwcap2SveBf16 = std::uint64_t{1} << 12;
inline constexpr std::uint64_t kArmHwcap2I8mm = std::uint64_t{1} << 13;
inline constexpr std::uint64_t kArmHwcap2Bf16 = std::uint64_t{1} << 14;
inline constexpr std::uint64_t kArmHwcap2Sme = std::uint64_t{1} << 23;
inline constexpr std::uint64_t kArmHwcap2Sme2 = std::uint64_t{1} << 37;

HYLE_HIDDEN constexpr CpuFeatureSet decode_arm_linux_features(
    std::uint64_t hwcap, std::uint64_t hwcap2) noexcept {
  CpuFeatureSet out{CpuFeature::kArmNeon};  // AArch64 baseline
  set_if(out, CpuFeature::kArmAes, (hwcap & kArmHwcapAes) != 0);
  set_if(out, CpuFeature::kArmPmull, (hwcap & kArmHwcapPmull) != 0);
  set_if(out, CpuFeature::kArmDotProd, (hwcap & kArmHwcapAsimddp) != 0);
  set_if(out, CpuFeature::kArmFp16,
         (hwcap & (kArmHwcapFphp | kArmHwcapAsimdhp)) ==
             (kArmHwcapFphp | kArmHwcapAsimdhp));
  const bool sve = (hwcap & kArmHwcapSve) != 0;
  const bool sve2 = sve && (hwcap2 & kArmHwcap2Sve2) != 0;
  set_if(out, CpuFeature::kArmSve, sve);
  set_if(out, CpuFeature::kArmSve2, sve2);
  set_if(out, CpuFeature::kArmSveAes, sve2 && (hwcap2 & kArmHwcap2SveAes) != 0);
  set_if(out, CpuFeature::kArmSvePmull,
         sve2 && (hwcap2 & kArmHwcap2SvePmull) != 0);
  set_if(out, CpuFeature::kArmSveI8mm,
         sve && (hwcap2 & kArmHwcap2SveI8mm) != 0);
  set_if(out, CpuFeature::kArmSveF32mm,
         sve && (hwcap2 & kArmHwcap2SveF32mm) != 0);
  set_if(out, CpuFeature::kArmSveF64mm,
         sve && (hwcap2 & kArmHwcap2SveF64mm) != 0);
  set_if(out, CpuFeature::kArmSveBf16,
         sve && (hwcap2 & kArmHwcap2SveBf16) != 0);
  set_if(out, CpuFeature::kArmI8mm, (hwcap2 & kArmHwcap2I8mm) != 0);
  set_if(out, CpuFeature::kArmBf16, (hwcap2 & kArmHwcap2Bf16) != 0);
  const bool sme = (hwcap2 & kArmHwcap2Sme) != 0;
  set_if(out, CpuFeature::kArmSme, sme);
  set_if(out, CpuFeature::kArmSme2, sme && (hwcap2 & kArmHwcap2Sme2) != 0);
  return out;
}

HYLE_HIDDEN inline bool parse_cpu_quota_u64(const char* text,
                                            std::uint64_t& value) noexcept {
  if (text == nullptr) return false;
  while (*text == ' ' || *text == '\t') ++text;
  if (*text < '0' || *text > '9') return false;
  errno = 0;
  char* end = nullptr;
  const std::uint64_t parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text) return false;
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end != '\0') return false;
  value = parsed;
  return true;
}

// Pure helpers shared with parser fixtures. The public value rejects zero
// fields; v2 "max" and v1 -1 intentionally report no hard quota.
HYLE_HIDDEN inline bool parse_linux_cpu_max(const char* text,
                                            CpuQuota& quota) noexcept {
  if (text == nullptr) return false;
  while (*text == ' ' || *text == '\t') ++text;
  const char* split = text;
  while (*split != '\0' && *split != ' ' && *split != '\t' && *split != '\r' &&
         *split != '\n') {
    ++split;
  }
  if (split == text) return false;
  const bool unlimited = static_cast<std::size_t>(split - text) == 3u &&
                         std::memcmp(text, "max", 3) == 0;
  std::uint64_t capacity = 0;
  if (!unlimited) {
    char capacity_text[96];
    const std::size_t length = static_cast<std::size_t>(split - text);
    if (length >= sizeof(capacity_text)) return false;
    std::memcpy(capacity_text, text, length);
    capacity_text[length] = '\0';
    if (!parse_cpu_quota_u64(capacity_text, capacity) || capacity == 0) {
      return false;
    }
  }
  while (*split == ' ' || *split == '\t') ++split;
  std::uint64_t period = 0;
  if (!parse_cpu_quota_u64(split, period) || period == 0 || unlimited) {
    return false;
  }
  quota = CpuQuota{capacity, period};
  return true;
}

HYLE_HIDDEN inline bool parse_linux_cpu_quota(const char* quota_text,
                                              const char* period_text,
                                              CpuQuota& quota) noexcept {
  if (quota_text == nullptr || period_text == nullptr) return false;
  while (*quota_text == ' ' || *quota_text == '\t') ++quota_text;
  if (quota_text[0] == '-' && quota_text[1] == '1') {
    const char* end = quota_text + 2;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
      ++end;
    }
    if (*end == '\0') return false;
  }
  std::uint64_t capacity = 0;
  std::uint64_t period = 0;
  if (!parse_cpu_quota_u64(quota_text, capacity) ||
      !parse_cpu_quota_u64(period_text, period) || capacity == 0 ||
      period == 0) {
    return false;
  }
  quota = CpuQuota{capacity, period};
  return true;
}

HYLE_HIDDEN inline bool positive_fraction_less(
    std::uint64_t left_numerator, std::uint64_t left_denominator,
    std::uint64_t right_numerator, std::uint64_t right_denominator) noexcept {
  bool reversed = false;
  for (;;) {
    const std::uint64_t left_whole = left_numerator / left_denominator;
    const std::uint64_t right_whole = right_numerator / right_denominator;
    if (left_whole != right_whole) {
      return reversed ? left_whole > right_whole : left_whole < right_whole;
    }
    left_numerator %= left_denominator;
    right_numerator %= right_denominator;
    if (left_numerator == 0 || right_numerator == 0) {
      if (left_numerator == right_numerator) return false;
      return reversed ? left_numerator != 0 : left_numerator == 0;
    }
    std::swap(left_numerator, left_denominator);
    std::swap(right_numerator, right_denominator);
    reversed = !reversed;
  }
}

HYLE_HIDDEN inline bool cpu_quota_tighter(CpuQuota candidate,
                                          CpuQuota current) noexcept {
  if (candidate.capacity == 0 || candidate.period == 0) return false;
  if (current.capacity == 0 || current.period == 0) return true;
  return positive_fraction_less(candidate.capacity, candidate.period,
                                current.capacity, current.period);
}

}  // namespace hyle::platform::detail
