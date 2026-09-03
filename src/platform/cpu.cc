#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE 1
#endif

#include "hyle/platform/cpu.h"

#include "cpu_internal.h"

// cpu.cc -- the ISOLATED, LTO-DISABLED probe TU (conventions §10 probe isolation).
// Nothing here may use an instruction above the baseline ISA: this code runs FIRST, on
// an as-yet-unknown CPU, to discover what that CPU can run. If LTO inlined an AVX
// instruction into FlagsFromCpuid(), an old CPU would SIGILL inside the very function
// meant to tell us it's old. The CMake for this TU disables LTO/IPO for that reason.
// We compile it at the default baseline (no -mavx etc.), so the compiler cannot emit
// anything beyond it here.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "hyle/platform/numa.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#if HYLE_OS_LINUX && (HYLE_ARCH_ARM64 || HYLE_ARCH_RISCV64)
#include <sys/auxv.h>
#endif

#if HYLE_OS_MACOS
#include <sys/sysctl.h>  // sysctlbyname("hw.cachelinesize")
#include <unistd.h>
#endif
#if HYLE_OS_LINUX
#include <dirent.h>
#include <sched.h>
#include <unistd.h>  // sysconf(_SC_LEVEL1_DCACHE_LINESIZE)
#endif

#if HYLE_OS_LINUX && HYLE_ARCH_RISCV64
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if HYLE_ARCH_X64
#  if HYLE_COMPILER_MSVC
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

namespace hyle::platform {
namespace {

#if HYLE_ARCH_X64

// CPUID(leaf, subleaf) -> {eax, ebx, ecx, edx} in abcd. Wraps the two compiler ABIs
// (Highway x86_cpuid.h does the same): MSVC's __cpuidex vs GCC/Clang's __cpuid_count.
void cpuid(std::uint32_t leaf, std::uint32_t subleaf,
           std::uint32_t abcd[4]) noexcept {
#  if HYLE_COMPILER_MSVC
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  for (int i = 0; i < 4; ++i) abcd[i] = static_cast<std::uint32_t>(regs[i]);
#  else
  std::uint32_t a, b, c, d;
  __cpuid_count(leaf, subleaf, a, b, c, d);
  abcd[0] = a;
  abcd[1] = b;
  abcd[2] = c;
  abcd[3] = d;
#  endif
}

// Lower 32 bits of XCR0 (extended control register 0). ONLY call after confirming the
// OSXSAVE bit (CPUID.1:ECX[27]) -- XGETBV is itself an illegal instruction if the OS
// hasn't enabled XSAVE. This is how we learn whether the OS actually preserves YMM/ZMM
// across context switches; a CPU can have AVX while a stale OS does not save YMM, and
// using AVX there silently corrupts register state. (Highway targets.cc ReadXCR0.)
std::uint32_t read_xcr0() noexcept {
#  if HYLE_COMPILER_MSVC
  return static_cast<std::uint32_t>(_xgetbv(0));
#  else
  std::uint32_t lo, hi;
  // .byte encoding of `xgetbv` so it assembles without -mxsave. ecx=0 selects XCR0.
  __asm__ volatile(".byte 0x0F, 0x01, 0xD0" : "=a"(lo), "=d"(hi) : "c"(0));
  (void)hi;
  return lo;
#  endif
}

inline bool bit(std::uint32_t reg, int idx) noexcept {
  return (reg & (std::uint32_t{1} << idx)) != 0;
}

constexpr bool has_avx10_2(std::uint32_t leaf24_ebx) noexcept {
  const std::uint32_t version = leaf24_ebx & 0xFFu;
  return version >= 2;
}
static_assert(!has_avx10_2(1));
static_assert(has_avx10_2(2));

CpuFeatures detect_x86() noexcept {
  CpuFeatures f;
  std::uint32_t abcd[4];

  cpuid(0, 0, abcd);
  const std::uint32_t max_leaf = abcd[0];

  f.sse2 = true;  // guaranteed on x86-64

  // leaf 1: legacy + AVX/F16C/OSXSAVE
  cpuid(1, 0, abcd);
  const std::uint32_t ecx1 = abcd[2];
  f.ssse3 = bit(ecx1, 9);
  f.sse41 = bit(ecx1, 19);
  f.sse42 = bit(ecx1, 20);
  f.fma = bit(ecx1, 12);
  f.f16c = bit(ecx1, 29);
  const bool cpu_avx = bit(ecx1, 28);
  const bool osxsave = bit(ecx1, 27);

  // leaf 0x80000001 (extended): LZCNT. Part of Highway's AVX2 feature group, so the
  // AVX2 tier (compute_best) requires it alongside avx2+fma+f16c+bmi.
  cpuid(0x80000000u, 0, abcd);  // max EXTENDED leaf
  if (abcd[0] >= 0x80000001u) {
    cpuid(0x80000001u, 0, abcd);
    f.lzcnt = bit(abcd[2], 5);  // ECX bit 5
  }

  // leaf 7,0 and 7,1: AVX2 / BMI / AVX-512 family
  bool cpu_avx2 = false;
  bool cpu_avx512f = false, cpu_avx512bw = false, cpu_avx512vl = false;
  bool cpu_avx512dq = false, cpu_avx512cd = false;
  bool cpu_avx10_2 = false;
  if (max_leaf >= 7) {
    cpuid(7, 0, abcd);
    const std::uint32_t ebx7 = abcd[1];
    const std::uint32_t ecx7 = abcd[2];
    const std::uint32_t edx7 = abcd[3];
    f.bmi1 = bit(ebx7, 3);
    cpu_avx2 = bit(ebx7, 5);
    f.bmi2 = bit(ebx7, 8);
    cpu_avx512f = bit(ebx7, 16);
    cpu_avx512dq = bit(ebx7, 17);
    cpu_avx512cd = bit(ebx7, 28);
    cpu_avx512bw = bit(ebx7, 30);
    cpu_avx512vl = bit(ebx7, 31);
    f.avx512vnni = bit(ecx7, 11);
    f.avx512fp16 = bit(edx7, 23);
    f.amx_bf16 = bit(edx7, 22);
    f.amx_tile = bit(edx7, 24);
    f.amx_int8 = bit(edx7, 25);

    cpuid(7, 1, abcd);
    // AVX-VNNI (VEX encoding, 256-bit): leaf 7,1 EAX bit 4 — independent of AVX-512.
    f.avx_vnni = bit(abcd[0], 4);
    f.avx512bf16 = bit(abcd[0], 5);
    // AVX10: leaf 7,1 EDX bit 19 advertises AVX10 support; leaf 0x24
    // subleaf 0 reports the version in EBX[7:0]. Current Intel SDM revisions
    // reserve EBX[18:16] (always 111b): every AVX10 processor supports all
    // vector widths, so they must not be interpreted as width capability bits.
    const bool avx10_supported = bit(abcd[3], 19);
    if (avx10_supported && max_leaf >= 0x24) {
      cpuid(0x24u, 0, abcd);
      cpu_avx10_2 = has_avx10_2(abcd[1]);
    }
  }

  // Gate everything that uses YMM/ZMM on the OS actually saving those registers.
  // CPUID alone is not enough (see read_xcr0 comment).
  bool os_ymm = false, os_zmm = false;
  if (osxsave) {
    const std::uint32_t xcr0 = read_xcr0();
    os_ymm = bit(xcr0, 1) && bit(xcr0, 2);                       // XMM + YMM state
    os_zmm = os_ymm && bit(xcr0, 5) && bit(xcr0, 6) && bit(xcr0, 7);  // + opmask/ZMM
  }

  f.avx = cpu_avx && os_ymm;
  f.avx2 = cpu_avx2 && os_ymm;
  const bool avx512_ok = os_zmm;
  f.avx512f = cpu_avx512f && avx512_ok;
  f.avx512bw = cpu_avx512bw && avx512_ok;
  f.avx512vl = cpu_avx512vl && avx512_ok;
  f.avx512dq = cpu_avx512dq && avx512_ok;
  f.avx512cd = cpu_avx512cd && avx512_ok;
  // AVX-512 VNNI/BF16/FP16 need OS ZMM save; AVX-VNNI only needs YMM (same as avx2).
  f.avx512vnni = f.avx512vnni && avx512_ok;
  f.avx_vnni = f.avx_vnni && os_ymm;
  f.avx512bf16 = f.avx512bf16 && avx512_ok;
  // AVX10.2 fp8 converts operate on ZMM (512-bit), so require OS ZMM save just like AVX-512.
  f.avx10_2 = cpu_avx10_2 && avx512_ok;
  f.avx512fp16 = f.avx512fp16 && avx512_ok;

  // AMX tile state is enabled independently from XCR0 and, on Linux, must be
  // requested for each thread. Keep these as static CPUID facts; the thread
  // factor owns the permission query/request mechanism.
  f.amx_int8 = f.amx_tile && f.amx_int8;
  f.amx_bf16 = f.amx_tile && f.amx_bf16;

  return f;
}

#endif  // HYLE_ARCH_X64

#if HYLE_OS_LINUX && HYLE_ARCH_RISCV64

constexpr std::uint64_t kRiscvExtV = std::uint64_t{1} << 2;
constexpr std::uint64_t kRiscvExtZvbb = std::uint64_t{1} << 17;
constexpr std::uint64_t kRiscvExtZvbc = std::uint64_t{1} << 18;
constexpr std::uint64_t kRiscvExtZvkned = std::uint64_t{1} << 21;
constexpr std::uint64_t kRiscvExtZvfh = std::uint64_t{1} << 30;
constexpr std::uint64_t kRiscvExtZvfhmin = std::uint64_t{1} << 31;

constexpr void apply_riscv_hwprobe(std::uint64_t bits,
                                   CpuFeatures& f) noexcept {
  f.rvv = f.rvv || (bits & kRiscvExtV) != 0;
  f.rvv_zvbb = f.rvv && (bits & kRiscvExtZvbb) != 0;
  f.rvv_zvbc = f.rvv && (bits & kRiscvExtZvbc) != 0;
  f.rvv_zvkned = f.rvv && (bits & kRiscvExtZvkned) != 0;
  f.rvv_zvfh = f.rvv && (bits & kRiscvExtZvfh) != 0;
  f.rvv_zvfhmin = f.rvv && (bits & kRiscvExtZvfhmin) != 0;
}

// Compile-time synthetic decoding keeps every stable Linux UABI bit covered
// even when qemu-linux-user cannot emulate riscv_hwprobe optional extensions.
static_assert([] {
  CpuFeatures f;
  apply_riscv_hwprobe(kRiscvExtV | kRiscvExtZvbb | kRiscvExtZvbc |
                          kRiscvExtZvkned | kRiscvExtZvfh |
                          kRiscvExtZvfhmin,
                      f);
  return f.rvv && f.rvv_zvbb && f.rvv_zvbc && f.rvv_zvkned &&
         f.rvv_zvfh && f.rvv_zvfhmin;
}());
static_assert([] {
  CpuFeatures f;
  apply_riscv_hwprobe(kRiscvExtZvbb | kRiscvExtZvbc, f);
  return !f.rvv && !f.rvv_zvbb && !f.rvv_zvbc;
}());

#endif

CpuFeatures detect() noexcept {
#if HYLE_ARCH_X64
  return detect_x86();
#elif HYLE_ARCH_ARM64
  CpuFeatures f;
  f.neon = true;  // mandatory on aarch64
#if HYLE_OS_LINUX
  // Keep the constants local instead of including asm/hwcap.h: old deployment
  // sysroots may not name newer bits, but the stable Linux userspace ABI does.
  constexpr unsigned long kHwcapSve = 1ul << 22;
  constexpr unsigned long kAtHwcap2 = 26;
  constexpr unsigned long kHwcap2Sve2 = 1ul << 1;
  constexpr unsigned long kHwcap2SveAes = 1ul << 2;
  constexpr unsigned long kHwcap2SvePmull = 1ul << 3;
  constexpr unsigned long kHwcap2SveI8mm = 1ul << 9;
  constexpr unsigned long kHwcap2SveF32mm = 1ul << 10;
  constexpr unsigned long kHwcap2SveF64mm = 1ul << 11;
  constexpr unsigned long kHwcap2SveBf16 = 1ul << 12;
  constexpr unsigned long kHwcap2I8mm = 1ul << 13;
  constexpr unsigned long kHwcap2Bf16 = 1ul << 14;
  constexpr unsigned long kHwcap2Sme = 1ul << 23;
  constexpr unsigned long kHwcap2Sme2 = 1ull << 37;

  const unsigned long caps = getauxval(AT_HWCAP);
  const unsigned long caps2 = getauxval(kAtHwcap2);
  f.sve = (caps & kHwcapSve) != 0;
  f.sve2 = f.sve && (caps2 & kHwcap2Sve2) != 0;
  // SVE AES/PMULL instructions are SVE2 subextensions. Requiring both bits makes
  // the implication explicit and prevents a malformed auxv from selecting a
  // +sve2-aes TU on an SVE1-only process.
  f.sve_aes = f.sve2 && (caps2 & kHwcap2SveAes) != 0;
  f.sve_pmull = f.sve2 && (caps2 & kHwcap2SvePmull) != 0;
  f.sve_i8mm = f.sve && (caps2 & kHwcap2SveI8mm) != 0;
  f.sve_f32mm = f.sve && (caps2 & kHwcap2SveF32mm) != 0;
  f.sve_f64mm = f.sve && (caps2 & kHwcap2SveF64mm) != 0;
  f.sve_bf16 = f.sve && (caps2 & kHwcap2SveBf16) != 0;
  f.arm_i8mm = (caps2 & kHwcap2I8mm) != 0;
  f.arm_bf16 = (caps2 & kHwcap2Bf16) != 0;
  f.sme = (caps2 & kHwcap2Sme) != 0;
  f.sme2 = f.sme && (caps2 & kHwcap2Sme2) != 0;
#elif HYLE_OS_MACOS
  auto sysctl_feature = [](const char* name) noexcept {
    int present = 0;
    std::size_t size = sizeof(present);
    return ::sysctlbyname(name, &present, &size, nullptr, 0) == 0 &&
           present != 0;
  };
  f.sme = sysctl_feature("hw.optional.arm.FEAT_SME");
  f.sme2 = f.sme && sysctl_feature("hw.optional.arm.FEAT_SME2");
#elif HYLE_OS_WINDOWS
  // These numeric values are the stable Windows processor-feature ABI. Keep
  // local fallbacks so older SDKs can still build for newer ARM64 hosts.
  constexpr DWORD kPfArmSme = 70;
  constexpr DWORD kPfArmSme2 = 71;
  f.sme = ::IsProcessorFeaturePresent(kPfArmSme) != 0;
  f.sme2 = f.sme && ::IsProcessorFeaturePresent(kPfArmSme2) != 0;
#endif
  return f;
#elif HYLE_ARCH_RISCV64
  CpuFeatures f;
#if HYLE_OS_LINUX
  constexpr unsigned long kHwcapIsaV = 1ul << ('V' - 'A');
  f.rvv = (getauxval(AT_HWCAP) & kHwcapIsaV) != 0;

  // riscv_hwprobe (Linux 6.4+) is the architecture's precise extension ABI.
  // Use a private copy of the stable two-word pair so this baseline detector
  // still builds against older kernel headers. On an older kernel/QEMU the
  // syscall returns ENOSYS and all optional bits safely remain false.
  struct RiscvHwprobePair {
    std::int64_t key;
    std::uint64_t value;
  };
  constexpr std::int64_t kKeyImaExt0 = 4;

  RiscvHwprobePair pair{kKeyImaExt0, 0};
  long rc = -1;
#  if defined(SYS_riscv_hwprobe)
  rc = syscall(SYS_riscv_hwprobe, &pair, std::size_t{1}, std::size_t{0},
               nullptr, 0ul);
#  elif defined(__NR_riscv_hwprobe)
  rc = syscall(__NR_riscv_hwprobe, &pair, std::size_t{1}, std::size_t{0},
               nullptr, 0ul);
#  endif
  if (rc == 0 && pair.key == kKeyImaExt0) {
    apply_riscv_hwprobe(pair.value, f);
  }
#endif
  return f;
#else
  return CpuFeatures{};  // scalar-only platform
#endif
}

Isa compute_best(const CpuFeatures& f) noexcept {
#if HYLE_ARCH_X64
  // Mirror Highway's group gating (targets.cc kGroupAVX2 / kGroupAVX3): a tier needs
  // ALL of its group's features, AND each tier's group SUBSUMES the lower tier's, so
  // a higher Isa genuinely implies every lower one (the enum's contract).
  // kGroupAVX2 = AVX2 + FMA + F16C + LZCNT + BMI1 + BMI2 (+ SSE4 below it).
  const auto groups = detail::x86_isa_groups(f);
  // kGroupAVX3 = AVX-512 F+BW+VL+DQ+CD on TOP of the full AVX2 group.
  if (groups.avx512) return Isa::kAvx512;
  if (groups.avx2) return Isa::kAvx2;
  // Pure AVX (256-bit FLOAT) without the full AVX2 integer group: Sandy/Ivy Bridge. f.avx
  // already AND's in the OS-YMM-save check (read_xcr0), so reporting kAvx here is SIGILL-
  // safe. This is a DETECTION tier -- best_isa() tells the truth about the CPU; there is no
  // avx-only KERNEL, so selected_tier() (dispatch_ops.cc) clamps an AVX host down to the
  // sse4 variant. The two layers are deliberately separate (probe honesty vs shipped code).
  if (f.avx) return Isa::kAvx;
  if (f.sse42) return Isa::kSse42;
  if (f.ssse3) return Isa::kSsse3;
  if (f.sse2) return Isa::kSse2;
  return Isa::kScalar;
#elif HYLE_ARCH_ARM64
  if (f.sve) return Isa::kSve;
  if (f.neon) return Isa::kNeon;
  return Isa::kScalar;
#elif HYLE_ARCH_RISCV64
  return f.rvv ? Isa::kRvv : Isa::kScalar;
#else
  (void)f;
  return Isa::kScalar;
#endif
}

// L1 data cache line size in bytes. x86: CPUID leaf 1 EBX[15:8] (CLFLUSH line
// size in 8-byte units). Windows non-x86: RelationCache. macOS: sysctl. Linux:
// sysconf. 64 fallback. Compiled in this LTO-off baseline TU so the x86 CPUID
// path is SIGILL-safe.
unsigned detect_cache_line() noexcept {
#if HYLE_ARCH_X64
  std::uint32_t abcd[4];
  cpuid(1, 0, abcd);
  const unsigned line = ((abcd[1] >> 8) & 0xffu) * 8u;
  return line ? line : 64u;
#elif HYLE_OS_WINDOWS
  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  DWORD bytes = 0;
  if (::GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes) ||
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER ||
      bytes < kRelationHeader) {
    return 64u;
  }
  auto *storage = new (std::nothrow) std::byte[bytes];
  if (storage == nullptr)
    return 64u;
  auto *buffer = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
      storage);
  if (!::GetLogicalProcessorInformationEx(RelationCache, buffer, &bytes)) {
    delete[] storage;
    return 64u;
  }

  unsigned detected = 0;
  DWORD offset = 0;
  while (offset + kRelationHeader <= bytes) {
    const auto *info =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
            storage + offset);
    if (info->Size < kRelationHeader ||
        info->Size > bytes - offset) {
      break;
    }
    const std::size_t cache_header =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
        offsetof(CACHE_RELATIONSHIP, GroupMasks);
    if (info->Relationship == RelationCache &&
        info->Size >= cache_header + sizeof(GROUP_AFFINITY) &&
        info->Cache.Level == 1 &&
        (info->Cache.Type == CacheData ||
         info->Cache.Type == CacheUnified)) {
      const unsigned line = info->Cache.LineSize;
      if (line >= 16u && line <= 256u && (line & (line - 1u)) == 0u)
        detected = std::max(detected, line);
    }
    offset += info->Size;
  }
  delete[] storage;
  return detected != 0 ? detected : 64u;
#elif HYLE_OS_MACOS
  std::size_t v = 0;
  std::size_t sz = sizeof(v);
  if (::sysctlbyname("hw.cachelinesize", &v, &sz, nullptr, 0) == 0 && v != 0) {
    return static_cast<unsigned>(v);
  }
  return 64u;
#elif HYLE_OS_LINUX
  const long v = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  return v > 0 ? static_cast<unsigned>(v) : 64u;
#else
  return 64u;
#endif
}

CacheInfo detect_cache_info() noexcept {
  CacheInfo out;
#if HYLE_ARCH_X64
  std::uint32_t abcd[4];
  cpuid(0, 0, abcd);
  const std::uint32_t max_basic = abcd[0];
  auto enumerate = [&](std::uint32_t leaf) noexcept {
    for (std::uint32_t subleaf = 0; subleaf != 64; ++subleaf) {
      cpuid(leaf, subleaf, abcd);
      const unsigned type = abcd[0] & 0x1fu;
      if (type == 0) break;
      const unsigned level = (abcd[0] >> 5) & 0x7u;
      std::uint64_t size = ((abcd[1] >> 22) & 0x3ffu) + 1u;
      const std::uint64_t factors[] = {
          ((abcd[1] >> 12) & 0x3ffu) + 1u,
          (abcd[1] & 0xfffu) + 1u,
          static_cast<std::uint64_t>(abcd[2]) + 1u};
      for (const std::uint64_t factor : factors) {
        if (size > std::numeric_limits<std::uint64_t>::max() / factor) {
          size = 0;
          break;
        }
        size *= factor;
      }
      if (level == 1 && (type == 1 || type == 3) && size > out.l1d)
        out.l1d = size;
      if (level == 1 && (type == 2 || type == 3) && size > out.l1i)
        out.l1i = size;
      if (level == 2 && size > out.l2) out.l2 = size;
      if (level == 3 && size > out.l3) out.l3 = size;
    }
  };
  if (max_basic >= 4) enumerate(4);
  if (out.l1d == 0 && out.l1i == 0 && out.l2 == 0) {
    cpuid(0x80000000u, 0, abcd);
    if (abcd[0] >= 0x8000001du) enumerate(0x8000001du);
  }
#elif HYLE_OS_MACOS
  auto get = [](const char* name) noexcept {
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
  };
  out.l1d = get("hw.l1dcachesize");
  out.l1i = get("hw.l1icachesize");
  out.l2 = get("hw.l2cachesize");
  out.l3 = get("hw.l3cachesize");
#elif HYLE_OS_LINUX
  auto get = [](int name) noexcept {
    const long value = ::sysconf(name);
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
  };
#  ifdef _SC_LEVEL1_DCACHE_SIZE
  out.l1d = get(_SC_LEVEL1_DCACHE_SIZE);
#  endif
#  ifdef _SC_LEVEL1_ICACHE_SIZE
  out.l1i = get(_SC_LEVEL1_ICACHE_SIZE);
#  endif
#  ifdef _SC_LEVEL2_CACHE_SIZE
  out.l2 = get(_SC_LEVEL2_CACHE_SIZE);
#  endif
#  ifdef _SC_LEVEL3_CACHE_SIZE
  out.l3 = get(_SC_LEVEL3_CACHE_SIZE);
#  endif
#endif
  return out;
}

#if HYLE_OS_LINUX
bool read_sysfs_unsigned(const char* path, unsigned& value) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (!file) return false;
  const bool ok = std::fscanf(file, "%u", &value) == 1;
  std::fclose(file);
  return ok;
}
#endif

#if HYLE_OS_WINDOWS || HYLE_OS_LINUX
Status cpu_query_status(const char* operation, unsigned long code) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_win32(static_cast<std::uint32_t>(code), operation);
#else
  return status_from_errno(static_cast<int>(code), operation);
#endif
}
#endif

Result<std::size_t> query_available_cpu_ids(
    unsigned* dst, std::size_t cap,
    unsigned* performance_ranks = nullptr) noexcept {
  if (dst == nullptr) cap = 0;
#if HYLE_OS_WINDOWS
  HANDLE process = ::GetCurrentProcess();

  // A process may explicitly select CPU Set ids. Empty means no CPU-Set
  // restriction; exclusive sets allocated to another process remain unusable.
  ULONG default_count = 0;
  ::SetLastError(ERROR_SUCCESS);
  BOOL defaults_ok =
      ::GetProcessDefaultCpuSets(process, nullptr, 0, &default_count);
  DWORD code = defaults_ok ? ERROR_SUCCESS : ::GetLastError();
  if (!defaults_ok && code != ERROR_INSUFFICIENT_BUFFER) {
    return std::unexpected(
        cpu_query_status("GetProcessDefaultCpuSets(size)", code));
  }
  ULONG default_stack[64];
  ULONG* default_ids = nullptr;
  bool default_ids_heap = false;
  if (default_count != 0) {
    if (default_count <= sizeof(default_stack) / sizeof(default_stack[0])) {
      default_ids = default_stack;
    } else {
      default_ids = new (std::nothrow) ULONG[default_count];
      default_ids_heap = true;
      if (default_ids == nullptr) {
        return std::unexpected(Status::ResourceExhausted(
            "available CPU ids: default-set allocation failed"));
      }
    }
    ULONG actual = default_count;
    if (!::GetProcessDefaultCpuSets(process, default_ids, default_count,
                                    &actual)) {
      code = ::GetLastError();
      if (default_ids_heap) delete[] default_ids;
      return std::unexpected(
          cpu_query_status("GetProcessDefaultCpuSets", code));
    }
    default_count = actual;
    // CPU-set ids are opaque ULONG values and need not be dense. Sorting the
    // caller-owned snapshot keeps membership O(log D) without a max-id bitmap;
    // the previous nested scan was O(system_sets * default_sets), quadratic on
    // large explicitly partitioned Windows hosts.
    std::sort(default_ids, default_ids + default_count);
  }

  DWORD bytes = 0;
  (void)::GetSystemCpuSetInformation(nullptr, 0, &bytes, process, 0);
  if (bytes == 0) {
    code = ::GetLastError();
    if (default_ids_heap) delete[] default_ids;
    return std::unexpected(
        cpu_query_status("GetSystemCpuSetInformation(size)", code));
  }
  alignas(SYSTEM_CPU_SET_INFORMATION) unsigned char storage_stack[4096];
  unsigned char* storage = nullptr;
  bool storage_heap = false;
  if (bytes <= sizeof(storage_stack)) {
    storage = storage_stack;
  } else {
    storage = new (std::nothrow) unsigned char[bytes];
    storage_heap = true;
    if (storage == nullptr) {
      if (default_ids_heap) delete[] default_ids;
      return std::unexpected(Status::ResourceExhausted(
          "available CPU ids: CPU-set allocation failed"));
    }
  }
  if (!::GetSystemCpuSetInformation(
          reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage), bytes,
          &bytes, process, 0)) {
    code = ::GetLastError();
    if (storage_heap) delete[] storage;
    if (default_ids_heap) delete[] default_ids;
    return std::unexpected(
        cpu_query_status("GetSystemCpuSetInformation", code));
  }

  const WORD groups = ::GetActiveProcessorGroupCount();
  DWORD_PTR process_mask = 0, system_mask = 0;
  const bool use_affinity_mask =
      groups == 1 &&
      ::GetProcessAffinityMask(process, &process_mask, &system_mask) != 0;
  std::size_t count = 0;
  for (DWORD offset = 0; offset < bytes;) {
    auto* info =
        reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage + offset);
    if (info->Size == 0 || info->Size > bytes - offset) break;
    if (info->Type == CpuSetInformation) {
      const auto& set = info->CpuSet;
      const bool selected =
          default_count == 0 ||
          std::binary_search(default_ids, default_ids + default_count, set.Id);
      if (selected && (!set.Allocated || set.AllocatedToTargetProcess)) {
        if (!use_affinity_mask ||
            (set.LogicalProcessorIndex < sizeof(DWORD_PTR) * 8u &&
             (process_mask & (DWORD_PTR{1} << set.LogicalProcessorIndex)))) {
          unsigned flat = set.LogicalProcessorIndex;
          for (WORD group = 0; group < set.Group; ++group) {
            flat += ::GetActiveProcessorCount(group);
          }
          if (count < cap) {
            dst[count] = flat;
            if (performance_ranks != nullptr) {
              // Zero is the portable "unknown" sentinel, so shift Windows'
              // native 0-based EfficiencyClass into a positive rank.
              performance_ranks[count] =
                  static_cast<unsigned>(set.EfficiencyClass) + 1u;
            }
          }
          ++count;
        }
      }
    }
    offset += info->Size;
  }
  if (storage_heap) delete[] storage;
  if (default_ids_heap) delete[] default_ids;
  if (count == 0) {
    return std::unexpected(Status::FailedPrecondition(
        "available CPU ids: process has no usable CPU set"));
  }
  return count;
#elif HYLE_OS_LINUX
  (void)performance_ranks;
  auto copy_set = [dst, cap](const cpu_set_t* set, std::size_t bytes,
                             std::size_t bits) noexcept
      -> Result<std::size_t> {
    std::size_t count = 0;
    for (std::size_t cpu = 0; cpu < bits; ++cpu) {
      if (CPU_ISSET_S(cpu, bytes, set)) {
        if (count < cap) dst[count] = static_cast<unsigned>(cpu);
        ++count;
      }
    }
    if (count == 0) {
      return std::unexpected(Status::FailedPrecondition(
          "available CPU ids: affinity mask is empty"));
    }
    return count;
  };

  // The common <=1024-id case needs no allocation. Linux returns EINVAL when
  // the supplied mask is smaller than the kernel's configured CPU id space;
  // only then grow a dynamic CPU_ALLOC mask.
  cpu_set_t fixed;
  CPU_ZERO(&fixed);
  if (::sched_getaffinity(0, sizeof(fixed), &fixed) == 0) {
    return copy_set(&fixed, sizeof(fixed), CPU_SETSIZE);
  }
  int error = errno;
  if (error != EINVAL) {
    return std::unexpected(cpu_query_status(
        "sched_getaffinity", static_cast<unsigned long>(error)));
  }

  for (std::size_t bits = CPU_SETSIZE * 2u;
       bits <= (std::size_t{1} << 20);
       bits *= 2) {
    cpu_set_t* set = CPU_ALLOC(bits);
    if (set == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "available CPU ids: affinity-mask allocation failed"));
    }
    const std::size_t bytes = CPU_ALLOC_SIZE(bits);
    CPU_ZERO_S(bytes, set);
    if (::sched_getaffinity(0, bytes, set) == 0) {
      auto result = copy_set(set, bytes, bits);
      CPU_FREE(set);
      return result;
    }
    error = errno;
    CPU_FREE(set);
    if (error != EINVAL) {
      return std::unexpected(cpu_query_status(
          "sched_getaffinity", static_cast<unsigned long>(error)));
    }
  }
  return std::unexpected(Status::OutOfRange(
      "available CPU ids: kernel CPU id space exceeds supported limit"));
#else
  (void)performance_ranks;
  const std::size_t count = hardware_concurrency();
  const std::size_t written = count < cap ? count : cap;
  for (std::size_t i = 0; i < written; ++i) {
    dst[i] = static_cast<unsigned>(i);
  }
  return count;
#endif
}

void fill_cpu_performance_ranks(const unsigned* ids, std::size_t count,
                                unsigned* ranks) noexcept {
  std::memset(ranks, 0, count * sizeof(*ranks));
#if HYLE_OS_WINDOWS
  // Windows writes ranks while scanning the same CPU-set snapshot as ids.
  // A second query here used to double the dominant native work and could
  // classify a different process-affinity snapshot.
  (void)ids;
#elif HYLE_OS_LINUX
  for (std::size_t i = 0; i < count; ++i) {
    char path[96];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/cpu_capacity", ids[i]);
    (void)read_sysfs_unsigned(path, ranks[i]);
  }
#else
  (void)ids;
#endif
}

void clear_cpu_location(CpuInfo& info) noexcept {
  info.package_id = kUnknownCpuTopologyId;
  info.die_id = kUnknownCpuTopologyId;
  info.core_id = kUnknownCpuTopologyId;
  info.smt_index = kUnknownCpuTopologyId;
  info.smt_count = 0;
  info.numa_node = kUnknownCpuTopologyId;
  info.l2_cache_domain = kUnknownCpuTopologyId;
  info.l3_cache_domain = kUnknownCpuTopologyId;
}

#if HYLE_OS_WINDOWS
void fill_cpu_locations(CpuInfo* info, std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) clear_cpu_location(info[i]);
  if (count == 0) return;

  unsigned maximum_id = 0;
  for (std::size_t i = 0; i < count; ++i) {
    maximum_id = (std::max)(maximum_id, info[i].id);
  }
  const std::size_t lookup_count = static_cast<std::size_t>(maximum_id) + 1u;
  CpuInfo* lookup_stack[256];
  CpuInfo** lookup = lookup_count <= std::size(lookup_stack)
                         ? lookup_stack
                         : new (std::nothrow) CpuInfo*[lookup_count];
  if (lookup == nullptr) return;
  std::fill(lookup, lookup + lookup_count, nullptr);
  for (std::size_t i = 0; i < count; ++i) lookup[info[i].id] = &info[i];

  const WORD group_count = ::GetActiveProcessorGroupCount();
  unsigned group_begin_stack[65]{};
  unsigned* group_begin = group_count < std::size(group_begin_stack)
                              ? group_begin_stack
                              : new (std::nothrow) unsigned[
                                    static_cast<std::size_t>(group_count) + 1u];
  if (group_begin == nullptr) {
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }
  for (WORD group = 0; group < group_count; ++group) {
    group_begin[group + 1u] =
        group_begin[group] + ::GetActiveProcessorCount(group);
  }

  auto for_each_mask = [&](const GROUP_AFFINITY& mask, auto&& fn) noexcept {
    if (mask.Group >= group_count) return;
    for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8u; ++bit) {
      if ((mask.Mask & (KAFFINITY{1} << bit)) == 0) continue;
      const std::size_t id =
          static_cast<std::size_t>(group_begin[mask.Group]) + bit;
      if (id < lookup_count && lookup[id] != nullptr) fn(*lookup[id]);
    }
  };

  DWORD bytes = 0;
  if (::GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes) ||
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
    if (group_begin != group_begin_stack) delete[] group_begin;
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }
  auto* storage = new (std::nothrow) std::byte[bytes];
  if (storage == nullptr) {
    if (group_begin != group_begin_stack) delete[] group_begin;
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
      storage);
  if (!::GetLogicalProcessorInformationEx(RelationAll, first, &bytes)) {
    delete[] storage;
    if (group_begin != group_begin_stack) delete[] group_begin;
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }

  unsigned package_id = 0;
  unsigned die_id = 0;
  unsigned core_id = 0;
  unsigned l2_id = 0;
  unsigned l3_id = 0;
  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  for (DWORD offset = 0; offset + kRelationHeader <= bytes;) {
    auto* relation =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage +
                                                                    offset);
    if (relation->Size < kRelationHeader || relation->Size > bytes - offset) {
      break;
    }
    if (relation->Relationship == RelationProcessorPackage ||
        relation->Relationship == RelationProcessorDie ||
        relation->Relationship == RelationProcessorCore) {
      const auto& processor = relation->Processor;
      const std::size_t processor_header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
          offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
      if (relation->Size < processor_header ||
          processor.GroupCount >
              (relation->Size - processor_header) / sizeof(GROUP_AFFINITY)) {
        offset += relation->Size;
        continue;
      }
      unsigned available_siblings = 0;
      if (relation->Relationship == RelationProcessorCore) {
        for (WORD group = 0; group < processor.GroupCount; ++group) {
          for_each_mask(processor.GroupMask[group],
                        [&](CpuInfo&) noexcept { ++available_siblings; });
        }
      }
      unsigned sibling_index = 0;
      for (WORD group = 0; group < processor.GroupCount; ++group) {
        for_each_mask(processor.GroupMask[group], [&](CpuInfo& cpu) noexcept {
              if (relation->Relationship == RelationProcessorPackage) {
                cpu.package_id = package_id;
              } else if (relation->Relationship == RelationProcessorDie) {
                cpu.die_id = die_id;
              } else {
                cpu.core_id = core_id;
                cpu.smt_index = sibling_index++;
                cpu.smt_count = available_siblings;
              }
            });
      }
      if (relation->Relationship == RelationProcessorPackage) ++package_id;
      else if (relation->Relationship == RelationProcessorDie) ++die_id;
      else ++core_id;
    } else if (relation->Relationship == RelationNumaNode ||
               relation->Relationship == RelationNumaNodeEx) {
      const auto& node = relation->NumaNode;
      const WORD masks = node.GroupCount == 0 ? 1 : node.GroupCount;
      const std::size_t node_header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, NumaNode) +
          offsetof(NUMA_NODE_RELATIONSHIP, GroupMasks);
      if (relation->Size < node_header ||
          masks > (relation->Size - node_header) / sizeof(GROUP_AFFINITY)) {
        offset += relation->Size;
        continue;
      }
      for (WORD group = 0; group < masks; ++group) {
        for_each_mask(node.GroupMasks[group], [&](CpuInfo& cpu) noexcept {
            cpu.numa_node = node.NodeNumber;
          });
      }
    } else if (relation->Relationship == RelationCache &&
               (relation->Cache.Level == 2 || relation->Cache.Level == 3)) {
      const bool l2 = relation->Cache.Level == 2;
      const unsigned domain = l2 ? l2_id++ : l3_id++;
      const WORD masks =
          relation->Cache.GroupCount == 0 ? 1 : relation->Cache.GroupCount;
      const std::size_t cache_header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
          offsetof(CACHE_RELATIONSHIP, GroupMasks);
      if (relation->Size < cache_header ||
          masks > (relation->Size - cache_header) / sizeof(GROUP_AFFINITY)) {
        offset += relation->Size;
        continue;
      }
      for (WORD group = 0; group < masks; ++group) {
        for_each_mask(relation->Cache.GroupMasks[group],
                      [&](CpuInfo& cpu) noexcept {
            (l2 ? cpu.l2_cache_domain : cpu.l3_cache_domain) = domain;
          });
      }
    }
    offset += relation->Size;
  }
  delete[] storage;
  if (group_begin != group_begin_stack) delete[] group_begin;
  if (lookup != lookup_stack) delete[] lookup;
}
#elif HYLE_OS_LINUX
bool read_sysfs_first_unsigned(const char* path, unsigned& value) noexcept {
  return read_sysfs_unsigned(path, value);
}

bool cpu_list_metrics(const char* path, const unsigned* available,
                      std::size_t available_count, unsigned target,
                      unsigned& first, unsigned& index,
                      unsigned& count) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) return false;
  char text[4096];
  const bool read = std::fgets(text, sizeof(text), file) != nullptr;
  std::fclose(file);
  if (!read) return false;
  first = kUnknownCpuTopologyId;
  index = kUnknownCpuTopologyId;
  count = 0;
  const char* cursor = text;
  while (*cursor != '\0') {
    char* end = nullptr;
    const unsigned long begin = std::strtoul(cursor, &end, 10);
    if (end == cursor || begin > (std::numeric_limits<unsigned>::max)()) {
      return false;
    }
    unsigned long finish = begin;
    cursor = end;
    if (*cursor == '-') {
      ++cursor;
      finish = std::strtoul(cursor, &end, 10);
      if (end == cursor || finish < begin ||
          finish > (std::numeric_limits<unsigned>::max)()) {
        return false;
      }
      cursor = end;
    }
    // Linux affinity enumeration is ascending. Intersect each reported range
    // with the process-visible ids by binary search instead of rescanning all
    // CPUs for every sibling range.
    const auto* range_begin =
        std::lower_bound(available, available + available_count,
                         static_cast<unsigned>(begin));
    const auto* range_end =
        std::upper_bound(range_begin, available + available_count,
                         static_cast<unsigned>(finish));
    if (range_begin != range_end) {
      if (first == kUnknownCpuTopologyId || *range_begin < first) {
        first = *range_begin;
      }
      const auto* target_it =
          std::lower_bound(range_begin, range_end, target);
      if (target_it != range_end && *target_it == target) {
        index = count + static_cast<unsigned>(target_it - range_begin);
      }
      count += static_cast<unsigned>(range_end - range_begin);
    }
    if (*cursor == ',') ++cursor;
    else if (*cursor == '\n' || *cursor == '\r' || *cursor == '\0') break;
    else return false;
  }
  return count != 0 && index != kUnknownCpuTopologyId;
}

bool linux_cpu_numa_node(unsigned cpu, unsigned& node) noexcept {
  const auto nodes = numa_nodes();
  if (nodes.size() == 1u) {
    node = nodes.front();
    return true;
  }
  char path[96];
  std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", cpu);
  DIR* directory = ::opendir(path);
  if (directory == nullptr) return false;
  bool found = false;
  while (const dirent* entry = ::readdir(directory)) {
    if (std::strncmp(entry->d_name, "node", 4) != 0) continue;
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(entry->d_name + 4, &end, 10);
    if (errno != 0 || end == entry->d_name + 4 || *end != '\0' ||
        parsed > (std::numeric_limits<unsigned>::max)()) {
      continue;
    }
    const unsigned candidate = static_cast<unsigned>(parsed);
    if (!numa_node_available(candidate)) continue;
    node = candidate;
    found = true;
    break;
  }
  ::closedir(directory);
  return found;
}

void fill_cpu_locations(CpuInfo* info, std::size_t count) noexcept {
  unsigned ids_stack[256];
  unsigned* ids = count <= std::size(ids_stack)
                      ? ids_stack
                      : new (std::nothrow) unsigned[count];
  if (ids == nullptr) {
    for (std::size_t i = 0; i < count; ++i) clear_cpu_location(info[i]);
    return;
  }
  for (std::size_t i = 0; i < count; ++i) ids[i] = info[i].id;

  for (std::size_t i = 0; i < count; ++i) {
    CpuInfo& cpu = info[i];
    clear_cpu_location(cpu);
    char path[160];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/topology/physical_package_id",
                  cpu.id);
    (void)read_sysfs_unsigned(path, cpu.package_id);
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/topology/die_id", cpu.id);
    (void)read_sysfs_unsigned(path, cpu.die_id);
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list",
                  cpu.id);
    unsigned first = 0;
    if (cpu_list_metrics(path, ids, count, cpu.id, first, cpu.smt_index,
                         cpu.smt_count)) {
      cpu.core_id = first;  // globally unique equivalence-class representative
    }
    (void)linux_cpu_numa_node(cpu.id, cpu.numa_node);
    for (unsigned cache = 0; cache != 16; ++cache) {
      unsigned level = 0;
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%u/cache/index%u/level", cpu.id,
                    cache);
      if (!read_sysfs_unsigned(path, level)) break;
      if (level != 2 && level != 3) continue;
      unsigned domain = 0;
      std::snprintf(
          path, sizeof(path),
          "/sys/devices/system/cpu/cpu%u/cache/index%u/shared_cpu_list", cpu.id,
          cache);
      if (!read_sysfs_first_unsigned(path, domain)) continue;
      if (level == 2) cpu.l2_cache_domain = domain;
      else cpu.l3_cache_domain = domain;
    }
  }
  if (ids != ids_stack) delete[] ids;
}
#else
void fill_cpu_locations(CpuInfo* info, std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) clear_cpu_location(info[i]);
}
#endif

Result<std::size_t> query_available_cpu_info(CpuInfo* dst,
                                             std::size_t cap) noexcept {
  if (dst == nullptr || cap == 0) {
    return query_available_cpu_ids(nullptr, 0);
  }

  // A caller-provided buffer is already the bounded output workspace. Query
  // the available set once, return the full count, and enrich only the prefix
  // that fits. The former count/fill/retry layering took two independent
  // affinity snapshots before doing any useful work.
  std::size_t capacity = hardware_concurrency();
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    if (capacity > (std::numeric_limits<std::size_t>::max() /
                    (2u * sizeof(unsigned)))) {
      return std::unexpected(
          Status::OutOfRange("available CPU info capacity overflows storage"));
    }
    unsigned stack[512];
    unsigned* storage = capacity <= 256
                            ? stack
                            : new (std::nothrow) unsigned[capacity * 2u];
    if (storage == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "available CPU info allocation failed"));
    }
    unsigned* ids = storage;
    unsigned* ranks = storage + capacity;
#if HYLE_OS_WINDOWS
    std::memset(ranks, 0, capacity * sizeof(*ranks));
    auto filled = query_available_cpu_ids(ids, capacity, ranks);
#else
    auto filled = query_available_cpu_ids(ids, capacity);
#endif
    if (!filled) {
      if (storage != stack) delete[] storage;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled > capacity) {
      capacity = *filled;
      if (storage != stack) delete[] storage;
      continue;
    }
    const std::size_t count = *filled;
#if !HYLE_OS_WINDOWS
    fill_cpu_performance_ranks(ids, count, ranks);
#endif
    unsigned max_rank = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (ranks[i] > max_rank) max_rank = ranks[i];
    }
    const std::size_t written = (std::min)(cap, count);
    for (std::size_t i = 0; i < written; ++i) {
      dst[i].id = ids[i];
      dst[i].performance_rank = ranks[i];
      dst[i].kind = ranks[i] == 0
                        ? CpuKind::kUnknown
                        : (ranks[i] == max_rank ? CpuKind::kPerformance
                                                : CpuKind::kEfficiency);
    }
    fill_cpu_locations(dst, written);
    if (storage != stack) delete[] storage;
    return count;
  }
  return std::unexpected(Status::Unavailable(
      "available CPU set expanded repeatedly during snapshot"));
}

unsigned count_distinct(const CpuInfo* info, std::size_t count,
                        unsigned CpuInfo::*field, unsigned* table,
                        std::size_t table_capacity) noexcept {
  std::fill(table, table + table_capacity, kUnknownCpuTopologyId);
  unsigned result = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const unsigned value = info[i].*field;
    if (value == kUnknownCpuTopologyId) continue;
    std::size_t slot =
        (static_cast<std::size_t>(value) * std::size_t{2654435761u}) &
        (table_capacity - 1u);
    while (table[slot] != kUnknownCpuTopologyId && table[slot] != value) {
      slot = (slot + 1u) & (table_capacity - 1u);
    }
    if (table[slot] == kUnknownCpuTopologyId) {
      table[slot] = value;
      ++result;
    }
  }
  return result;
}

#if HYLE_OS_WINDOWS
Result<CpuTopology> query_windows_available_cpu_topology() noexcept {
  std::size_t capacity = hardware_concurrency();
  unsigned stack[512];
  unsigned* storage = nullptr;
  Result<std::size_t> filled = std::unexpected(
      Status::Unavailable("CPU topology query did not run"));
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    if (capacity > (std::numeric_limits<std::size_t>::max)() /
                       (2u * sizeof(unsigned))) {
      return std::unexpected(
          Status::OutOfRange("available CPU count exceeds topology limits"));
    }
    storage = capacity <= 256
                  ? stack
                  : new (std::nothrow) unsigned[capacity * 2u];
    if (storage == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("CPU topology allocation failed"));
    }
    unsigned* ranks = storage + capacity;
    std::memset(ranks, 0, capacity * sizeof(*ranks));
    filled = query_available_cpu_ids(storage, capacity, ranks);
    if (!filled) {
      if (storage != stack) delete[] storage;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled <= capacity) break;
    if (storage != stack) delete[] storage;
    storage = nullptr;
    capacity = *filled;
  }
  if (storage == nullptr || *filled > capacity) {
    if (storage != nullptr && storage != stack) delete[] storage;
    return std::unexpected(Status::Unavailable(
        "available CPU set expanded repeatedly during topology query"));
  }

  const std::size_t count = *filled;
  const unsigned* ids = storage;
  const unsigned* ranks = storage + capacity;
  unsigned maximum_id = 0;
  unsigned maximum_rank = 0;
  for (std::size_t i = 0; i < count; ++i) {
    maximum_id = (std::max)(maximum_id, ids[i]);
    maximum_rank = (std::max)(maximum_rank, ranks[i]);
  }
  const std::size_t selected_count = static_cast<std::size_t>(maximum_id) + 1u;
  unsigned char selected_stack[4096];
  unsigned char* selected =
      selected_count <= std::size(selected_stack)
          ? selected_stack
          : new (std::nothrow) unsigned char[selected_count];
  if (selected == nullptr) {
    if (storage != stack) delete[] storage;
    return std::unexpected(Status::ResourceExhausted(
        "CPU topology visibility allocation failed"));
  }
  std::memset(selected, 0, selected_count);

  CpuTopology out;
  out.logical_cpus = static_cast<unsigned>(count);
  for (std::size_t i = 0; i < count; ++i) {
    selected[ids[i]] = 1;
    if (ranks[i] == 0) continue;
    if (ranks[i] == maximum_rank) ++out.performance_cpus;
    else ++out.efficiency_cpus;
  }
  out.hybrid = out.performance_cpus != 0 && out.efficiency_cpus != 0;

  const WORD group_count = ::GetActiveProcessorGroupCount();
  if (group_count == 0 || group_count > 64) {
    if (selected != selected_stack) delete[] selected;
    if (storage != stack) delete[] storage;
    return std::unexpected(
        Status::Unavailable("active processor group count is invalid"));
  }
  unsigned group_begin[65]{};
  for (WORD group = 0; group < group_count; ++group) {
    group_begin[group + 1u] =
        group_begin[group] + ::GetActiveProcessorCount(group);
  }
  auto intersects = [&](const GROUP_AFFINITY& mask) noexcept {
    if (mask.Group >= group_count) return false;
    for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8u; ++bit) {
      if ((mask.Mask & (KAFFINITY{1} << bit)) == 0) continue;
      const std::size_t id =
          static_cast<std::size_t>(group_begin[mask.Group]) + bit;
      if (id < selected_count && selected[id] != 0) return true;
    }
    return false;
  };

  DWORD bytes = 0;
  const BOOL sized =
      ::GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
  const DWORD size_code = ::GetLastError();
  if (sized) {
    if (selected != selected_stack) delete[] selected;
    if (storage != stack) delete[] storage;
    return std::unexpected(Status::DataLoss(
        "CPU topology sizing unexpectedly returned relation data"));
  }
  if (size_code != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
    if (selected != selected_stack) delete[] selected;
    if (storage != stack) delete[] storage;
    if (size_code == ERROR_INSUFFICIENT_BUFFER) {
      return std::unexpected(
          Status::DataLoss("CPU topology relation size is zero"));
    }
    return std::unexpected(cpu_query_status(
        "GetLogicalProcessorInformationEx(size)", size_code));
  }
  auto* relations = new (std::nothrow) std::byte[bytes];
  if (relations == nullptr) {
    if (selected != selected_stack) delete[] selected;
    if (storage != stack) delete[] storage;
    return std::unexpected(Status::ResourceExhausted(
        "CPU topology relation allocation failed"));
  }
  auto* first = reinterpret_cast<
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(relations);
  if (!::GetLogicalProcessorInformationEx(RelationAll, first, &bytes)) {
    const DWORD code = ::GetLastError();
    delete[] relations;
    if (selected != selected_stack) delete[] selected;
    if (storage != stack) delete[] storage;
    return std::unexpected(
        cpu_query_status("GetLogicalProcessorInformationEx", code));
  }

  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  for (DWORD offset = 0; offset + kRelationHeader <= bytes;) {
    const auto* relation = reinterpret_cast<
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(relations + offset);
    if (relation->Size < kRelationHeader || relation->Size > bytes - offset) {
      delete[] relations;
      if (selected != selected_stack) delete[] selected;
      if (storage != stack) delete[] storage;
      return std::unexpected(
          Status::DataLoss("CPU topology relation record is malformed"));
    }
    bool visible = false;
    if (relation->Relationship == RelationProcessorCore ||
        relation->Relationship == RelationProcessorPackage) {
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
          offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
      const auto masks = relation->Processor.GroupCount;
      if (relation->Size < header ||
          masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        delete[] relations;
        if (selected != selected_stack) delete[] selected;
        if (storage != stack) delete[] storage;
        return std::unexpected(
            Status::DataLoss("CPU processor relation is malformed"));
      }
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->Processor.GroupMask[group]);
      }
      if (visible && relation->Relationship == RelationProcessorCore) {
        ++out.physical_cores;
      } else if (visible) {
        ++out.packages;
      }
    } else if (relation->Relationship == RelationNumaNode ||
               relation->Relationship == RelationNumaNodeEx) {
      const WORD masks = relation->NumaNode.GroupCount == 0
                             ? 1
                             : relation->NumaNode.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, NumaNode) +
          offsetof(NUMA_NODE_RELATIONSHIP, GroupMasks);
      if (relation->Size < header ||
          masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        delete[] relations;
        if (selected != selected_stack) delete[] selected;
        if (storage != stack) delete[] storage;
        return std::unexpected(
            Status::DataLoss("CPU NUMA relation is malformed"));
      }
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->NumaNode.GroupMasks[group]);
      }
      if (visible) ++out.numa_nodes;
    } else if (relation->Relationship == RelationCache &&
               (relation->Cache.Level == 2 || relation->Cache.Level == 3)) {
      const WORD masks = relation->Cache.GroupCount == 0
                             ? 1
                             : relation->Cache.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
          offsetof(CACHE_RELATIONSHIP, GroupMasks);
      if (relation->Size < header ||
          masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        delete[] relations;
        if (selected != selected_stack) delete[] selected;
        if (storage != stack) delete[] storage;
        return std::unexpected(
            Status::DataLoss("CPU cache relation is malformed"));
      }
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->Cache.GroupMasks[group]);
      }
      if (visible && relation->Cache.Level == 2) ++out.l2_cache_domains;
      else if (visible) ++out.l3_cache_domains;
    }
    offset += relation->Size;
  }
  delete[] relations;
  if (selected != selected_stack) delete[] selected;
  if (storage != stack) delete[] storage;
  return out;
}
#endif

}  // namespace

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures f = detect();  // Meyers: thread-safe init-once (C++11+)
  return f;
}

unsigned cache_line_size() noexcept {
  static const unsigned v = detect_cache_line();  // Meyers: probe once, cache
  return v;
}

const CacheInfo& cache_info() noexcept {
  static const CacheInfo info = detect_cache_info();
  return info;
}

Result<std::size_t> available_cpu_ids(unsigned* dst,
                                      std::size_t cap) noexcept {
  return query_available_cpu_ids(dst, cap);
}

Result<std::size_t> available_cpu_info(CpuInfo* dst,
                                       std::size_t cap) noexcept {
  return query_available_cpu_info(dst, cap);
}

Result<std::size_t> available_cpu_ids(CpuOrder order, unsigned* dst,
                                      std::size_t cap) noexcept {
  if (order == CpuOrder::kOsDefault) return query_available_cpu_ids(dst, cap);
  if (order != CpuOrder::kPerformanceFirst &&
      order != CpuOrder::kEfficiencyFirst) {
    return std::unexpected(Status::InvalidArgument("CPU order is invalid"));
  }
  if (dst == nullptr) return query_available_cpu_ids(nullptr, 0);

  for (unsigned attempt = 0; attempt < 4; ++attempt) {
    auto count_result = query_available_cpu_info(nullptr, 0);
    if (!count_result) return std::unexpected(std::move(count_result).error());
    const std::size_t capacity = *count_result;
    CpuInfo stack[256];
    CpuInfo* infos = capacity <= 256
                         ? stack
                         : new (std::nothrow) CpuInfo[capacity];
    if (infos == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "ordered available CPU allocation failed"));
    }
    auto filled = query_available_cpu_info(infos, capacity);
    if (!filled) {
      if (infos != stack) delete[] infos;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled > capacity) {
      if (infos != stack) delete[] infos;
      continue;
    }

    const std::size_t count = *filled;
    bool have_different_known_ranks = false;
    bool have_unknown_rank = false;
    unsigned first_rank = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (infos[i].performance_rank == 0) {
        have_unknown_rank = true;
        continue;
      }
      if (first_rank == 0) first_rank = infos[i].performance_rank;
      if (infos[i].performance_rank != first_rank) {
        have_different_known_ranks = true;
      }
    }
    const bool have_known_rank = first_rank != 0;
    if (have_different_known_ranks || (have_known_rank && have_unknown_rank)) {
      std::sort(infos, infos + count, [order](const CpuInfo& a,
                                             const CpuInfo& b) noexcept {
        const bool a_known = a.performance_rank != 0;
        const bool b_known = b.performance_rank != 0;
        if (a_known != b_known) return a_known;  // unknown is never guessed first
        if (a.performance_rank != b.performance_rank) {
          return order == CpuOrder::kPerformanceFirst
                     ? a.performance_rank > b.performance_rank
                     : a.performance_rank < b.performance_rank;
        }
        return a.id < b.id;
      });
    }
    const std::size_t written = std::min(cap, count);
    for (std::size_t i = 0; i < written; ++i) dst[i] = infos[i].id;
    if (infos != stack) delete[] infos;
    return count;
  }
  return std::unexpected(Status::Unavailable(
      "available CPU order changed repeatedly during snapshot"));
}

Result<std::size_t> available_cpu_count() noexcept {
  return query_available_cpu_ids(nullptr, 0);
}

Result<CpuTopology> available_cpu_topology() noexcept {
#if HYLE_OS_WINDOWS
  // Aggregates can be counted directly from the visibility set and native
  // relation records. Building a complete CpuInfo array and hashing its domain
  // ids is necessary for the per-CPU API, but needless work for this value API.
  return query_windows_available_cpu_topology();
#else
  CpuInfo stack[256];
  CpuInfo* info = stack;
  std::size_t capacity = std::size(stack);
  auto filled = query_available_cpu_info(info, capacity);
  if (!filled) {
    return std::unexpected(std::move(filled).error());
  }
  if (*filled > capacity) {
    capacity = *filled;
    if (capacity > (std::numeric_limits<unsigned>::max)()) {
      return std::unexpected(
          Status::OutOfRange("available CPU count exceeds topology limits"));
    }
    info = new (std::nothrow) CpuInfo[capacity];
    if (info == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("CPU topology allocation failed"));
    }
    filled = query_available_cpu_info(info, capacity);
    if (!filled) {
      delete[] info;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled > capacity) {
      delete[] info;
      return std::unexpected(Status::Unavailable(
          "available CPU set expanded during topology query"));
    }
  }
  const std::size_t count = *filled;
  std::size_t distinct_capacity = 1;
  const std::size_t needed_capacity = count * 2u;
  while (distinct_capacity < needed_capacity) {
    if (distinct_capacity >
        (std::numeric_limits<std::size_t>::max)() / 2u) {
      if (info != stack) delete[] info;
      return std::unexpected(
          Status::OutOfRange("CPU topology set capacity overflows"));
    }
    distinct_capacity *= 2u;
  }
  unsigned distinct_stack[512];
  unsigned* distinct = distinct_capacity <= std::size(distinct_stack)
                           ? distinct_stack
                           : new (std::nothrow) unsigned[distinct_capacity];
  if (distinct == nullptr) {
    if (info != stack) delete[] info;
    return std::unexpected(
        Status::ResourceExhausted("CPU topology set allocation failed"));
  }
  CpuTopology out;
  out.logical_cpus = static_cast<unsigned>(count);
  out.physical_cores = count_distinct(info, count, &CpuInfo::core_id,
                                      distinct, distinct_capacity);
  out.packages = count_distinct(info, count, &CpuInfo::package_id,
                                distinct, distinct_capacity);
  out.numa_nodes = count_distinct(info, count, &CpuInfo::numa_node,
                                  distinct, distinct_capacity);
  out.l2_cache_domains =
      count_distinct(info, count, &CpuInfo::l2_cache_domain, distinct,
                     distinct_capacity);
  out.l3_cache_domains =
      count_distinct(info, count, &CpuInfo::l3_cache_domain, distinct,
                     distinct_capacity);
  for (std::size_t i = 0; i < count; ++i) {
    if (info[i].kind == CpuKind::kPerformance) ++out.performance_cpus;
    else if (info[i].kind == CpuKind::kEfficiency) ++out.efficiency_cpus;
  }
  out.hybrid = out.performance_cpus != 0 && out.efficiency_cpus != 0;
  if (distinct != distinct_stack) delete[] distinct;
  if (info != stack) delete[] info;

#if HYLE_OS_MACOS
  // macOS reports trustworthy aggregate physical/performance counts but no
  // public logical-id affinity map. Preserve those facts without inventing a
  // CpuInfo mapping.
  auto sysctl_unsigned = [](const char* name, unsigned& value) noexcept {
    std::size_t size = sizeof(value);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0;
  };
  unsigned physical = 0;
  if (sysctl_unsigned("hw.physicalcpu", physical)) {
    out.physical_cores = physical;
  }
  unsigned performance = 0, efficiency = 0;
  if (sysctl_unsigned("hw.perflevel0.logicalcpu", performance) &&
      sysctl_unsigned("hw.perflevel1.logicalcpu", efficiency) &&
      performance + efficiency == out.logical_cpus) {
    out.performance_cpus = performance;
    out.efficiency_cpus = efficiency;
    out.hybrid = performance != 0 && efficiency != 0;
  }
  out.packages = out.logical_cpus == 0 ? 0 : 1;
  out.numa_nodes = out.logical_cpus == 0 ? 0 : 1;
#endif
  return out;
#endif
}

unsigned hardware_concurrency() noexcept {
#if HYLE_OS_WINDOWS
  const DWORD all = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  if (all != 0) return static_cast<unsigned>(all);
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  return info.dwNumberOfProcessors != 0 ? info.dwNumberOfProcessors : 1u;
#else
  const long count = ::sysconf(_SC_NPROCESSORS_ONLN);
  return count > 0 ? static_cast<unsigned>(count) : 1u;
#endif
}

Isa best_isa() noexcept {
  static const Isa isa = compute_best(cpu_features());
  return isa;
}

bool supports(Isa isa) noexcept {
  if (isa == Isa::kScalar) return true;
#if HYLE_ARCH_X64
  if (isa < Isa::kSse2 || isa > Isa::kAvx512) return false;
#elif HYLE_ARCH_ARM64
  if (isa != Isa::kNeon && isa != Isa::kSve) return false;
#elif HYLE_ARCH_RISCV64
  if (isa != Isa::kRvv) return false;
#else
  return false;
#endif
  return static_cast<std::uint32_t>(best_isa()) >= static_cast<std::uint32_t>(isa);
}

const char* isa_name(Isa isa) noexcept {
  switch (isa) {
    case Isa::kScalar: return "scalar";
    case Isa::kSse2: return "sse2";
    case Isa::kSsse3: return "ssse3";
    case Isa::kSse42: return "sse4.2";
    case Isa::kAvx: return "avx";
    case Isa::kAvx2: return "avx2";
    case Isa::kAvx512: return "avx512";
    case Isa::kNeon: return "neon";
    case Isa::kSve: return "sve";
    case Isa::kRvv: return "rvv";
  }
  return "unknown";
}

}  // namespace hyle::platform
