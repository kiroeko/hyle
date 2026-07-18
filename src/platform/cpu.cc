#include "hyle/platform/cpu.h"

// cpu.cc -- the ISOLATED, LTO-DISABLED probe TU (conventions §10 probe isolation).
// Nothing here may use an instruction above the baseline ISA: this code runs FIRST, on
// an as-yet-unknown CPU, to discover what that CPU can run. If LTO inlined an AVX
// instruction into FlagsFromCpuid(), an old CPU would SIGILL inside the very function
// meant to tell us it's old. The CMake for this TU disables LTO/IPO for that reason.
// We compile it at the default baseline (no -mavx etc.), so the compiler cannot emit
// anything beyond it here.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

#include "hyle/platform/thread.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#if defined(HYLE_OS_LINUX) && (defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64))
#include <sys/auxv.h>
#endif

#if defined(HYLE_OS_MACOS)
#include <sys/sysctl.h>  // sysctlbyname("hw.cachelinesize")
#endif
#if defined(HYLE_OS_LINUX)
#include <unistd.h>  // sysconf(_SC_LEVEL1_DCACHE_LINESIZE)
#endif

#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_RISCV64)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(HYLE_ARCH_X64)
#  if defined(HYLE_COMPILER_MSVC)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

namespace hyle::platform {
namespace {

#if defined(HYLE_ARCH_X64)

// CPUID(leaf, subleaf) -> {eax, ebx, ecx, edx} in abcd. Wraps the two compiler ABIs
// (Highway x86_cpuid.h does the same): MSVC's __cpuidex vs GCC/Clang's __cpuid_count.
void cpuid(std::uint32_t leaf, std::uint32_t subleaf,
           std::uint32_t abcd[4]) noexcept {
#  if defined(HYLE_COMPILER_MSVC)
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
#  if defined(HYLE_COMPILER_MSVC)
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

    cpuid(7, 1, abcd);
    // AVX-VNNI (VEX encoding, 256-bit): leaf 7,1 EAX bit 4 — independent of AVX-512.
    f.avx_vnni = bit(abcd[0], 4);
    f.avx512bf16 = bit(abcd[0], 5);
    // AVX10: leaf 7,1 EDX bit 19 advertises AVX10 support; the converged version is in leaf 0x24
    // subleaf 0, EBX[7:0]. AVX10.2 (the fp8 vcvtph2hf8/bf8 converts) = version >= 2.
    const bool avx10_supported = bit(abcd[3], 19);
    if (avx10_supported && max_leaf >= 0x24) {
      cpuid(0x24u, 0, abcd);
      cpu_avx10_2 = (abcd[1] & 0xFFu) >= 2;  // EBX[7:0] = AVX10 converged version
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

  return f;
}

#endif  // HYLE_ARCH_X64

#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_RISCV64)

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
#if defined(HYLE_ARCH_X64)
  return detect_x86();
#elif defined(HYLE_ARCH_ARM64)
  CpuFeatures f;
  f.neon = true;  // mandatory on aarch64
#if defined(HYLE_OS_LINUX)
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
#endif
  return f;
#elif defined(HYLE_ARCH_RISCV64)
  CpuFeatures f;
#if defined(HYLE_OS_LINUX)
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
#if defined(HYLE_ARCH_X64)
  // Mirror Highway's group gating (targets.cc kGroupAVX2 / kGroupAVX3): a tier needs
  // ALL of its group's features, AND each tier's group SUBSUMES the lower tier's, so
  // a higher Isa genuinely implies every lower one (the enum's contract).
  // kGroupAVX2 = AVX2 + FMA + F16C + LZCNT + BMI1 + BMI2 (+ SSE4 below it).
  const bool avx2_group =
      f.avx2 && f.fma && f.f16c && f.lzcnt && f.bmi1 && f.bmi2 && f.sse42;
  // kGroupAVX3 = AVX-512 F+BW+VL+DQ+CD on TOP of the full AVX2 group.
  const bool avx512_group = avx2_group && f.avx512f && f.avx512bw && f.avx512vl &&
                            f.avx512dq && f.avx512cd;
  if (avx512_group) return Isa::kAvx512;
  if (avx2_group) return Isa::kAvx2;
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
#elif defined(HYLE_ARCH_ARM64)
  if (f.sve) return Isa::kSve;
  if (f.neon) return Isa::kNeon;
  return Isa::kScalar;
#elif defined(HYLE_ARCH_RISCV64)
  return f.rvv ? Isa::kRvv : Isa::kScalar;
#else
  (void)f;
  return Isa::kScalar;
#endif
}

// L1 data cache line size in bytes. x86: CPUID leaf 1 EBX[15:8] (CLFLUSH line size
// in 8-byte units). macOS ARM: sysctl hw.cachelinesize. Linux: sysconf. 64 fallback.
// Compiled in this LTO-off baseline TU so the x86 CPUID path is SIGILL-safe.
unsigned detect_cache_line() noexcept {
#if defined(HYLE_ARCH_X64)
  std::uint32_t abcd[4];
  cpuid(1, 0, abcd);
  const unsigned line = ((abcd[1] >> 8) & 0xffu) * 8u;
  return line ? line : 64u;
#elif defined(HYLE_OS_MACOS)
  std::size_t v = 0;
  std::size_t sz = sizeof(v);
  if (::sysctlbyname("hw.cachelinesize", &v, &sz, nullptr, 0) == 0 && v != 0) {
    return static_cast<unsigned>(v);
  }
  return 64u;
#elif defined(HYLE_OS_LINUX)
  const long v = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  return v > 0 ? static_cast<unsigned>(v) : 64u;
#else
  return 64u;
#endif
}

CacheInfo detect_cache_info() noexcept {
  CacheInfo out;
#if defined(HYLE_ARCH_X64)
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
#elif defined(HYLE_OS_MACOS)
  auto get = [](const char* name) noexcept {
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
  };
  out.l1d = get("hw.l1dcachesize");
  out.l1i = get("hw.l1icachesize");
  out.l2 = get("hw.l2cachesize");
  out.l3 = get("hw.l3cachesize");
#elif defined(HYLE_OS_LINUX)
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

#if defined(HYLE_OS_LINUX)
bool read_sysfs_unsigned(const char* path, unsigned& value) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (!file) return false;
  const bool ok = std::fscanf(file, "%u", &value) == 1;
  std::fclose(file);
  return ok;
}
#endif

CpuTopology detect_topology() noexcept {
  CpuTopology out;
  out.logical_cpus = hardware_concurrency();
  if (out.logical_cpus == 0) out.logical_cpus = 1;
  out.performance_cpus = out.logical_cpus;

#if defined(HYLE_OS_WINDOWS)
  DWORD bytes = 0;
  (void)::GetSystemCpuSetInformation(nullptr, 0, &bytes,
                                     ::GetCurrentProcess(), 0);
  auto* storage = new (std::nothrow) unsigned char[bytes];
  if (!storage || bytes == 0) {
    delete[] storage;
    return out;
  }
  if (!::GetSystemCpuSetInformation(
          reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage), bytes,
          &bytes, ::GetCurrentProcess(), 0)) {
    delete[] storage;
    return out;
  }
  unsigned min_class = 255, max_class = 0, count = 0;
  for (DWORD offset = 0; offset < bytes;) {
    auto* info =
        reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage + offset);
    if (info->Size == 0 || info->Size > bytes - offset) break;
    if (info->Type == CpuSetInformation) {
      const unsigned cls = info->CpuSet.EfficiencyClass;
      if (cls < min_class) min_class = cls;
      if (cls > max_class) max_class = cls;
      ++count;
    }
    offset += info->Size;
  }
  if (count == out.logical_cpus && min_class < max_class) {
    unsigned performance = 0;
    for (DWORD offset = 0; offset < bytes;) {
      auto* info =
          reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage + offset);
      if (info->Size == 0 || info->Size > bytes - offset) break;
      if (info->Type == CpuSetInformation &&
          info->CpuSet.EfficiencyClass == max_class) {
        ++performance;
      }
      offset += info->Size;
    }
    if (performance != 0 && performance != count) {
      out.performance_cpus = performance;
      out.efficiency_cpus = count - performance;
      out.hybrid = true;
    }
  }
  delete[] storage;
#elif defined(HYLE_OS_MACOS)
  int levels = 0;
  std::size_t levels_size = sizeof(levels);
  if (::sysctlbyname("hw.nperflevels", &levels, &levels_size, nullptr, 0) ==
          0 &&
      levels > 1) {
    unsigned performance = 0, efficiency = 0;
    for (int level = 0; level < levels; ++level) {
      char name[48];
      std::snprintf(name, sizeof(name), "hw.perflevel%d.logicalcpu", level);
      unsigned count = 0;
      std::size_t size = sizeof(count);
      if (::sysctlbyname(name, &count, &size, nullptr, 0) != 0) {
        performance = efficiency = 0;
        break;
      }
      if (level == 0) performance += count;
      else efficiency += count;
    }
    if (performance != 0 && efficiency != 0 &&
        performance + efficiency == out.logical_cpus) {
      out.performance_cpus = performance;
      out.efficiency_cpus = efficiency;
      out.hybrid = true;
    }
  }
#elif defined(HYLE_OS_LINUX)
  // Linux exposes scheduler capacity on heterogeneous ARM systems. Only trust
  // it when every process-visible logical CPU has a value and capacities differ.
  unsigned capacities[4096];
  unsigned found = 0, min_capacity = ~0u, max_capacity = 0;
  for (unsigned cpu = 0; cpu < 4096 && found < out.logical_cpus; ++cpu) {
    char path[96];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/online", cpu);
    unsigned online = 1;
    if (read_sysfs_unsigned(path, online) && online == 0) continue;
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/cpu_capacity", cpu);
    unsigned capacity = 0;
    if (!read_sysfs_unsigned(path, capacity)) continue;
    capacities[found++] = capacity;
    if (capacity < min_capacity) min_capacity = capacity;
    if (capacity > max_capacity) max_capacity = capacity;
  }
  if (found == out.logical_cpus && min_capacity < max_capacity) {
    unsigned performance = 0;
    for (unsigned i = 0; i < found; ++i) {
      if (capacities[i] == max_capacity) ++performance;
    }
    if (performance != 0 && performance != found) {
      out.performance_cpus = performance;
      out.efficiency_cpus = found - performance;
      out.hybrid = true;
    }
  }
#endif
  return out;
}

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

const CpuTopology& cpu_topology() noexcept {
  static const CpuTopology topology = detect_topology();
  return topology;
}

Isa best_isa() noexcept {
  static const Isa isa = compute_best(cpu_features());
  return isa;
}

bool supports(Isa isa) noexcept {
  if (isa == Isa::kScalar) return true;
#if defined(HYLE_ARCH_X64)
  if (isa < Isa::kSse2 || isa > Isa::kAvx512) return false;
#elif defined(HYLE_ARCH_ARM64)
  if (isa != Isa::kNeon && isa != Isa::kSve) return false;
#elif defined(HYLE_ARCH_RISCV64)
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
