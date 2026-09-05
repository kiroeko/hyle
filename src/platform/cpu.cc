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
#include <bit>
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

CpuFeatureSet detect_x86() noexcept {
  detail::X86FeatureRegisters input;
  std::uint32_t abcd[4];

  cpuid(0, 0, abcd);
  const std::uint32_t max_leaf = abcd[0];

  // Leaf 1: legacy + AVX/F16C/OSXSAVE.
  cpuid(1, 0, abcd);
  input.leaf1_ecx = abcd[2];

  // Leaf 0x80000001 (extended): LZCNT.
  cpuid(0x80000000u, 0, abcd);  // max EXTENDED leaf
  if (abcd[0] >= 0x80000001u) {
    cpuid(0x80000001u, 0, abcd);
    input.extended1_ecx = abcd[2];
  }

  // Leaf 7,0 and 7,1: AVX2/BMI, AVX-512, AMX and AVX10 advertisement.
  if (max_leaf >= 7) {
    cpuid(7, 0, abcd);
    const std::uint32_t max_subleaf7 = abcd[0];
    input.leaf7_0_ebx = abcd[1];
    input.leaf7_0_ecx = abcd[2];
    input.leaf7_0_edx = abcd[3];

    if (max_subleaf7 >= 1) {
      cpuid(7, 1, abcd);
      input.leaf7_1_eax = abcd[0];
      input.leaf7_1_edx = abcd[3];
      // AVX10: leaf 7,1 EDX bit 19 advertises the feature; leaf 0x24
      // reports its version and supported vector widths.
      if (detail::feature_bit(input.leaf7_1_edx, 19) &&
          max_leaf >= 0x24) {
        cpuid(0x24u, 0, abcd);
        input.leaf24_ebx = abcd[1];
      }
    }
  }

  // XGETBV is legal only after OSXSAVE. The pure decoder applies the YMM/ZMM
  // gates uniformly to every extension that can require that state.
  if (detail::feature_bit(input.leaf1_ecx, 27)) {
    input.xcr0 = read_xcr0();
#  if HYLE_OS_MACOS
    // macOS may lazily leave the ZMM XCR0 bits clear until first use. Apple
    // documents hw.optional.avx512f as the process-availability query (and it
    // is false under Rosetta, which cannot execute AVX-512), so let a
    // successful query authoritatively set or clear the ZMM state bits.
    int avx512_available = 0;
    std::size_t size = sizeof(avx512_available);
    if (::sysctlbyname("hw.optional.avx512f", &avx512_available, &size,
                       nullptr, 0) == 0) {
      constexpr std::uint32_t kZmmState = (1u << 5) | (1u << 6) | (1u << 7);
      if (avx512_available != 0)
        input.xcr0 |= kZmmState;
      else
        input.xcr0 &= ~kZmmState;
    }
#  endif
  }
  return detail::decode_x86_features(input);
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
                                   CpuFeatureSet& features) noexcept {
  const bool vector = features.has(CpuFeature::kRiscvV) ||
                      (bits & kRiscvExtV) != 0;
  features.set(CpuFeature::kRiscvV, vector);
  if (!vector) return;
  detail::set_if(features, CpuFeature::kRiscvZvbb,
                 (bits & kRiscvExtZvbb) != 0);
  detail::set_if(features, CpuFeature::kRiscvZvbc,
                 (bits & kRiscvExtZvbc) != 0);
  detail::set_if(features, CpuFeature::kRiscvZvkned,
                 (bits & kRiscvExtZvkned) != 0);
  detail::set_if(features, CpuFeature::kRiscvZvfh,
                 (bits & kRiscvExtZvfh) != 0);
  detail::set_if(features, CpuFeature::kRiscvZvfhmin,
                 (bits & kRiscvExtZvfhmin) != 0);
}

// Compile-time synthetic decoding keeps every stable Linux UABI bit covered
// even when qemu-linux-user cannot emulate riscv_hwprobe optional extensions.
static_assert([] {
  CpuFeatureSet f;
  apply_riscv_hwprobe(kRiscvExtV | kRiscvExtZvbb | kRiscvExtZvbc |
                          kRiscvExtZvkned | kRiscvExtZvfh |
                          kRiscvExtZvfhmin,
                      f);
  return f.has_all(CpuFeatureSet{
      CpuFeature::kRiscvV, CpuFeature::kRiscvZvbb,
      CpuFeature::kRiscvZvbc, CpuFeature::kRiscvZvkned,
      CpuFeature::kRiscvZvfh, CpuFeature::kRiscvZvfhmin});
}());
static_assert([] {
  CpuFeatureSet f;
  apply_riscv_hwprobe(kRiscvExtZvbb | kRiscvExtZvbc, f);
  return !f.has_any(CpuFeatureSet{
      CpuFeature::kRiscvV, CpuFeature::kRiscvZvbb,
      CpuFeature::kRiscvZvbc});
}());

#endif

CpuFeatureSet detect() noexcept {
#if HYLE_ARCH_X64
  return detect_x86();
#elif HYLE_ARCH_ARM64
#if HYLE_OS_LINUX
  // Keep numeric bits in cpu_internal.h instead of relying on a new sysroot.
  constexpr unsigned long kAtHwcap2 = 26;
  return detail::decode_arm_linux_features(
      static_cast<std::uint64_t>(getauxval(AT_HWCAP)),
      static_cast<std::uint64_t>(getauxval(kAtHwcap2)));
#elif HYLE_OS_MACOS
  CpuFeatureSet f{CpuFeature::kArmNeon};
  auto sysctl_feature = [](const char* name) noexcept {
    int present = 0;
    std::size_t size = sizeof(present);
    return ::sysctlbyname(name, &present, &size, nullptr, 0) == 0 &&
           present != 0;
  };
  const bool crypto = sysctl_feature("hw.optional.arm.FEAT_AES");
  f.set(CpuFeature::kArmAes, crypto);
  // Apple exposes FEAT_AES as the Arm AES/PMULL extension group.
  f.set(CpuFeature::kArmPmull, crypto);
  f.set(CpuFeature::kArmDotProd,
        sysctl_feature("hw.optional.arm.FEAT_DotProd"));
  f.set(CpuFeature::kArmFp16,
        sysctl_feature("hw.optional.arm.FEAT_FP16"));
  f.set(CpuFeature::kArmI8mm,
        sysctl_feature("hw.optional.arm.FEAT_I8MM"));
  f.set(CpuFeature::kArmBf16,
        sysctl_feature("hw.optional.arm.FEAT_BF16"));
  const bool sve = sysctl_feature("hw.optional.arm.FEAT_SVE");
  const bool sve2 = sve && sysctl_feature("hw.optional.arm.FEAT_SVE2");
  f.set(CpuFeature::kArmSve, sve);
  f.set(CpuFeature::kArmSve2, sve2);
  f.set(CpuFeature::kArmSveAes,
        sve2 && sysctl_feature("hw.optional.arm.FEAT_SVE_AES"));
  f.set(CpuFeature::kArmSvePmull,
        sve2 && sysctl_feature("hw.optional.arm.FEAT_SVE_PMULL128"));
  f.set(CpuFeature::kArmSveI8mm,
        sve && sysctl_feature("hw.optional.arm.FEAT_SVE_I8MM"));
  f.set(CpuFeature::kArmSveF32mm,
        sve && sysctl_feature("hw.optional.arm.FEAT_SVE_F32MM"));
  f.set(CpuFeature::kArmSveF64mm,
        sve && sysctl_feature("hw.optional.arm.FEAT_SVE_F64MM"));
  f.set(CpuFeature::kArmSveBf16,
        sve && sysctl_feature("hw.optional.arm.FEAT_SVE_BF16"));
  const bool sme = sysctl_feature("hw.optional.arm.FEAT_SME");
  f.set(CpuFeature::kArmSme, sme);
  f.set(CpuFeature::kArmSme2,
        sme && sysctl_feature("hw.optional.arm.FEAT_SME2"));
  return f;
#elif HYLE_OS_WINDOWS
  CpuFeatureSet f{CpuFeature::kArmNeon};
  // Stable Windows processor-feature ABI values. Local numbers keep old SDKs
  // buildable for newer ARM64 hosts.
  constexpr DWORD kPfArmCrypto = 30;
  constexpr DWORD kPfArmDotProd = 43;
  constexpr DWORD kPfArmSve = 46;
  constexpr DWORD kPfArmSve2 = 47;
  constexpr DWORD kPfArmSveAes = 49;
  constexpr DWORD kPfArmSvePmull = 50;
  constexpr DWORD kPfArmSveBf16 = 52;
  constexpr DWORD kPfArmSveI8mm = 57;
  constexpr DWORD kPfArmSveF32mm = 58;
  constexpr DWORD kPfArmSveF64mm = 59;
  constexpr DWORD kPfArmI8mm = 66;
  constexpr DWORD kPfArmFp16 = 67;
  constexpr DWORD kPfArmBf16 = 68;
  constexpr DWORD kPfArmSme = 70;
  constexpr DWORD kPfArmSme2 = 71;
  const auto present = [](DWORD feature) noexcept {
    return ::IsProcessorFeaturePresent(feature) != 0;
  };
  const bool crypto = present(kPfArmCrypto);
  f.set(CpuFeature::kArmAes, crypto);
  f.set(CpuFeature::kArmPmull, crypto);
  f.set(CpuFeature::kArmDotProd, present(kPfArmDotProd));
  f.set(CpuFeature::kArmFp16, present(kPfArmFp16));
  f.set(CpuFeature::kArmI8mm, present(kPfArmI8mm));
  f.set(CpuFeature::kArmBf16, present(kPfArmBf16));
  const bool sve = present(kPfArmSve);
  const bool sve2 = sve && present(kPfArmSve2);
  f.set(CpuFeature::kArmSve, sve);
  f.set(CpuFeature::kArmSve2, sve2);
  f.set(CpuFeature::kArmSveAes, sve2 && present(kPfArmSveAes));
  f.set(CpuFeature::kArmSvePmull, sve2 && present(kPfArmSvePmull));
  f.set(CpuFeature::kArmSveI8mm, sve && present(kPfArmSveI8mm));
  f.set(CpuFeature::kArmSveF32mm, sve && present(kPfArmSveF32mm));
  f.set(CpuFeature::kArmSveF64mm, sve && present(kPfArmSveF64mm));
  f.set(CpuFeature::kArmSveBf16, sve && present(kPfArmSveBf16));
  const bool sme = present(kPfArmSme);
  f.set(CpuFeature::kArmSme, sme);
  f.set(CpuFeature::kArmSme2, sme && present(kPfArmSme2));
  return f;
#else
  return CpuFeatureSet{CpuFeature::kArmNeon};
#endif
#elif HYLE_ARCH_RISCV64
  CpuFeatureSet f;
#if HYLE_OS_LINUX
  constexpr unsigned long kHwcapIsaV = 1ul << ('V' - 'A');
  f.set(CpuFeature::kRiscvV,
        (getauxval(AT_HWCAP) & kHwcapIsaV) != 0);

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
  return CpuFeatureSet{};  // scalar-only platform
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

const CpuFeatureSet& cpu_features() noexcept {
  static const CpuFeatureSet f = detect();  // thread-safe init-once
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

namespace {

#if !HYLE_OS_MACOS
struct CpuCacheObservation {
  CpuInfo cpu{};
  CacheInfo cache{};
  unsigned line_size = 0;
  unsigned cpus_per_l2 = 0;
  unsigned cpus_per_l3 = 0;
};

bool same_cache_profile(const CpuCacheObservation& left,
                        const CpuCacheObservation& right) noexcept {
  return left.cpu.performance_rank == right.cpu.performance_rank &&
         left.cpu.kind == right.cpu.kind &&
         left.line_size == right.line_size &&
         left.cpus_per_l2 == right.cpus_per_l2 &&
         left.cpus_per_l3 == right.cpus_per_l3 &&
         left.cache.l1d == right.cache.l1d &&
         left.cache.l1i == right.cache.l1i &&
         left.cache.l2 == right.cache.l2 &&
         left.cache.l3 == right.cache.l3;
}

bool cache_profile_less(const CpuCacheObservation& left,
                        const CpuCacheObservation& right) noexcept {
  if (left.cpu.performance_rank != right.cpu.performance_rank) {
    return left.cpu.performance_rank > right.cpu.performance_rank;
  }
  if (left.cpu.kind != right.cpu.kind) {
    return static_cast<unsigned>(left.cpu.kind) >
           static_cast<unsigned>(right.cpu.kind);
  }
#define HYLE_CACHE_PROFILE_DESCENDING(field) \
  if (left.field != right.field) return left.field > right.field
  HYLE_CACHE_PROFILE_DESCENDING(cache.l1d);
  HYLE_CACHE_PROFILE_DESCENDING(cache.l1i);
  HYLE_CACHE_PROFILE_DESCENDING(cache.l2);
  HYLE_CACHE_PROFILE_DESCENDING(cache.l3);
  HYLE_CACHE_PROFILE_DESCENDING(line_size);
  HYLE_CACHE_PROFILE_DESCENDING(cpus_per_l2);
  HYLE_CACHE_PROFILE_DESCENDING(cpus_per_l3);
#undef HYLE_CACHE_PROFILE_DESCENDING
  if (left.cpu.core_id != right.cpu.core_id) {
    return left.cpu.core_id < right.cpu.core_id;
  }
  return left.cpu.id < right.cpu.id;
}

#if HYLE_OS_WINDOWS
void fill_cache_observations(CpuCacheObservation* observations,
                             std::size_t count) noexcept {
  if (count == 0) return;
  unsigned maximum_id = 0;
  for (std::size_t i = 0; i < count; ++i) {
    maximum_id = (std::max)(maximum_id, observations[i].cpu.id);
  }
  const std::size_t lookup_count = static_cast<std::size_t>(maximum_id) + 1u;
  CpuCacheObservation* lookup_stack[256];
  CpuCacheObservation** lookup =
      lookup_count <= std::size(lookup_stack)
          ? lookup_stack
          : new (std::nothrow) CpuCacheObservation*[lookup_count];
  if (lookup == nullptr) return;
  std::fill(lookup, lookup + lookup_count, nullptr);
  for (std::size_t i = 0; i < count; ++i) {
    lookup[observations[i].cpu.id] = observations + i;
  }

  const WORD group_count = ::GetActiveProcessorGroupCount();
  unsigned group_begin_stack[65]{};
  unsigned* group_begin =
      group_count < std::size(group_begin_stack)
          ? group_begin_stack
          : new (std::nothrow) unsigned[static_cast<std::size_t>(group_count) +
                                         1u];
  if (group_begin == nullptr) {
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }
  for (WORD group = 0; group < group_count; ++group) {
    group_begin[group + 1u] =
        group_begin[group] + ::GetActiveProcessorCount(group);
  }

  DWORD bytes = 0;
  if (::GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes) ||
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
  if (!::GetLogicalProcessorInformationEx(RelationCache, first, &bytes)) {
    delete[] storage;
    if (group_begin != group_begin_stack) delete[] group_begin;
    if (lookup != lookup_stack) delete[] lookup;
    return;
  }

  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  for (DWORD offset = 0; offset + kRelationHeader <= bytes;) {
    const auto* relation = reinterpret_cast<
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(storage + offset);
    if (relation->Size < kRelationHeader || relation->Size > bytes - offset) {
      break;
    }
    if (relation->Relationship != RelationCache) {
      offset += relation->Size;
      continue;
    }
    const WORD masks = relation->Cache.GroupCount == 0
                           ? 1
                           : relation->Cache.GroupCount;
    const std::size_t header =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
        offsetof(CACHE_RELATIONSHIP, GroupMasks);
    if (relation->Size < header ||
        masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
      offset += relation->Size;
      continue;
    }
    unsigned sharing = 0;
    for (WORD group = 0; group < masks; ++group) {
      sharing += static_cast<unsigned>(std::popcount(
          static_cast<std::uint64_t>(relation->Cache.GroupMasks[group].Mask)));
    }
    auto assign = [&](CpuCacheObservation& observation) noexcept {
      observation.line_size =
          (std::max)(observation.line_size,
                     static_cast<unsigned>(relation->Cache.LineSize));
      const auto capacity =
          static_cast<std::uint64_t>(relation->Cache.CacheSize);
      if (relation->Cache.Level == 1) {
        if (relation->Cache.Type == CacheData ||
            relation->Cache.Type == CacheUnified) {
          observation.cache.l1d =
              (std::max)(observation.cache.l1d, capacity);
        }
        if (relation->Cache.Type == CacheInstruction ||
            relation->Cache.Type == CacheUnified) {
          observation.cache.l1i =
              (std::max)(observation.cache.l1i, capacity);
        }
      } else if (relation->Cache.Level == 2) {
        observation.cache.l2 = (std::max)(observation.cache.l2, capacity);
        observation.cpus_per_l2 =
            (std::max)(observation.cpus_per_l2, sharing);
      } else if (relation->Cache.Level == 3) {
        observation.cache.l3 = (std::max)(observation.cache.l3, capacity);
        observation.cpus_per_l3 =
            (std::max)(observation.cpus_per_l3, sharing);
      }
    };
    for (WORD group = 0; group < masks; ++group) {
      const auto& mask = relation->Cache.GroupMasks[group];
      if (mask.Group >= group_count) continue;
      for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8u; ++bit) {
        if ((mask.Mask & (KAFFINITY{1} << bit)) == 0) continue;
        const std::size_t id =
            static_cast<std::size_t>(group_begin[mask.Group]) + bit;
        if (id < lookup_count && lookup[id] != nullptr) assign(*lookup[id]);
      }
    }
    offset += relation->Size;
  }
  delete[] storage;
  if (group_begin != group_begin_stack) delete[] group_begin;
  if (lookup != lookup_stack) delete[] lookup;
}
#elif HYLE_OS_LINUX
bool read_linux_cache_size(const char* path, std::uint64_t& value) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) return false;
  char text[64];
  const bool read = std::fgets(text, sizeof(text), file) != nullptr;
  std::fclose(file);
  if (!read) return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text) return false;
  std::uint64_t multiplier = 1;
  if (*end == 'K' || *end == 'k') {
    multiplier = 1024u;
    ++end;
  } else if (*end == 'M' || *end == 'm') {
    multiplier = 1024u * 1024u;
    ++end;
  } else if (*end == 'G' || *end == 'g') {
    multiplier = std::uint64_t{1024} * 1024u * 1024u;
    ++end;
  }
  while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
  if (*end != '\0' || parsed >
          (std::numeric_limits<std::uint64_t>::max)() / multiplier) {
    return false;
  }
  value = static_cast<std::uint64_t>(parsed) * multiplier;
  return true;
}

bool read_linux_cpu_list_count(const char* path, unsigned& count) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) return false;
  char text[4096];
  const bool read = std::fgets(text, sizeof(text), file) != nullptr;
  std::fclose(file);
  if (!read) return false;
  std::uint64_t total = 0;
  const char* cursor = text;
  while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r') {
    char* end = nullptr;
    errno = 0;
    const unsigned long begin = std::strtoul(cursor, &end, 10);
    if (errno != 0 || end == cursor) return false;
    unsigned long finish = begin;
    cursor = end;
    if (*cursor == '-') {
      ++cursor;
      errno = 0;
      finish = std::strtoul(cursor, &end, 10);
      if (errno != 0 || end == cursor || finish < begin) return false;
      cursor = end;
    }
    total += static_cast<std::uint64_t>(finish - begin) + 1u;
    if (total > (std::numeric_limits<unsigned>::max)()) return false;
    if (*cursor == ',') ++cursor;
    else if (*cursor != '\0' && *cursor != '\n' && *cursor != '\r') return false;
  }
  count = static_cast<unsigned>(total);
  return count != 0;
}

void fill_cache_observations(CpuCacheObservation* observations,
                             std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    auto& observation = observations[i];
    for (unsigned index = 0; index != 16; ++index) {
      char path[192];
      unsigned level = 0;
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%u/cache/index%u/level",
                    observation.cpu.id, index);
      if (!read_sysfs_unsigned(path, level)) break;

      char type[32]{};
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%u/cache/index%u/type",
                    observation.cpu.id, index);
      std::FILE* type_file = std::fopen(path, "r");
      if (type_file != nullptr) {
        const bool have_type =
            std::fgets(type, sizeof(type), type_file) != nullptr;
        std::fclose(type_file);
        if (!have_type) type[0] = '\0';
      }
      std::uint64_t capacity = 0;
      std::snprintf(path, sizeof(path),
                    "/sys/devices/system/cpu/cpu%u/cache/index%u/size",
                    observation.cpu.id, index);
      (void)read_linux_cache_size(path, capacity);
      unsigned line = 0;
      std::snprintf(
          path, sizeof(path),
          "/sys/devices/system/cpu/cpu%u/cache/index%u/coherency_line_size",
          observation.cpu.id, index);
      if (read_sysfs_unsigned(path, line)) {
        observation.line_size = (std::max)(observation.line_size, line);
      }
      unsigned sharing = 0;
      std::snprintf(
          path, sizeof(path),
          "/sys/devices/system/cpu/cpu%u/cache/index%u/shared_cpu_list",
          observation.cpu.id, index);
      (void)read_linux_cpu_list_count(path, sharing);

      if (level == 1) {
        if (std::strncmp(type, "Data", 4) == 0 ||
            std::strncmp(type, "Unified", 7) == 0) {
          observation.cache.l1d =
              (std::max)(observation.cache.l1d, capacity);
        }
        if (std::strncmp(type, "Instruction", 11) == 0 ||
            std::strncmp(type, "Unified", 7) == 0) {
          observation.cache.l1i =
              (std::max)(observation.cache.l1i, capacity);
        }
      } else if (level == 2) {
        observation.cache.l2 = (std::max)(observation.cache.l2, capacity);
        observation.cpus_per_l2 =
            (std::max)(observation.cpus_per_l2, sharing);
      } else if (level == 3) {
        observation.cache.l3 = (std::max)(observation.cache.l3, capacity);
        observation.cpus_per_l3 =
            (std::max)(observation.cpus_per_l3, sharing);
      }
    }
  }
}
#else
void fill_cache_observations(CpuCacheObservation* observations,
                             std::size_t count) noexcept {
  const CacheInfo coarse = cache_info();
  const unsigned line = cache_line_size();
  for (std::size_t i = 0; i < count; ++i) {
    observations[i].cache = coarse;
    observations[i].line_size = line;
  }
}
#endif
#endif

[[maybe_unused]] Result<std::size_t> generic_cache_class(
    CpuCacheClassInfo* dst, std::size_t cap) noexcept {
  if (dst != nullptr && cap != 0) {
    CpuCacheClassInfo out;
    auto topology = available_cpu_topology();
    if (!topology) return std::unexpected(std::move(topology).error());
    out.logical_cpus = topology->logical_cpus;
    out.physical_cores = topology->physical_cores;
    out.cache_line_size = cache_line_size();
    out.cache = cache_info();
    dst[0] = out;
  }
  return std::size_t{1};
}

#if HYLE_OS_MACOS
bool mac_sysctl_unsigned(const char* name, unsigned& value) noexcept {
  std::size_t size = sizeof(value);
  return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0;
}

std::uint64_t mac_sysctl_u64(const char* name) noexcept {
  std::uint64_t value = 0;
  std::size_t size = sizeof(value);
  return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
}

Result<std::size_t> mac_cache_classes(CpuCacheClassInfo* dst,
                                     std::size_t cap) noexcept {
  unsigned levels = 0;
  if (!mac_sysctl_unsigned("hw.nperflevels", levels) || levels == 0) {
    return generic_cache_class(dst, cap);
  }
  std::size_t count = 0;
  for (unsigned level = 0; level < levels; ++level) {
    char name[96];
    CpuCacheClassInfo out;
    std::snprintf(name, sizeof(name), "hw.perflevel%u.logicalcpu", level);
    if (!mac_sysctl_unsigned(name, out.logical_cpus) ||
        out.logical_cpus == 0) {
      continue;
    }
    out.performance_rank = levels - level;
    out.kind = level == 0 ? CpuKind::kPerformance : CpuKind::kEfficiency;
    std::snprintf(name, sizeof(name), "hw.perflevel%u.physicalcpu", level);
    (void)mac_sysctl_unsigned(name, out.physical_cores);
    out.cache_line_size = cache_line_size();
    std::snprintf(name, sizeof(name), "hw.perflevel%u.cpusperl2", level);
    (void)mac_sysctl_unsigned(name, out.logical_cpus_per_l2);
    std::snprintf(name, sizeof(name), "hw.perflevel%u.cpusperl3", level);
    (void)mac_sysctl_unsigned(name, out.logical_cpus_per_l3);
    std::snprintf(name, sizeof(name), "hw.perflevel%u.l1dcachesize", level);
    out.cache.l1d = mac_sysctl_u64(name);
    std::snprintf(name, sizeof(name), "hw.perflevel%u.l1icachesize", level);
    out.cache.l1i = mac_sysctl_u64(name);
    std::snprintf(name, sizeof(name), "hw.perflevel%u.l2cachesize", level);
    out.cache.l2 = mac_sysctl_u64(name);
    std::snprintf(name, sizeof(name), "hw.perflevel%u.l3cachesize", level);
    out.cache.l3 = mac_sysctl_u64(name);
    if (dst != nullptr && count < cap) dst[count] = out;
    ++count;
  }
  return count == 0 ? generic_cache_class(dst, cap)
                    : Result<std::size_t>{count};
}
#endif

}  // namespace

Result<std::size_t> available_cpu_cache_classes(
    CpuCacheClassInfo* dst, std::size_t cap) noexcept {
  if (dst == nullptr) cap = 0;
#if HYLE_OS_MACOS
  return mac_cache_classes(dst, cap);
#else
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    auto count_result = query_available_cpu_info(nullptr, 0);
    if (!count_result) return std::unexpected(std::move(count_result).error());
    const std::size_t capacity = *count_result;
    CpuInfo info_stack[256];
    CpuCacheObservation observation_stack[256];
    CpuInfo* info = capacity <= std::size(info_stack)
                        ? info_stack
                        : new (std::nothrow) CpuInfo[capacity];
    CpuCacheObservation* observations =
        capacity <= std::size(observation_stack)
            ? observation_stack
            : new (std::nothrow) CpuCacheObservation[capacity];
    if (info == nullptr || observations == nullptr) {
      if (info != nullptr && info != info_stack) delete[] info;
      if (observations != nullptr && observations != observation_stack) {
        delete[] observations;
      }
      return std::unexpected(Status::ResourceExhausted(
          "CPU cache-class snapshot allocation failed"));
    }
    auto filled = query_available_cpu_info(info, capacity);
    if (!filled) {
      if (info != info_stack) delete[] info;
      if (observations != observation_stack) delete[] observations;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled > capacity) {
      if (info != info_stack) delete[] info;
      if (observations != observation_stack) delete[] observations;
      continue;
    }
    const std::size_t count = *filled;
    for (std::size_t i = 0; i < count; ++i) observations[i].cpu = info[i];
    fill_cache_observations(observations, count);
    std::sort(observations, observations + count, cache_profile_less);

    std::size_t classes = 0;
    for (std::size_t begin = 0; begin < count;) {
      std::size_t end = begin + 1;
      while (end < count &&
             same_cache_profile(observations[begin], observations[end])) {
        ++end;
      }
      if (classes < cap) {
        CpuCacheClassInfo out;
        out.performance_rank = observations[begin].cpu.performance_rank;
        out.kind = observations[begin].cpu.kind;
        out.logical_cpus = static_cast<unsigned>(end - begin);
        out.cache_line_size = observations[begin].line_size;
        out.logical_cpus_per_l2 = observations[begin].cpus_per_l2;
        out.logical_cpus_per_l3 = observations[begin].cpus_per_l3;
        out.cache = observations[begin].cache;
        unsigned previous_core = kUnknownCpuTopologyId;
        for (std::size_t i = begin; i < end; ++i) {
          const unsigned core = observations[i].cpu.core_id;
          if (core != kUnknownCpuTopologyId && core != previous_core) {
            ++out.physical_cores;
            previous_core = core;
          }
        }
        dst[classes] = out;
      }
      ++classes;
      begin = end;
    }
    if (info != info_stack) delete[] info;
    if (observations != observation_stack) delete[] observations;
    return classes;
  }
  return std::unexpected(Status::Unavailable(
      "available CPU set changed repeatedly during cache-class snapshot"));
#endif
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

const char* cpu_feature_name(CpuFeature feature) noexcept {
  static constexpr const char* kNames[] = {
      "x86.sse2",          "x86.sse3",          "x86.ssse3",
      "x86.sse4.1",        "x86.sse4.2",        "x86.aes",
      "x86.pclmulqdq",     "x86.avx",           "x86.avx2",
      "x86.fma",           "x86.f16c",          "x86.lzcnt",
      "x86.bmi1",          "x86.bmi2",          "x86.avx512f",
      "x86.avx512bw",      "x86.avx512vl",      "x86.avx512dq",
      "x86.avx512cd",      "x86.avx512vnni",    "x86.avx512vbmi",
      "x86.avx512vbmi2",   "x86.avx512bitalg",  "x86.avx512vpopcntdq",
      "x86.gfni",          "x86.vaes",          "x86.vpclmulqdq",
      "x86.avx-vnni",      "x86.avx512bf16",    "x86.avx512fp16",
      "x86.avx10.2-512",   "x86.amx-tile-hw",   "x86.amx-int8-hw",
      "x86.amx-bf16-hw",   "x86.amx-fp16-hw",   "arm.neon",
      "arm.aes",           "arm.pmull",         "arm.dotprod",
      "arm.fp16",          "arm.sve",           "arm.sve2",
      "arm.sve-aes",       "arm.sve-pmull",     "arm.sve-i8mm",
      "arm.sve-f32mm",     "arm.sve-f64mm",     "arm.sve-bf16",
      "arm.i8mm",          "arm.bf16",          "arm.sme",
      "arm.sme2",          "riscv.v",           "riscv.zvbb",
      "riscv.zvbc",        "riscv.zvkned",      "riscv.zvfh",
      "riscv.zvfhmin",
  };
  static_assert(sizeof(kNames) / sizeof(*kNames) ==
                static_cast<std::size_t>(CpuFeature::kCount));
  const std::size_t index = static_cast<std::size_t>(feature);
  return index < sizeof(kNames) / sizeof(*kNames) ? kNames[index] : "unknown";
}

}  // namespace hyle::platform
