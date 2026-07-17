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

#if defined(HYLE_OS_LINUX) && (defined(HYLE_ARCH_ARM64) || defined(HYLE_ARCH_RISCV64))
#include <sys/auxv.h>
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

}  // namespace

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures f = detect();  // Meyers: thread-safe init-once (C++11+)
  return f;
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
