option(HYLE_TEST_SIMD_AVX512_RUNNABLE
  "register AVX-512 SIMD tests for direct execution on this host" OFF)
option(HYLE_TEST_SIMD_SVE
  "build the opt-in SVE test variant on an SVE-capable AArch64 toolchain" OFF)
option(HYLE_TEST_SIMD_RVV
  "build the opt-in RVV test variant on an RVV-capable RISC-V toolchain" OFF)
mark_as_advanced(
  HYLE_TEST_SIMD_AVX512_RUNNABLE
  HYLE_TEST_SIMD_SVE
  HYLE_TEST_SIMD_RVV)

# simd: the scalar test (dispatch + vec floor) always builds. The vec op suite is ALSO
# compiled per real ISA backend (sse4 / avx2) -- SAME source, different HYLE_SIMD_TARGET_*
# + -march, each pairing the backend against the naive reference. GCC/Clang need the
# arch flag to enable the intrinsics; MSVC x64 has SSE4.2 unconditionally and uses
# /arch:AVX2 for the 256-bit build. These variant exes run on the build host, which must
# support the ISA (both dev platforms are AVX2 -- matches the variant-DLL test model).
if(TARGET hyle_simd)
hyle_add_test(simd SOURCES dispatch_test.cc vec_test.cc native_test.cc
              runtime_test.cc ops_test.cc search_test.cc scan_test.cc dtype_test.cc
              math_test.cc sort_test.cc radix_sort_test.cc bitpack_test.cc
              narrow_float_test.cc narrow_ops_test.cc)
target_include_directories(hyle_simd_test PRIVATE "${PROJECT_SOURCE_DIR}")
hyle_add_fat_kernel(
  TARGET hyle_simd_test
  NAME public_fat_fixture
  IMPL "tests/simd/fat_kernel_fixture.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
# vec_test compiled per ISA variant: same source, different HYLE_SIMD_TARGET_* + the
# matching codegen flags from the single-source hyle_isa_flags (no MSVC/else drift here).
# Arch-gated (mirrors src/simd's HYLE_TARGET_PROCESSOR split): x86 builds the sse4/avx2
# variants, aarch64 builds the neon variant -- feeding -mavx2 to an ARM build (or the ARM
# neon.h to an x86 build) would fail. Each variant runs on the build host, so it must be
# the host's real ISA (the variant-DLL test model).
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_isa_flags(_sse4_flags SSE42)
  hyle_isa_flags(_avx2_flags AVX2)
  set(_sse4_options "")
  if(_sse4_flags)
    list(APPEND _sse4_options OPTIONS ${_sse4_flags})
  endif()
  if(MSVC)
    # The ported vqsort sorting networks deliberately prune impossible template
    # branches. MSVC 19.50 reports those discarded paths as C4702 after inlining;
    # keep the exception on this one test source instead of weakening /WX for the
    # SIMD module or either ISA test target.
    set_property(SOURCE sort_kernel_test.cc APPEND PROPERTY COMPILE_OPTIONS /wd4702)
  endif()
  # Fixed-width partial-vector proof: every logical width below the target native
  # width gets exact-byte LoadU/StoreU guards; every half-able width gets the scalar
  # half/combine oracle. Windows uses VirtualProtect guard pages, Unix uses mprotect.
  set(_fixed_partial_tests vec_partial_boundary_test.cc vec_half_combine_oracle_test.cc)
  hyle_add_test(simd NAME sse4 SOURCES vec_test.cc native_test.cc narrow_float_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_SSE4 ${_sse4_options})
  hyle_add_test(simd NAME avx2 SOURCES vec_test.cc native_test.cc
                sort_kernel_test.cc narrow_float_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX2 OPTIONS ${_avx2_flags})
  target_include_directories(hyle_simd_avx2_test PRIVATE "${PROJECT_SOURCE_DIR}")
  # simd-08 档C crypto: the base sse4/avx2 variants above compile WITHOUT -maes/-mpclmul, so on
  # GCC/Clang the AES/CLMul op face takes the SOFTWARE fallback. This extra variant adds those
  # flags (SSE42_AES tier) so the AES-NI / PCLMULQDQ HARDWARE path is actually compiled + run
  # and its result checked against the same in-test FIPS-197 oracle -- proving hardware ==
  # software. (On MSVC both variants are identical since AES-NI is unconditional there, but the
  # variant is harmless and keeps one build matrix across compilers.)
  hyle_isa_flags(_sse4_aes_flags SSE42_AES)
  set(_sse4_aes_options "")
  if(_sse4_aes_flags)
    list(APPEND _sse4_aes_options OPTIONS ${_sse4_aes_flags})
  endif()
  hyle_add_test(simd NAME sse4_aes SOURCES vec_test.cc native_test.cc narrow_float_test.cc
                DEFINES HYLE_SIMD_TARGET_SSE4 ${_sse4_aes_options})
  # AVX-512 vec_test variant: exercises the avx512.h backend's FULL op face (incl. simd-08
  # crypto + simd-09 cmp128/intdiv/RNG that route through the 512-bit backend). Historically
  # avx512.h was "written + objdump-verified zmm but NEVER RUN" (no AVX-512 dev machine). This
  # variant + Intel SDE closes that: build it always (GCC/Clang emit AVX-512 codegen without
  # needing the host to support it), then RUN under `sde64 -future` on an AVX2-only box. CTest
  # can't auto-run it (would SIGILL on a non-AVX512 host), so it is NOT registered as an
  # auto-run ctest -- it is built for compile validation + manual SDE execution. The optional
  # HYLE_TEST_SIMD_AVX512_RUNNABLE marks the host as true AVX-512 (or SDE-wrapped).
  hyle_isa_flags(_avx512_flags AVX512)
  # Default: build for compile-validation only, do NOT auto-run (bare metal
  # SIGILLs on a non-AVX512 host, e.g. WSL2/Hyper-V hides AVX-512). Set
  # HYLE_TEST_SIMD_AVX512_RUNNABLE (a true AVX-512 host, or ctest wrapped in `sde64
  # -future`) to register avx512/avx512_dl as auto-run ctests.
  if(HYLE_TEST_SIMD_AVX512_RUNNABLE)
    set(_avx512_autorun "")
  else()
    set(_avx512_autorun NO_AUTORUN)
  endif()
  set(_avx512_sort_kernel_test sort_kernel_test.cc)
  if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # The direct backend test bypasses product dispatch, so it cannot represent
    # cl.exe's AVX2 sort fallback. Public sort_test.cc remains the correctness
    # gate and exercises that fallback on an AVX-512 product/host.
    set(_avx512_sort_kernel_test "")
  endif()
  hyle_add_test(simd NAME avx512 SOURCES vec_test.cc native_test.cc
                ${_avx512_sort_kernel_test} narrow_float_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512_flags}
                ${_avx512_autorun})
  target_include_directories(hyle_simd_avx512_test PRIVATE "${PROJECT_SOURCE_DIR}")
  # AVX512_DL variant: same avx512.h backend, compiled with GFNI/VBMI/BITALG so the NATIVE
  # BitShuffle (VPSHUFBITQMB) / MultiRotateRight (VPMULTISHIFTQB) paths are taken instead of the
  # store+reload fallback. Built for compile validation + SDE execution (Ice Lake ISA); NOT
  # auto-run (SIGILLs on a Skylake-X / AVX2 host). Running BOTH avx512 and avx512_dl under SDE
  # verifies the fallback AND the native path agree with the scalar oracle. GCC/Clang only: MSVC's
  # /arch:AVX512 does not expose VBMI/BITALG macros (probe-verified), so the native path never
  # compiles on MSVC -- the base avx512 variant already covers the MSVC fallback, making a dl
  # variant there a pointless duplicate.
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    hyle_isa_flags(_avx512dl_flags AVX512_DL)
    hyle_add_test(simd NAME avx512_dl SOURCES vec_test.cc native_test.cc narrow_float_test.cc DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512dl_flags} ${_avx512_autorun})
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_isa_flags(_neon_flags NEON)
  set(_neon_options "")
  if(_neon_flags)
    list(APPEND _neon_options OPTIONS ${_neon_flags})
  endif()
  # The same fixed-width partial proof covers NEON's physical D/Q split. All cases
  # are permanent default-on regressions.
  hyle_add_test(simd NAME neon SOURCES vec_test.cc native_test.cc
                sort_kernel_test.cc narrow_float_test.cc vec_partial_boundary_test.cc
                vec_half_combine_oracle_test.cc
                DEFINES HYLE_SIMD_TARGET_NEON ${_neon_options})
  target_include_directories(hyle_simd_neon_test PRIVATE "${PROJECT_SOURCE_DIR}")
  # SVE variant: the FIRST scalable-width backend (backend/sve.h). Its op face is VL-agnostic, so
  # it uses a DEDICATED sve_test.cc (not vec_test.cc, which pins Lanes to a compile-time constant).
  # Gated by HYLE_TEST_SIMD_SVE=ON because SVE needs `-march=...+sve` and a CPU/emulator that supports
  # it: on a real SVE host (Graviton3, etc.) ctest runs it directly; on a non-SVE dev box it is
  # cross-compiled (aarch64-linux-gnu-g++ -march=armv8-a+sve) and run under qemu-aarch64-static at
  # multiple VLs (128/256/512) -- the "one binary, every width" scalable proof. See the recipe in
  # docs/architecture/simd-design.md (SVE section) / scripts. NOT auto-built on a plain aarch64
  # build (would need +sve which some arm64 hosts lack), so it is opt-in.
  if(HYLE_TEST_SIMD_SVE)
    hyle_add_test(simd NAME sve SOURCES sve_test.cc native_test.cc
                  narrow_float_test.cc
                  scalable_math_product_test.cc
                  scalable_math_dispatch_test.cc scalable_dtype_product_test.cc
                  scalable_bitpack_product_test.cc fixed128_bitpack_reference.cc
                  scalable_sort_product_test.cc
                  DEFINES HYLE_SIMD_TARGET_SVE HYLE_SIMD_HAS_SVE_VARIANT=1
                  OPTIONS -march=armv8-a+sve)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "riscv64|riscv")
  # RVV variant (backend/rvv.h), the SECOND scalable backend. Same gating rationale as SVE:
  # RVV needs -march=rv64gcv + a CPU/emulator with the V extension. On a real RVV host ctest runs
  # it; on a non-RVV box it is cross-compiled (riscv64-linux-gnu-g++ -march=rv64gcv) and run under
  # qemu-riscv64-static (scripts/rvv-qemu-test.sh). Gated by HYLE_TEST_SIMD_RVV.
  if(HYLE_TEST_SIMD_RVV)
    hyle_add_test(simd NAME rvv SOURCES rvv_test.cc native_test.cc
                  narrow_float_test.cc
                  scalable_math_product_test.cc
                  scalable_math_dispatch_test.cc scalable_dtype_product_test.cc
                  scalable_bitpack_product_test.cc fixed128_bitpack_reference.cc
                  scalable_sort_product_test.cc
                  DEFINES HYLE_SIMD_TARGET_RVV HYLE_SIMD_HAS_RVV_VARIANT=1
                  OPTIONS -march=rv64gcv)
  endif()
endif()
endif() # TARGET hyle_simd
