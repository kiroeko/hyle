include_guard(GLOBAL)

# hyle_isa_flags(<out_var> <TIER>) -- SINGLE SOURCE OF TRUTH for the per-compiler
# ISA codegen flags. "Enable AVX2" is `/arch:AVX2` on MSVC but
# `-mavx2 -mfma -mf16c -mbmi2`
# on GCC/Clang; hand-writing the if(MSVC)/else mapping at every TU drifts (e.g. one
# AVX2 site forgetting -mf16c) -> the same logical tier compiled differently = silent
# semantic skew. Define each tier ONCE here; every call site asks for a TIER name.
#
# TIERS (x86-64; the HYLE per-ISA kernel contract -- see src/simd/CMakeLists.txt):
#   SSE42    : SSE4.2. MSVC x64 has these intrinsics unconditionally (NO flag); GCC/Clang -msse4.2.
#   AVX_F16C : AVX + F16C, capped BELOW AVX2 on purpose. F16C shipped WITH AVX on Ivy
#              Bridge (which lacks AVX2); going to AVX2 would let the auto-vectorized tail
#              emit an AVX2 int op that SIGILLs there. This cap is a SIGILL-safety contract,
#              not an oversight -- keep it distinct from AVX2.
#   AVX2     : AVX2 + FMA + F16C + BMI2. BMI2 is part of HYLE's runtime AVX2
#              group and is used by the 16-bit mask extraction fast path.
#   AVX2_INT : AVX2 256-bit INTEGER only; no FMA/F16C needed. MSVC has
#              no finer knob than /arch:AVX2, so it collapses to AVX2 there; GCC/Clang can
#              ask for just -mavx2.
#   AVX512   : AVX-512 F+BW+VL+DQ plus BMI2 (512-bit zmm and the inherited mask helper).
#              MSVC /arch:AVX512; GCC/Clang use the matching explicit flags. ONLY entered
#              when best_isa() >= kAvx512 (the AVX3 group is runtime-gated), so the zmm
#              codegen never runs on a non-AVX512 CPU.
#   AVX512_BF16 : AVX512 + the bf16 convert extension (VCVTNEPS2BF16). MSVC's /arch:AVX512
#              already enables the bf16 intrinsics (probe-verified, CL 19.50); GCC/Clang add
#              -mavx512bf16 on top of the AVX512 four. Clang permits FMA/F16C under this flag
#              set, so specialty runtime dispatch gates those compiler prerequisites too.
# Returns a CMake list in <out_var> (may be empty, e.g. SSE42 on cl.exe).
function(_hyle_gnu_like_isa_flags out_var tier)
  if(tier STREQUAL "SSE42")
    set(_flags "-msse4.2")
  elseif(tier STREQUAL "SSE42_AES")
    # AES-NI + PCLMULQDQ are optional and orthogonal to SSE4.2.
    set(_flags "-msse4.2;-maes;-mpclmul")
  elseif(tier STREQUAL "AVX_F16C")
    set(_flags "-mavx;-mf16c")
  elseif(tier STREQUAL "AVX2")
    set(_flags "-mavx2;-mfma;-mf16c;-mbmi2")
  elseif(tier STREQUAL "AVX2_INT")
    set(_flags "-mavx2")
  elseif(tier STREQUAL "AVX2_VNNI")
    set(_flags "-mavx2;-mavxvnni")
  elseif(tier STREQUAL "AVX512")
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mbmi2")
  elseif(tier STREQUAL "AVX512_BF16")
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512bf16")
  elseif(tier STREQUAL "AVX512_VNNI")
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni")
  elseif(tier STREQUAL "AVX512_DL")
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mavx512bitalg")
  elseif(tier STREQUAL "NEON")
    set(_flags "")
  elseif(tier STREQUAL "SVE")
    set(_flags "-march=armv8-a+sve")
  elseif(tier STREQUAL "RVV")
    set(_flags "-march=rv64gcv")
  else()
    message(FATAL_ERROR "hyle_isa_flags: unknown tier '${tier}'")
  endif()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

function(_hyle_forward_clang_cl_flags out_var)
  set(_forwarded "")
  foreach(_flag IN LISTS ARGN)
    list(APPEND _forwarded "/clang:${_flag}")
  endforeach()
  set(${out_var} "${_forwarded}" PARENT_SCOPE)
endfunction()

function(hyle_isa_flags out_var tier)
  # CMake's MSVC variable means the MSVC-compatible frontend family and is also
  # true for clang-cl. Only cl.exe gets /arch mappings; clang-cl needs Clang's
  # precise feature flags forwarded through /clang: so always_inline intrinsics
  # and runtime dispatch describe the same target contract.
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(tier STREQUAL "SSE42")
      set(${out_var} "" PARENT_SCOPE)
    elseif(tier STREQUAL "SSE42_AES")
      # MSVC x64 exposes AES-NI / PCLMULQDQ intrinsics unconditionally (no flag), same as
      # the rest of SSE4.2. The crypto op face's hardware path is always live here.
      set(${out_var} "" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX_F16C")
      set(${out_var} "/arch:AVX" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX2" OR tier STREQUAL "AVX2_INT" OR
           tier STREQUAL "AVX2_VNNI")
      set(${out_var} "/arch:AVX2" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX512")
      set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX512_BF16")
      set(${out_var} "/arch:AVX512" PARENT_SCOPE)  # MSVC /arch:AVX512 already covers bf16 intrinsics
    elseif(tier STREQUAL "AVX512_VNNI")
      set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX512_DL")
      # MSVC's /arch:AVX512 does NOT define __AVX512VBMI__/__AVX512BITALG__ (probe-verified: only
      # AVX512F+BW), so the native VPSHUFBITQMB/VPMULTISHIFTQB path is #if-compiled OUT and the
      # store+reload fallback is taken -- identical to the base AVX512 variant. The dl test variant
      # is therefore a GCC/Clang+SDE construct (see tests/simd/CMakeLists.txt gate); this case exists so
      # a stray AVX512_DL request on MSVC still resolves rather than FATAL_ERROR.
      set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    elseif(tier STREQUAL "NEON")
      set(${out_var} "" PARENT_SCOPE)  # AArch64 baseline; meaningful only on an ARM target
    else()
      message(FATAL_ERROR "hyle_isa_flags: unknown tier '${tier}'")
    endif()
  else()
    _hyle_gnu_like_isa_flags(_flags "${tier}")
    if(MSVC)
      _hyle_forward_clang_cl_flags(_flags ${_flags})
    endif()
    set(${out_var} "${_flags}" PARENT_SCOPE)
  endif()
endfunction()

# hyle_set_isa(<source-file> <TIER>) -- apply the tier's flags to ONE TU's COMPILE_OPTIONS.
# Convenience over hyle_isa_flags for the common "this kernel TU compiles at <tier>" case.
function(hyle_set_isa src tier)
  hyle_isa_flags(_flags "${tier}")
  if(_flags)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
  endif()
endfunction()

# Highway derives HWY_STATIC_TARGET from a stricter feature chain than HYLE's legacy
# intrinsic backends. In particular, Highway's default SSE4 includes AES+PCLMUL and AVX2
# also includes BMI1/BMI2+FMA+F16C. HYLE's canonical x86 tiers do not promise AES/PCLMUL,
# so direct-Highway TUs define the upstream-supported HWY_DISABLE_PCLMUL_AES and never
# enable those instructions accidentally. Reusing hyle_isa_flags() can otherwise compile a
# nominal AVX2 source as a 128-bit Highway target. Every direct-Highway fixed/fat TU must
# use this mapping; generated TUs additionally assert the resulting HWY_STATIC_TARGET.
function(hyle_highway_isa_flags out_var tier)
  string(TOUPPER "${tier}" _tier)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(_tier STREQUAL "SCALAR" OR _tier STREQUAL "SSE42" OR
       _tier STREQUAL "NEON")
      set(_flags "")
    elseif(_tier STREQUAL "AVX2")
      set(_flags "/arch:AVX2")
    elseif(_tier STREQUAL "AVX512")
      message(FATAL_ERROR
        "locked Highway blocks AVX3 with classic cl.exe; use clang-cl or AVX2")
    else()
      message(FATAL_ERROR
        "hyle_highway_isa_flags: unsupported classic-MSVC tier '${tier}'")
    endif()
  else()
    if(_tier STREQUAL "SCALAR" OR _tier STREQUAL "NEON")
      set(_flags "")
    elseif(_tier STREQUAL "SSE42")
      set(_flags "-msse4.2")
    elseif(_tier STREQUAL "AVX2")
      set(_flags "-mavx2;-mfma;-mf16c;-mbmi;-mbmi2")
    elseif(_tier STREQUAL "AVX512")
      set(_flags
        "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512cd;-mavx2;-mfma;-mf16c;-mbmi;-mbmi2")
    elseif(_tier STREQUAL "SVE")
      set(_flags "-march=armv8-a+sve")
    elseif(_tier STREQUAL "RVV")
      set(_flags "-march=rv64gcv")
    else()
      message(FATAL_ERROR "hyle_highway_isa_flags: unknown tier '${tier}'")
    endif()
    if(MSVC)
      _hyle_forward_clang_cl_flags(_flags ${_flags})
    endif()
  endif()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

function(hyle_set_highway_target_isa src tier)
  hyle_highway_isa_flags(_flags "${tier}")
  string(TOUPPER "${tier}" _tier)
  if(_tier MATCHES "^(SSE42|AVX2|AVX512)$")
    set_property(SOURCE "${src}" APPEND PROPERTY COMPILE_DEFINITIONS
      HWY_DISABLE_PCLMUL_AES=1)
    if(_tier STREQUAL "SSE42")
      # cl.exe has no SSE4 feature macro unless /arch:AVX is enabled. Forcing
      # Highway's static SSE4 body is safe because HYLE selects this TU only
      # after Platform proves SSE4.2; disabling AES keeps the contracts equal.
      set_property(SOURCE "${src}" APPEND PROPERTY COMPILE_DEFINITIONS
        HWY_WANT_SSE4=1)
    endif()
  endif()
  if(_flags)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
  endif()
endfunction()

# Compatibility spelling for existing AVX2 Highway product TUs.
function(hyle_set_highway_isa src)
  hyle_set_highway_target_isa("${src}" AVX2)
endfunction()

# Clear ISA flags previously assigned to a source when a reused build directory changes architecture.
# CMake's source-file properties persist across reconfigure, so setting an empty value in a child
# CMakeLists cannot override the parent-directory property left by an x86 configuration.
function(hyle_clear_source_isa src)
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "")
endfunction()

# hyle_set_avx512_math_isa(<source-file>) -- ISA flags for the AVX-512 (16-wide) borrow-core
# direct Highway math TU (transcendental_highway_avx512.cc). DISTINCT from
# the ops AVX512 tier (f/bw/vl/dq) for TWO reasons proven by compile probes:
#   - xsimd's avx512dq kernels assume avx512cd (its headers #error "avx512dq requires
#     avx512cd" without it), so the math group needs CD on top of the ops four.
#   - Highway's HWY_STATIC_TARGET must see the FULL baseline macro chain to pick AVX3: the
#     AVX2 baseline (SSE4=PCLMUL+AES + BMI2+FMA+F16C) PLUS the AVX-512 group -- else it
#     silently downgrades (same trap as hyle_set_highway_isa, now one tier wider).
# So this is the AVX2 highway baseline + the full AVX-512 group (f/bw/vl/dq/cd). MSVC's
# /arch:AVX512 already implies the whole set (probe-verified, CL 19.50). Only reached when
# best_isa() >= kAvx512 (runtime-gated in math_dispatch), so the zmm never runs on a non-512 CPU.
function(hyle_set_avx512_math_isa src)
  hyle_set_highway_target_isa("${src}" AVX512)
endfunction()

# hyle_set_avx512_sort_isa(<source-file>) -- ISA flags for an AVX-512 (512-bit) vqsort TU.
# The full AVX2 baseline plus AVX-512 f/bw/vl/dq/cd is required. This helper is intentionally
# unavailable to the classic MSVC frontend: official Highway marks AVX3 targets HWY_BROKEN_MSVC,
# and the HYLE port's cl.exe correctness gate confirms data corruption. Product CMake links its
# AVX2 sort TU instead. clang-cl is a distinct frontend and retains this AVX-512 path.
function(hyle_set_avx512_sort_isa src)
  hyle_set_highway_target_isa("${src}" AVX512)
endfunction()
