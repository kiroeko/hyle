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

# hyle_set_highway_isa(<source-file>) -- ISA flags for ANY TU that uses Highway's STATIC
# dispatch (hn::Exp/PromoteTo/etc) -- both bench comparison TUs AND product-lib cores that
# borrow Highway (e.g. src/simd/math/transcendental_highway.cc).
# DISTINCT from hyle_set_isa(AVX2) on purpose: HYLE's own kernels are hand-written intrinsics
# that need only `-mavx2 -mfma -mf16c -mbmi2` (BMI2 serves the i16 mask extractor and is in
# HYLE's runtime AVX2 group). Highway's HWY_STATIC_TARGET additionally resolves its
# vector width from the FULL baseline-feature macro chain in detect_targets.h:
#   AVX2 baseline  <- SSE4 baseline (needs __PCLMUL__ + __AES__) + BMI2 + FMA + F16C + __AVX2__
# With only `-mavx2 -mfma -mf16c`, Highway silently DOWNGRADES to SSSE3 (128-bit, Lanes=4) on
# GCC/Clang. This bit TWICE: (1) the parity benches raced HYLE-256b vs Highway-128b (bogus
# "win"); (2) the SHIPPED transcendental_highway core ran at 128-bit on Linux -- log batch
# 0.53ns vs the 256-bit 0.27ns, a real product perf bug (objdump: 0 ymm/170 xmm). MSVC's
# /arch:AVX2 already sets the whole set, so this only bites GCC/Clang. Add the missing
# baseline so Highway picks true 256-bit AVX2. (Verified: HWY_STATIC_TARGET=AVX2, Lanes=8,
# ymm emitted; Linux log batch 0.53->0.27ns, ties Highway core.)
function(hyle_set_highway_isa src)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_flags "/arch:AVX2")  # MSVC's /arch:AVX2 already implies the full AVX2 baseline
  else()
    set(_flags "-mavx2;-mfma;-mf16c;-mbmi;-mbmi2;-maes;-mpclmul")
    if(MSVC)
      _hyle_forward_clang_cl_flags(_flags ${_flags})
    endif()
  endif()
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()

# Clear ISA flags previously assigned to a source when a reused build directory changes architecture.
# CMake's source-file properties persist across reconfigure, so setting an empty value in a child
# CMakeLists cannot override the parent-directory property left by an x86 configuration.
function(hyle_clear_source_isa src)
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "")
endfunction()

# hyle_set_avx512_math_isa(<source-file>) -- ISA flags for the AVX-512 (16-wide) borrow-core
# math TUs (transcendental_xsimd_avx512.cc / transcendental_highway_avx512.cc). DISTINCT from
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
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_flags "/arch:AVX512")
  else()
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512cd;-mavx2;-mfma;-mf16c;-mbmi;-mbmi2;-maes;-mpclmul")
    if(MSVC)
      _hyle_forward_clang_cl_flags(_flags ${_flags})
    endif()
  endif()
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()

# hyle_set_avx512_sort_isa(<source-file>) -- ISA flags for the AVX-512 (512-bit) Highway vqsort
# borrow TU (src/simd/sort/vqsort_highway_avx512.cc). SAME requirement as the math AVX-512 borrow:
# Highway's HWY_STATIC_TARGET must see the FULL baseline macro chain to pick AVX3 -- the AVX2
# baseline (SSE4=PCLMUL+AES + BMI2+FMA+F16C) PLUS the AVX-512 group (f/bw/vl/dq/cd), else it
# silently downgrades to the 256-bit target (the shipped vqsort would run at half width). It is a
# separate function from hyle_set_avx512_math_isa only for a self-documenting call site (the flag
# set is identical -- vqsort has no extra ISA need beyond the Highway AVX3 baseline). MSVC's
# /arch:AVX512 implies the whole set; UNLIKE the math AVX3 log core, vqsort's AVX3 target is NOT
# HWY_BROKEN on MSVC (integer/compare SIMD, not FP transcendentals), so this TU is compiled there.
# Only reached when best_isa() >= kAvx512 (runtime-gated in sort_dispatch), so the zmm codegen
# never runs on a non-AVX512 CPU.
function(hyle_set_avx512_sort_isa src)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_flags "/arch:AVX512")
  else()
    set(_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512cd;-mavx2;-mfma;-mf16c;-mbmi;-mbmi2;-maes;-mpclmul")
    if(MSVC)
      _hyle_forward_clang_cl_flags(_flags ${_flags})
    endif()
  endif()
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()
