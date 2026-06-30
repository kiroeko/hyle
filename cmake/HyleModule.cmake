# hyle_isa_flags(<out_var> <TIER>) -- SINGLE SOURCE OF TRUTH for the per-compiler
# ISA codegen flags. "Enable AVX2" is `/arch:AVX2` on MSVC but `-mavx2 -mfma -mf16c`
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
#   AVX2     : full AVX2 + FMA + F16C (the canonical 256-bit float tier).
#   AVX2_INT : AVX2 256-bit INTEGER only (bf16 shift/pack) -- no FMA/F16C needed. MSVC has
#              no finer knob than /arch:AVX2, so it collapses to AVX2 there; GCC/Clang can
#              ask for just -mavx2.
#   AVX512   : AVX-512 F+BW+VL+DQ (512-bit zmm). MSVC /arch:AVX512; GCC/Clang the four
#              -mavx512* flags matching backend/avx512.h's compile contract. ONLY entered
#              when best_isa() >= kAvx512 (the AVX3 group is runtime-gated), so the zmm
#              codegen never runs on a non-AVX512 CPU.
# Returns a CMake list in <out_var> (may be empty, e.g. SSE42 on MSVC).
function(hyle_isa_flags out_var tier)
  if(MSVC)
    if(tier STREQUAL "SSE42")
      set(${out_var} "" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX_F16C")
      set(${out_var} "/arch:AVX" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX2" OR tier STREQUAL "AVX2_INT")
      set(${out_var} "/arch:AVX2" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX512")
      set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    else()
      message(FATAL_ERROR "hyle_isa_flags: unknown tier '${tier}'")
    endif()
  else()  # GCC / Clang
    if(tier STREQUAL "SSE42")
      set(${out_var} "-msse4.2" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX_F16C")
      set(${out_var} "-mavx;-mf16c" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX2")
      set(${out_var} "-mavx2;-mfma;-mf16c" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX2_INT")
      set(${out_var} "-mavx2" PARENT_SCOPE)
    elseif(tier STREQUAL "AVX512")
      set(${out_var} "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq" PARENT_SCOPE)
    else()
      message(FATAL_ERROR "hyle_isa_flags: unknown tier '${tier}'")
    endif()
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
# that emit 256-bit code from just `-mavx2 -mfma -mf16c` (the AVX2 tier = our kernel ISA
# contract, kept minimal for SIGILL-safety). But Highway's HWY_STATIC_TARGET resolves its
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
  if(MSVC)
    set(_flags "/arch:AVX2")  # MSVC's /arch:AVX2 already implies the full AVX2 baseline
  else()
    set(_flags "-mavx2;-mfma;-mf16c;-mbmi;-mbmi2;-maes;-mpclmul")
  endif()
  set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()

# hyle_add_module(<name> [SOURCES ...] [DEPS ...] [PUBLIC_DEPS ...])
# Declares a HYLE module as a library target `hyle_<name>` with conventional
# include dirs, warnings, sanitizer flags, and a tri-state export macro.
function(hyle_add_module name)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS;PUBLIC_DEPS" ${ARGN})

  set(target "hyle_${name}")
  if(ARG_SOURCES)
    add_library(${target} ${ARG_SOURCES})
  else()
    add_library(${target} INTERFACE)
  endif()

  if(ARG_SOURCES)
    target_include_directories(${target}
      PUBLIC  "${PROJECT_SOURCE_DIR}/include"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Export the C++23 requirement to CONSUMERS: a downstream target linking hyle::<mod>
    # includes our headers (which use `concept`, std::expected, etc.), so it must compile
    # at C++23 too. CMAKE_CXX_STANDARD only governs HYLE's OWN build; PUBLIC compile
    # features propagate the floor to consumers (verified by samples/ -- a consumer that
    # didn't set the standard failed on `concept` until this was added).
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_link_libraries(${target} PRIVATE ${ARG_DEPS} PUBLIC ${ARG_PUBLIC_DEPS})
    hyle_apply_warnings(${target})
    hyle_apply_sanitizer(${target})
    # tri-state export macro: HYLE_<NAME>_API
    string(TOUPPER "${name}" up)
    if(BUILD_SHARED_LIBS)
      target_compile_definitions(${target} PRIVATE "HYLE_${up}_BUILD")
    else()
      target_compile_definitions(${target} PUBLIC "HYLE_STATIC")
    endif()
  else()
    target_include_directories(${target} INTERFACE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(${target} INTERFACE cxx_std_23)  # header-only modules too
    target_link_libraries(${target} INTERFACE ${ARG_PUBLIC_DEPS})
  endif()

  add_library(hyle::${name} ALIAS ${target})
endfunction()

# hyle_add_test(<module> SOURCES ... [NAME <suffix>] [DEFINES ...] [OPTIONS ...] [DEPS ...])
# -> test exe linked to hyle_<module> + gtest. NAME gives a distinct target suffix so a
# module can have several test exes (e.g. simd compiles the SAME source per ISA variant
# with different DEFINES/OPTIONS -- HYLE_SIMD_TARGET_AVX2 + -mavx2 etc.).
function(hyle_add_test module)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES;DEPS;DEFINES;OPTIONS" ${ARGN})
  if(ARG_NAME)
    set(target "hyle_${module}_${ARG_NAME}_test")
  else()
    set(target "hyle_${module}_test")
  endif()
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE hyle_${module} GTest::gtest GTest::gtest_main ${ARG_DEPS})
  if(ARG_DEFINES)
    target_compile_definitions(${target} PRIVATE ${ARG_DEFINES})
  endif()
  if(ARG_OPTIONS)
    target_compile_options(${target} PRIVATE ${ARG_OPTIONS})
  endif()
  hyle_apply_warnings(${target})
  hyle_apply_sanitizer(${target})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

# hyle_add_bench(<module> SOURCES ...) -> bench exe linked to hyle_<module> + nanobench
function(hyle_add_bench module)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  set(target "hyle_${module}_bench")
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE hyle_${module} nanobench ${ARG_DEPS})
  hyle_apply_warnings(${target})
endfunction()
