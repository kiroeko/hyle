# Builds the per-ISA translation units for a dispatched kernel body. The body is a source-rooted
# impl header that defines its own suffixed entry points from HYLE_SIMD_VARIANT_SUFFIX; this
# function only owns the mechanical architecture split, target macro, and codegen flag mapping.
#
# hyle_add_dispatched_kernel(
#   TARGET hyle_simd
#   NAME ops
#   IMPL "src/simd/ops/kernel_impl.h"
#   TARGETS scalar sse4 avx2 avx512 neon)
#
# TARGETS intentionally names the full portable set. Unsupported architectures are filtered here,
# not at call sites: x64 emits scalar/sse4/avx2/avx512, arm64 emits scalar/neon (plus SVE on Linux),
# riscv64 emits scalar/rvv, and unknown architectures emit scalar only. The generated source is
# configure-time build output, never a repository file. Dispatchers remain handwritten because
# their feature predicates and fallback semantics are module contracts, not CMake metadata.
include_guard(GLOBAL)

# Bind an ordinary consumer target to one explicit native SIMD backend. This is the zero-dispatch
# path: it defines exactly one HYLE_SIMD_TARGET_* and applies the matching compiler flags to the
# consumer target, so calls through hyle::simd::native can inline completely. It deliberately does
# not inspect CPUID; the produced binary requires the requested ISA.
#
#   hyle_target_simd(TARGET my_app ISA AVX2)
#
# Supported ISA names: SCALAR, SSE4, AVX2, AVX512, NEON, SVE, RVV.
function(hyle_target_simd)
  cmake_parse_arguments(ARG "" "TARGET;ISA" "" ${ARGN})
  if(NOT ARG_TARGET OR NOT ARG_ISA)
    message(FATAL_ERROR "hyle_target_simd requires TARGET and ISA")
  endif()
  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_target_simd: target '${ARG_TARGET}' does not exist")
  endif()
  string(TOUPPER "${ARG_ISA}" _isa)
  set(_is_x64 FALSE)
  set(_is_arm64 FALSE)
  set(_is_riscv64 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(_is_x64 TRUE)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_is_arm64 TRUE)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(_is_riscv64 TRUE)
  endif()

  if((_isa MATCHES "^(SSE4|AVX2|AVX512)$" AND NOT _is_x64) OR
     (_isa MATCHES "^(NEON|SVE)$" AND NOT _is_arm64) OR
     (_isa STREQUAL "SVE" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
     (_isa STREQUAL "RVV" AND NOT _is_riscv64))
    message(FATAL_ERROR
      "hyle_target_simd: ISA '${_isa}' is incompatible with ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(_define "")
  set(_tier "")
  if(_isa STREQUAL "SCALAR")
    # No target macro and no extra flags: vec.h selects the portable floor.
  elseif(_isa STREQUAL "SSE4")
    set(_define HYLE_SIMD_TARGET_SSE4)
    set(_tier SSE42)
  elseif(_isa STREQUAL "AVX2")
    set(_define HYLE_SIMD_TARGET_AVX2)
    set(_tier AVX2)
  elseif(_isa STREQUAL "AVX512")
    set(_define HYLE_SIMD_TARGET_AVX512)
    set(_tier AVX512)
  elseif(_isa STREQUAL "NEON")
    set(_define HYLE_SIMD_TARGET_NEON)
    set(_tier NEON)
  elseif(_isa STREQUAL "SVE")
    set(_define HYLE_SIMD_TARGET_SVE)
    set(_tier SVE)
  elseif(_isa STREQUAL "RVV")
    set(_define HYLE_SIMD_TARGET_RVV)
    set(_tier RVV)
  else()
    message(FATAL_ERROR "hyle_target_simd: unknown ISA '${ARG_ISA}'")
  endif()

  if(_define)
    target_compile_definitions(${ARG_TARGET} PRIVATE ${_define})
  endif()
  target_compile_definitions(${ARG_TARGET} PRIVATE
    HYLE_SIMD_EXACT=1 HYLE_SIMD_EXACT_TARGET_${_isa}=1)
  if(_tier)
    hyle_isa_flags(_flags "${_tier}")
    if(_flags)
      target_compile_options(${ARG_TARGET} PRIVATE ${_flags})
    endif()
  endif()
endfunction()

# Exact-deployment mode for the HYLE library. Per-ISA implementation TUs keep their own backend
# macro; baseline dispatch TUs receive a compile-time selection and matching minimum-ISA flags.
function(hyle_exact_simd_library)
  cmake_parse_arguments(ARG "" "TARGET;ISA" "" ${ARGN})
  if(NOT ARG_TARGET OR NOT ARG_ISA OR NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_exact_simd_library requires an existing TARGET and ISA")
  endif()
  string(TOUPPER "${ARG_ISA}" _isa)
  if((_isa MATCHES "^(SSE4|AVX2|AVX512)$" AND
      NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64") OR
     (_isa MATCHES "^(NEON|SVE)$" AND
      NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64") OR
     (_isa STREQUAL "SVE" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
     (_isa STREQUAL "RVV" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64"))
    message(FATAL_ERROR
      "hyle_exact_simd_library: ISA '${_isa}' is incompatible with ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(_isa STREQUAL "SSE4")
    set(_tier SSE42)
  elseif(_isa MATCHES "^(SCALAR|AVX2|AVX512|NEON|SVE|RVV)$")
    set(_tier "${_isa}")
  else()
    message(FATAL_ERROR "hyle_exact_simd_library: unknown ISA '${ARG_ISA}'")
  endif()
  target_compile_definitions(${ARG_TARGET} PRIVATE
    HYLE_SIMD_EXACT=1 HYLE_SIMD_EXACT_TARGET_${_isa}=1)
  if(NOT _isa STREQUAL "SCALAR")
    hyle_isa_flags(_flags "${_tier}")
    if(_flags)
      target_compile_options(${ARG_TARGET} PRIVATE ${_flags})
    endif()
  endif()
endfunction()

function(hyle_add_dispatched_kernel)
  cmake_parse_arguments(ARG "" "TARGET;NAME;IMPL" "TARGETS" ${ARGN})

  if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_IMPL OR NOT ARG_TARGETS)
    message(FATAL_ERROR "hyle_add_dispatched_kernel requires TARGET, NAME, IMPL, and TARGETS")
  endif()
  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_add_dispatched_kernel: target '${ARG_TARGET}' does not exist")
  endif()
  if(NOT COMMAND hyle_set_isa)
    message(FATAL_ERROR "hyle_add_dispatched_kernel requires hyle_set_isa from HyleModule.cmake")
  endif()
  if(NOT ARG_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "hyle_add_dispatched_kernel: NAME '${ARG_NAME}' is not an identifier")
  endif()
  list(FIND ARG_TARGETS "scalar" _scalar_index)
  if(_scalar_index EQUAL -1)
    message(FATAL_ERROR "hyle_add_dispatched_kernel: TARGETS must include scalar")
  endif()

  set(_is_x64 FALSE)
  set(_is_arm64 FALSE)
  set(_is_riscv64 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(_is_x64 TRUE)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_is_arm64 TRUE)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(_is_riscv64 TRUE)
  endif()

  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/simd-dispatch/${ARG_NAME}")
  file(MAKE_DIRECTORY "${_generated_dir}")

  foreach(_variant IN LISTS ARG_TARGETS)
    set(_target_define "")
    set(_tier "")
    set(_supported TRUE)

    if(_variant STREQUAL "scalar")
      # The portable floor intentionally has neither an ISA macro nor extra compiler flags.
    elseif(_variant STREQUAL "sse4")
      if(NOT _is_x64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_SSE4)
        set(_tier SSE42)
      endif()
    elseif(_variant STREQUAL "avx2")
      if(NOT _is_x64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_AVX2)
        set(_tier AVX2)
      endif()
    elseif(_variant STREQUAL "avx512")
      if(NOT _is_x64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_AVX512)
        set(_tier AVX512)
      endif()
    elseif(_variant STREQUAL "neon")
      if(NOT _is_arm64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_NEON)
        set(_tier NEON)
      endif()
    elseif(_variant STREQUAL "sve")
      if(NOT _is_arm64 OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_SVE)
        set(_tier SVE)
        target_compile_definitions(${ARG_TARGET} PRIVATE HYLE_SIMD_HAS_SVE_VARIANT=1)
      endif()
    elseif(_variant STREQUAL "rvv")
      if(NOT _is_riscv64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_RVV)
        set(_tier RVV)
        target_compile_definitions(${ARG_TARGET} PRIVATE HYLE_SIMD_HAS_RVV_VARIANT=1)
      endif()
    else()
      message(FATAL_ERROR "hyle_add_dispatched_kernel: unknown target '${_variant}'")
    endif()

    if(NOT _supported)
      continue()
    endif()

    set(_source "${_generated_dir}/${ARG_NAME}_${_variant}.cc")
    set(_contents "// Generated by hyle_add_dispatched_kernel; do not edit.\n")
    if(_target_define)
      string(APPEND _contents "#define ${_target_define}\n")
    endif()
    string(APPEND _contents "#define HYLE_SIMD_VARIANT_SUFFIX ${_variant}\n")
    string(APPEND _contents "#include \"${ARG_IMPL}\"\n")
    string(APPEND _contents "#undef HYLE_SIMD_VARIANT_SUFFIX\n")
    if(_target_define)
      string(APPEND _contents "#undef ${_target_define}\n")
    endif()
    file(WRITE "${_source}" "${_contents}")

    set_source_files_properties("${_source}" PROPERTIES GENERATED TRUE)
    target_sources(${ARG_TARGET} PRIVATE "${_source}")
    if(_tier)
      hyle_set_isa("${_source}" "${_tier}")
    endif()
  endforeach()
endfunction()

# Public spelling for source consumers. IMPL defines suffixed pointer/POD entry points using
# HYLE_SIMD_VARIANT_SUFFIX; pair it with hyle::simd::select_fat_kernel in the baseline TU.
function(hyle_add_fat_kernel)
  hyle_add_dispatched_kernel(${ARGN})
endfunction()
