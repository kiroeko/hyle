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

# Resolve the one expert target spelling shared by source and installed SDKs.
function(_hyle_simd_expert_target out_var)
  if(TARGET hyle::simd_expert)
    set(${out_var} hyle::simd_expert PARENT_SCOPE)
  elseif(TARGET hyle_simd_expert)
    set(${out_var} hyle_simd_expert PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "The selected HYLE package/source tree has no SIMD expert capability; "
      "select the simd component before calling this helper")
  endif()
endfunction()

# Internal source builds use the physical factor target. Installed expert
# helpers see the same factor under its deliberately non-product export name.
# Never route HYLE's own fat-kernel implementation through the public Platform
# facade: facades are graph roots, not internal dependencies.
function(_hyle_platform_factor_target out_var)
  if(TARGET hyle_platform)
    set(${out_var} hyle_platform PARENT_SCOPE)
  elseif(TARGET hyle::_factor_platform)
    set(${out_var} hyle::_factor_platform PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "HYLE SIMD expert helper requires the Platform factor target")
  endif()
endfunction()

# CMake directory scopes do not propagate ordinary variables from HYLE's
# add_subdirectory() back to a source consumer. The expert target exports the
# normalized package target, while the global property remains the source-tree
# fallback established by HyleTargetPlatform.
function(_hyle_simd_target_processor out_var)
  set(_processor "")
  if(TARGET hyle_simd_expert)
    get_target_property(_processor hyle_simd_expert HYLE_SIMD_TARGET_PROCESSOR)
  elseif(TARGET hyle::simd_expert)
    get_target_property(_processor hyle::simd_expert HYLE_SIMD_TARGET_PROCESSOR)
  endif()
  if(_processor MATCHES "-NOTFOUND$")
    set(_processor "")
  endif()
  if(_processor STREQUAL "")
    set(_processor "${HYLE_TARGET_PROCESSOR}")
  endif()
  if(_processor STREQUAL "")
    get_property(_processor GLOBAL PROPERTY HYLE_TARGET_PROCESSOR)
  endif()
  if(_processor STREQUAL "")
    message(FATAL_ERROR
      "HYLE's normalized target processor is unavailable; call this helper only "
      "after add_subdirectory(hyle) or find_package(Hyle COMPONENTS simd)")
  endif()
  set(${out_var} "${_processor}" PARENT_SCOPE)
endfunction()

function(_hyle_simd_expert_exact_target out_var)
  set(_exact "")
  if(TARGET hyle_simd_expert)
    get_target_property(_exact hyle_simd_expert HYLE_SIMD_EXACT_TARGET)
  elseif(TARGET hyle::simd_expert)
    get_target_property(_exact hyle::simd_expert HYLE_SIMD_EXACT_TARGET)
  endif()
  if(_exact MATCHES "-NOTFOUND$")
    set(_exact "")
  endif()
  set(${out_var} "${_exact}" PARENT_SCOPE)
endfunction()

# Resolve the canonical SIMD targets compiled into hyle_simd from one manifest. Fat builds keep
# every target supported by the architecture; exact builds keep only the requested canonical
# target. The generated header is consumed by runtime/dispatch code, so CMake source selection and
# C++ table construction cannot silently drift apart.
#
#   hyle_simd_target_manifest(
#     OUT_VARIANTS compiled_variants
#     OUT_HEADER generated_header)
function(hyle_simd_target_manifest)
  cmake_parse_arguments(ARG "" "OUT_VARIANTS;OUT_HEADER" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_simd_target_manifest: unknown arguments '${ARG_UNPARSED_ARGUMENTS}', "
      "missing values '${ARG_KEYWORDS_MISSING_VALUES}'")
  endif()
  if(NOT ARG_OUT_VARIANTS OR NOT ARG_OUT_HEADER)
    message(FATAL_ERROR
      "hyle_simd_target_manifest requires OUT_VARIANTS and OUT_HEADER")
  endif()

  if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(_supported scalar sse4 avx2 avx512)
  elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_supported scalar neon)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND _supported sve)
    endif()
  elseif(HYLE_TARGET_PROCESSOR MATCHES "riscv64|RISCV64")
    set(_supported scalar rvv)
  else()
    set(_supported scalar)
  endif()

  if(HYLE_SIMD_EXACT_TARGET)
    string(TOLOWER "${HYLE_SIMD_EXACT_TARGET}" _exact)
    list(FIND _supported "${_exact}" _exact_index)
    if(_exact_index EQUAL -1)
      message(FATAL_ERROR
        "HYLE_SIMD_EXACT_TARGET='${HYLE_SIMD_EXACT_TARGET}' is incompatible with "
        "${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}; supported: ${_supported}")
    endif()
    set(_compiled "${_exact}")
  else()
    set(_compiled "${_supported}")
    set(_exact "")
  endif()

  set(_header_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/simd-dispatch")
  set(_header "${_header_dir}/target_manifest.h")
  # A build directory may switch from fat to exact (or between exact tiers). Remove stale
  # configure outputs before writing the new manifest so artifact audits see the actual source
  # set, not files left by an older configuration.
  file(REMOVE_RECURSE "${_header_dir}")
  file(MAKE_DIRECTORY "${_header_dir}")
  set(_contents
    "#pragma once\n\n// Generated by hyle_simd_target_manifest; do not edit.\n")
  set(_defines "")
  set(_foreach "#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) \\\n")
  list(LENGTH _compiled _compiled_count)
  set(_index 0)
  foreach(_variant IN LISTS _compiled)
    math(EXPR _index "${_index} + 1")
    if(_variant STREQUAL "scalar")
      set(_pretty Scalar)
      set(_target Scalar)
    elseif(_variant STREQUAL "sse4")
      set(_pretty Sse4)
      set(_target Sse4)
    elseif(_variant STREQUAL "avx2")
      set(_pretty Avx2)
      set(_target Avx2)
    elseif(_variant STREQUAL "avx512")
      set(_pretty Avx512)
      set(_target Avx512)
    elseif(_variant STREQUAL "neon")
      set(_pretty Neon)
      set(_target Neon)
    elseif(_variant STREQUAL "sve")
      set(_pretty Sve)
      set(_target Sve)
    elseif(_variant STREQUAL "rvv")
      set(_pretty Rvv)
      set(_target Rvv)
    endif()
    string(TOUPPER "${_variant}" _upper)
    if(_exact)
      set(_exact_pretty "${_pretty}")
    endif()
    string(APPEND _defines "#define HYLE_SIMD_COMPILED_${_upper} 1\n")
    if(_index LESS _compiled_count)
      string(APPEND _foreach "  M(${_pretty}, ${_variant}, ${_target}) \\\n")
    else()
      string(APPEND _foreach "  M(${_pretty}, ${_variant}, ${_target})\n")
    endif()
  endforeach()
  string(APPEND _contents "${_defines}\n${_foreach}")
  if(_exact)
    string(APPEND _contents
      "\n#define HYLE_SIMD_EXACT_VARIANT ${_exact}\n"
      "#define HYLE_SIMD_EXACT_RUNTIME k${_exact_pretty}Runtime\n")
  endif()
  file(CONFIGURE OUTPUT "${_header}" CONTENT "${_contents}" @ONLY)

  set_property(GLOBAL PROPERTY HYLE_SIMD_COMPILED_VARIANTS "${_compiled}")
  set(${ARG_OUT_VARIANTS} "${_compiled}" PARENT_SCOPE)
  set(${ARG_OUT_HEADER} "${_header}" PARENT_SCOPE)
endfunction()

# Bind a consumer target to one explicit SIMD target. This is
# the zero-dispatch expert path: it exposes the centrally locked Highway/xsimd
# headers, selects one static Highway target (xsimd observes the same compiler
# ISA flags), and applies those flags. Provider register types must remain
# inside this target-bound translation unit.
# It deliberately does not inspect CPUID; the produced binary requires the ISA.
#
#   hyle_target_simd(TARGET my_app ISA AVX2)
#
# Supported ISA names: SCALAR, SSE4, AVX2, AVX512, NEON, SVE, RVV.
function(hyle_target_simd)
  cmake_parse_arguments(ARG "" "TARGET;ISA" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_target_simd: unknown arguments '${ARG_UNPARSED_ARGUMENTS}', "
      "missing values '${ARG_KEYWORDS_MISSING_VALUES}'")
  endif()
  if(NOT ARG_TARGET OR NOT ARG_ISA)
    message(FATAL_ERROR "hyle_target_simd requires TARGET and ISA")
  endif()
  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_target_simd: target '${ARG_TARGET}' does not exist")
  endif()
  _hyle_simd_expert_target(_expert_target)
  string(TOUPPER "${ARG_ISA}" _isa)
  _hyle_simd_target_processor(_target_processor)
  set(_is_x64 FALSE)
  set(_is_arm64 FALSE)
  set(_is_riscv64 FALSE)
  if(_target_processor MATCHES "x86_64|AMD64|amd64")
    set(_is_x64 TRUE)
  elseif(_target_processor MATCHES "arm64|aarch64|ARM64")
    set(_is_arm64 TRUE)
  elseif(_target_processor MATCHES "riscv64|RISCV64")
    set(_is_riscv64 TRUE)
  endif()

  if((_isa MATCHES "^(SSE4|AVX2|AVX512)$" AND NOT _is_x64) OR
     (_isa MATCHES "^(NEON|SVE)$" AND NOT _is_arm64) OR
     (_isa STREQUAL "SVE" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
    (_isa STREQUAL "RVV" AND NOT _is_riscv64))
    message(FATAL_ERROR
      "hyle_target_simd: ISA '${_isa}' is incompatible with ${_target_processor}")
  endif()

  set(_define "")
  set(_tier SCALAR)
  if(_isa STREQUAL "SCALAR")
    # No target macro and no extra flags: provider headers select the portable
    # scalar floor and the generated body keeps its baseline codegen contract.
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

  # Highway intentionally blocks AVX3 under classic cl.exe. Refuse to label an
  # AVX2 fallback as an exact AVX-512 source kernel; use clang-cl or AVX2 until
  # the owning ticket admits a qualified specialty implementation.
  if(_isa STREQUAL "AVX512" AND MSVC AND
     CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR
      "hyle_target_simd(ISA AVX512) cannot use locked Highway with classic cl.exe; "
      "use clang-cl or select AVX2 (no silent target downgrade)")
  endif()

  if(_define)
    target_compile_definitions(${ARG_TARGET} PRIVATE ${_define})
  endif()
  target_compile_definitions(${ARG_TARGET} PRIVATE
    HYLE_SIMD_EXACT=1 HYLE_SIMD_EXACT_TARGET_${_isa}=1
    HYLE_SIMD_EXPERT=1)
  if(_isa MATCHES "^(SSE4|AVX2|AVX512)$")
    target_compile_definitions(${ARG_TARGET} PRIVATE
      HWY_DISABLE_PCLMUL_AES=1)
    if(_isa STREQUAL "SSE4")
      target_compile_definitions(${ARG_TARGET} PRIVATE HWY_WANT_SSE4=1)
    endif()
  endif()
  if(_isa STREQUAL "SCALAR")
    target_compile_definitions(${ARG_TARGET} PRIVATE HWY_COMPILE_ONLY_SCALAR=1)
  else()
    target_compile_definitions(${ARG_TARGET} PRIVATE HWY_COMPILE_ONLY_STATIC=1)
  endif()
  target_link_libraries(${ARG_TARGET} PRIVATE ${_expert_target})
  hyle_highway_isa_flags(_flags "${_tier}")
  if(_flags)
    target_compile_options(${ARG_TARGET} PRIVATE ${_flags})
  endif()
endfunction()

# Exact-deployment mode for the HYLE library. Per-ISA implementation TUs keep their own backend
# macro; baseline dispatch TUs receive a compile-time selection and matching minimum-ISA flags.
function(hyle_exact_simd_library)
  cmake_parse_arguments(ARG "" "TARGET;ISA" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_exact_simd_library: unknown arguments '${ARG_UNPARSED_ARGUMENTS}', "
      "missing values '${ARG_KEYWORDS_MISSING_VALUES}'")
  endif()
  if(NOT ARG_TARGET OR NOT ARG_ISA OR NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_exact_simd_library requires an existing TARGET and ISA")
  endif()
  string(TOUPPER "${ARG_ISA}" _isa)
  if((_isa MATCHES "^(SSE4|AVX2|AVX512)$" AND
      NOT HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64") OR
     (_isa MATCHES "^(NEON|SVE)$" AND
      NOT HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64") OR
     (_isa STREQUAL "SVE" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux") OR
     (_isa STREQUAL "RVV" AND NOT HYLE_TARGET_PROCESSOR MATCHES "riscv64|RISCV64"))
    message(FATAL_ERROR
      "hyle_exact_simd_library: ISA '${_isa}' is incompatible with ${HYLE_TARGET_PROCESSOR}")
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
  if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
     HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND
     (_isa STREQUAL "SCALAR" OR _isa STREQUAL "SSE4"))
    # CL 19.50 may otherwise emit guarded AVX loop versions even without /arch:AVX. Exact
    # scalar/SSE artifacts promise that higher-tier instructions are absent, not merely guarded.
    target_compile_options(${ARG_TARGET} PRIVATE /arch:SSE2)
  elseif(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND
         _isa STREQUAL "SCALAR")
    set(_scalar_flags -march=x86-64 -mssse3 -mno-avx -mno-avx2
                      -mno-avx512f)
    if(MSVC)
      _hyle_forward_clang_cl_flags(_scalar_flags ${_scalar_flags})
    endif()
    target_compile_options(${ARG_TARGET} PRIVATE ${_scalar_flags})
  elseif(NOT _isa STREQUAL "SCALAR")
    hyle_isa_flags(_flags "${_tier}")
    if(_flags)
      target_compile_options(${ARG_TARGET} PRIVATE ${_flags})
    endif()
  endif()
endfunction()

function(hyle_add_dispatched_kernel)
  cmake_parse_arguments(ARG "HIGHWAY;XSIMD;OWN_TARGET_SET" "TARGET;NAME;IMPL" "TARGETS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_dispatched_kernel: unknown arguments '${ARG_UNPARSED_ARGUMENTS}', "
      "missing values '${ARG_KEYWORDS_MISSING_VALUES}'")
  endif()

  if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_IMPL OR NOT ARG_TARGETS)
    message(FATAL_ERROR "hyle_add_dispatched_kernel requires TARGET, NAME, IMPL, and TARGETS")
  endif()
  if(NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR "hyle_add_dispatched_kernel: target '${ARG_TARGET}' does not exist")
  endif()
  if(NOT COMMAND hyle_set_isa)
    message(FATAL_ERROR "hyle_add_dispatched_kernel requires hyle_set_isa from HyleIsa.cmake")
  endif()
  if(NOT ARG_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "hyle_add_dispatched_kernel: NAME '${ARG_NAME}' is not an identifier")
  endif()
  list(FIND ARG_TARGETS "scalar" _scalar_index)
  if(_scalar_index EQUAL -1)
    message(FATAL_ERROR "hyle_add_dispatched_kernel: TARGETS must include scalar")
  endif()

  _hyle_simd_target_processor(_target_processor)
  set(_is_x64 FALSE)
  set(_is_arm64 FALSE)
  set(_is_riscv64 FALSE)
  if(_target_processor MATCHES "x86_64|AMD64|amd64")
    set(_is_x64 TRUE)
  elseif(_target_processor MATCHES "arm64|aarch64|ARM64")
    set(_is_arm64 TRUE)
  elseif(_target_processor MATCHES "riscv64|RISCV64")
    set(_is_riscv64 TRUE)
  endif()
  if(NOT ARG_OWN_TARGET_SET)
    get_property(_manifest_variants GLOBAL PROPERTY HYLE_SIMD_COMPILED_VARIANTS)
  endif()

  if(IS_ABSOLUTE "${ARG_IMPL}")
    set(_impl "${ARG_IMPL}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_IMPL}")
    set(_impl "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_IMPL}")
  elseif(EXISTS "${PROJECT_SOURCE_DIR}/${ARG_IMPL}")
    set(_impl "${PROJECT_SOURCE_DIR}/${ARG_IMPL}")
  else()
    message(FATAL_ERROR
      "hyle_add_dispatched_kernel: IMPL '${ARG_IMPL}' does not exist relative to "
      "${CMAKE_CURRENT_SOURCE_DIR} or ${PROJECT_SOURCE_DIR}")
  endif()
  file(TO_CMAKE_PATH "${_impl}" _impl)

  if(ARG_HIGHWAY)
    if(TARGET hyle_highway_headers)
      # HYLE's own source build keeps provider targets build-only so compiled
      # libraries do not retain a package dependency after instantiation.
      target_link_libraries(${ARG_TARGET} PRIVATE
        "$<BUILD_INTERFACE:hyle_highway_headers>")
    else()
      _hyle_simd_expert_target(_expert_target)
      target_link_libraries(${ARG_TARGET} PRIVATE ${_expert_target})
    endif()
  endif()
  if(ARG_XSIMD)
    if(TARGET hyle_xsimd_headers)
      target_link_libraries(${ARG_TARGET} PRIVATE
        "$<BUILD_INTERFACE:hyle_xsimd_headers>")
    else()
      _hyle_simd_expert_target(_expert_target)
      target_link_libraries(${ARG_TARGET} PRIVATE ${_expert_target})
    endif()
  endif()

  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/simd-dispatch/${ARG_NAME}")
  file(MAKE_DIRECTORY "${_generated_dir}")

  foreach(_variant IN LISTS ARG_TARGETS)
    set(_target_define "")
    set(_tier "")
    set(_supported TRUE)
    set(_highway_classic_avx512_fallback FALSE)

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
      endif()
    elseif(_variant STREQUAL "rvv")
      if(NOT _is_riscv64)
        set(_supported FALSE)
      else()
        set(_target_define HYLE_SIMD_TARGET_RVV)
        set(_tier RVV)
      endif()
    else()
      message(FATAL_ERROR "hyle_add_dispatched_kernel: unknown target '${_variant}'")
    endif()

    # Official Highway blocks every AVX3 target under classic cl.exe. Product
    # families still generate their AVX-512-named leg with an AVX2 Highway body,
    # so their private runtime tables can select a lower tier without a wrapper
    # call. Source-fat remains capped at AVX2; its exact AVX-512 request is
    # rejected by hyle_add_fat_kernel instead.
    if(ARG_HIGHWAY AND _variant STREQUAL "avx512" AND MSVC AND
       CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(_target_define HYLE_SIMD_TARGET_AVX2)
      set(_tier AVX2)
      set(_highway_classic_avx512_fallback TRUE)
      target_compile_definitions(${ARG_TARGET} PRIVATE
        HYLE_SIMD_FAT_AVX512_CAPPED=1)
    endif()

    if(NOT _supported)
      continue()
    endif()
    if(_manifest_variants)
      list(FIND _manifest_variants "${_variant}" _manifest_index)
      if(_manifest_index EQUAL -1)
        continue()
      endif()
    endif()

    # Exact manifests deliberately omit every foreign leg. Do not mutate the
    # consumer target until after that membership decision: defining HAS_SVE or
    # HAS_RVV for an exact NEON/scalar target makes baseline dispatchers refer to
    # symbols that were never generated.
    if(_variant STREQUAL "sve")
      target_compile_definitions(${ARG_TARGET} PRIVATE HYLE_SIMD_HAS_SVE_VARIANT=1)
    elseif(_variant STREQUAL "rvv")
      target_compile_definitions(${ARG_TARGET} PRIVATE HYLE_SIMD_HAS_RVV_VARIANT=1)
    endif()

    set(_source "${_generated_dir}/${ARG_NAME}_${_variant}.cc")
    set(_contents "// Generated by hyle_add_dispatched_kernel; do not edit.\n")
    if(ARG_HIGHWAY)
      if(_variant STREQUAL "scalar")
        string(APPEND _contents "#define HWY_COMPILE_ONLY_SCALAR 1\n")
        set(_expected "HWY_STATIC_TARGET != HWY_SCALAR")
      else()
        string(APPEND _contents "#define HWY_COMPILE_ONLY_STATIC 1\n")
        if(_variant STREQUAL "sse4")
          string(APPEND _contents
            "#define HWY_DISABLE_PCLMUL_AES 1\n"
            "#define HWY_WANT_SSE4 1\n")
          set(_expected "HWY_STATIC_TARGET != HWY_SSE4")
        elseif(_variant STREQUAL "avx2")
          string(APPEND _contents "#define HWY_DISABLE_PCLMUL_AES 1\n")
          set(_expected "HWY_STATIC_TARGET != HWY_AVX2")
        elseif(_variant STREQUAL "avx512" AND
               _highway_classic_avx512_fallback)
          string(APPEND _contents "#define HWY_DISABLE_PCLMUL_AES 1\n")
          set(_expected "HWY_STATIC_TARGET != HWY_AVX2")
        elseif(_variant STREQUAL "avx512")
          string(APPEND _contents "#define HWY_DISABLE_PCLMUL_AES 1\n")
          set(_expected "HWY_STATIC_TARGET != HWY_AVX3")
        elseif(_variant STREQUAL "neon")
          set(_expected
            "HWY_STATIC_TARGET != HWY_NEON && HWY_STATIC_TARGET != HWY_NEON_WITHOUT_AES")
        elseif(_variant STREQUAL "sve")
          set(_expected "HWY_STATIC_TARGET != HWY_SVE")
        elseif(_variant STREQUAL "rvv")
          set(_expected "HWY_STATIC_TARGET != HWY_RVV")
        endif()
      endif()
      string(APPEND _contents
        "#include \"hwy/detect_targets.h\"\n"
        "#if ${_expected}\n"
        "#error \"HYLE fat kernel compiler flags selected the wrong Highway target\"\n"
        "#endif\n")
    endif()
    if(_target_define)
      string(APPEND _contents "#define ${_target_define}\n")
    endif()
    string(APPEND _contents "#define HYLE_SIMD_VARIANT_SUFFIX ${_variant}\n")
    string(APPEND _contents "#include \"${_impl}\"\n")
    string(APPEND _contents "#undef HYLE_SIMD_VARIANT_SUFFIX\n")
    if(_target_define)
      string(APPEND _contents "#undef ${_target_define}\n")
    endif()
    # This is configure output, not a build byproduct. Marking it GENERATED makes `ninja clean`
    # delete it without creating a rule that can restore it; file(CONFIGURE) plus an ordinary
    # source keeps clean/rebuild self-contained.
    file(CONFIGURE OUTPUT "${_source}" CONTENT "${_contents}" @ONLY)

    target_sources(${ARG_TARGET} PRIVATE "${_source}")
    if(ARG_HIGHWAY)
      if(_variant STREQUAL "scalar")
        hyle_set_highway_target_isa("${_source}" SCALAR)
      else()
        hyle_set_highway_target_isa("${_source}" "${_tier}")
      endif()
    elseif(_tier)
      hyle_set_isa("${_source}" "${_tier}")
    endif()
  endforeach()
endfunction()

# Public spelling for C++ consumers. IMPL defines suffixed pointer/POD entry points using
# HYLE_SIMD_VARIANT_SUFFIX and may instantiate a compile-time Highway/xsimd/HYLE route; pair it
# with HYLE_SIMD_SELECT_FAT_KERNEL in the baseline TU.
# In an exact build the manifest emits only one variant. Mark the consumer target so the
# selection macro preprocesses away references to variants that intentionally do not exist.
function(hyle_add_fat_kernel)
  cmake_parse_arguments(ARG "" "TARGET;NAME;IMPL" "TARGETS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_fat_kernel: unknown arguments '${ARG_UNPARSED_ARGUMENTS}', "
      "missing values '${ARG_KEYWORDS_MISSING_VALUES}'")
  endif()
  if(NOT ARG_TARGET OR NOT TARGET ${ARG_TARGET})
    message(FATAL_ERROR
      "hyle_add_fat_kernel requires an existing TARGET")
  endif()
  _hyle_simd_expert_target(_expert_target)
  _hyle_platform_factor_target(_platform_factor_target)
  target_link_libraries(${ARG_TARGET} PRIVATE
    ${_expert_target} ${_platform_factor_target})
  _hyle_simd_target_processor(_target_processor)
  _hyle_simd_expert_exact_target(_exact_target)
  if(_exact_target)
    string(TOLOWER "${_exact_target}" _exact_lower)
    list(FIND ARG_TARGETS "${_exact_lower}" _exact_index)
    if(_exact_index EQUAL -1)
      message(FATAL_ERROR
        "hyle_add_fat_kernel: exact target '${_exact_lower}' is absent from TARGETS")
    endif()
    if(_exact_lower STREQUAL "avx512" AND MSVC AND
       CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      message(FATAL_ERROR
        "hyle_add_fat_kernel exact AVX512 cannot use locked Highway with classic "
        "cl.exe; use clang-cl or select AVX2")
    endif()
  else()
    set(_required scalar)
    if(_target_processor MATCHES "x86_64|AMD64|amd64")
      list(APPEND _required sse4 avx2 avx512)
    elseif(_target_processor MATCHES "arm64|aarch64|ARM64")
      list(APPEND _required neon)
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        list(APPEND _required sve)
      endif()
    elseif(_target_processor MATCHES "riscv64|RISCV64")
      list(APPEND _required rvv)
    endif()
    foreach(_variant IN LISTS _required)
      list(FIND ARG_TARGETS "${_variant}" _variant_index)
      if(_variant_index EQUAL -1)
        message(FATAL_ERROR
          "hyle_add_fat_kernel: fat ${CMAKE_SYSTEM_NAME}/${_target_processor} "
          "build requires '${_variant}' in TARGETS")
      endif()
    endforeach()
  endif()

  hyle_add_dispatched_kernel(HIGHWAY XSIMD ${ARGN})
  if(_exact_target)
    string(TOUPPER "${_exact_target}" _exact)
    target_compile_definitions(${ARG_TARGET} PRIVATE
      HYLE_SIMD_FAT_EXACT_VARIANT_${_exact}=1
      HYLE_SIMD_EXACT=1
      HYLE_SIMD_EXACT_TARGET_${_exact}=1)
  endif()
endfunction()
