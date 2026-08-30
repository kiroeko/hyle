option(HYLE_TEST_SIMD_AVX512_RUNNABLE
  "register AVX-512 ordering tests for direct execution on this host" OFF)
option(HYLE_TEST_SIMD_SVE
  "build the opt-in SVE product variant on an SVE-capable toolchain" OFF)
option(HYLE_TEST_SIMD_RVV
  "build the opt-in RVV product variant on an RVV-capable toolchain" OFF)
mark_as_advanced(
  HYLE_TEST_SIMD_AVX512_RUNNABLE
  HYLE_TEST_SIMD_SVE
  HYLE_TEST_SIMD_RVV)

if(TARGET hyle_simd)
  # Public pointer/count and runtime contracts. Register-provider correctness
  # belongs to Highway plus the finite winner tests below; the removed HYLE
  # vec/native backend is no longer rebuilt as a parallel SIMD library.
  hyle_add_test(simd SOURCES
                dispatch_test.cc runtime_test.cc ops_test.cc search_test.cc
                scan_test.cc dtype_test.cc math_test.cc sort_test.cc
                radix_sort_test.cc bitpack_test.cc narrow_ops_test.cc)
  hyle_add_test(simd NAME provider_route SOURCES provider_route_test.cc)

  # Compile-time provider selection must disappear in final object code.
  if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(_hyle_provider_codegen_isa AVX2)
  elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(_hyle_provider_codegen_isa NEON)
  else()
    set(_hyle_provider_codegen_isa "")
  endif()
  if(_hyle_provider_codegen_isa)
    add_library(hyle_simd_provider_codegen_probe OBJECT provider_codegen_probe.cc)
    set_target_properties(hyle_simd_provider_codegen_probe PROPERTIES
      FOLDER "hyle/validation/simd")
    hyle_target_simd(TARGET hyle_simd_provider_codegen_probe
                     ISA ${_hyle_provider_codegen_isa})
    add_dependencies(hyle_simd_tests hyle_simd_provider_codegen_probe)
    add_test(NAME hyle_simd_provider_codegen_test
      COMMAND "${Python3_EXECUTABLE}"
        "${HYLE_SOURCE_DIR}/scripts/simd-provider-codegen-audit.py"
        --object "$<TARGET_OBJECTS:hyle_simd_provider_codegen_probe>"
        --architecture "${HYLE_TARGET_PROCESSOR}"
        --configuration "$<CONFIG>")
    set_tests_properties(hyle_simd_provider_codegen_test PROPERTIES
      LABELS "simd;codegen")
  endif()
  unset(_hyle_provider_codegen_isa)

  # Every material HYLE exception has a finite target-native owner independent
  # of the deleted legacy register stack.
  if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    hyle_add_test(simd NAME provider_winner_avx2
                  SOURCES provider_winner_test.cc)
    hyle_target_simd(TARGET hyle_simd_provider_winner_avx2_test ISA AVX2)
  elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    hyle_add_test(simd NAME provider_winner_neon
                  SOURCES provider_winner_test.cc)
    hyle_target_simd(TARGET hyle_simd_provider_winner_neon_test ISA NEON)
  endif()

  target_compile_definitions(hyle_simd PRIVATE HYLE_SIMD_INTERNAL_TESTING=1)
  target_compile_definitions(hyle_simd_test PRIVATE HYLE_SIMD_INTERNAL_TESTING=1)
  target_include_directories(hyle_simd_test PRIVATE "${PROJECT_SOURCE_DIR}")
  hyle_add_fat_kernel(
    TARGET hyle_simd_test
    NAME public_fat_fixture
    IMPL "tests/simd/fat_kernel_fixture.h"
    TARGETS scalar sse4 avx2 avx512 neon sve rvv)

  # The retained ordering port is a HYLE whole-buffer algorithm over locked
  # Highway target-native registers. Compile/run its direct target body rather
  # than reviving the old general register test suite.
  if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    if(MSVC)
      set_property(SOURCE sort_kernel_test.cc APPEND PROPERTY COMPILE_OPTIONS /wd4702)
    endif()
    hyle_add_test(simd NAME avx2_ordering SOURCES sort_kernel_test.cc)
    hyle_target_simd(TARGET hyle_simd_avx2_ordering_test ISA AVX2)
    target_include_directories(hyle_simd_avx2_ordering_test PRIVATE
      "${PROJECT_SOURCE_DIR}")

    if(NOT (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
      if(HYLE_TEST_SIMD_AVX512_RUNNABLE)
        set(_hyle_avx512_autorun "")
      else()
        set(_hyle_avx512_autorun NO_AUTORUN)
      endif()
      hyle_add_test(simd NAME avx512_ordering SOURCES sort_kernel_test.cc
                    ${_hyle_avx512_autorun})
      hyle_target_simd(TARGET hyle_simd_avx512_ordering_test ISA AVX512)
      target_include_directories(hyle_simd_avx512_ordering_test PRIVATE
        "${PROJECT_SOURCE_DIR}")
      unset(_hyle_avx512_autorun)
    endif()
  elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    hyle_add_test(simd NAME neon_ordering SOURCES sort_kernel_test.cc)
    hyle_target_simd(TARGET hyle_simd_neon_ordering_test ISA NEON)
    target_include_directories(hyle_simd_neon_ordering_test PRIVATE
      "${PROJECT_SOURCE_DIR}")

    if(HYLE_TEST_SIMD_SVE)
      hyle_add_test(simd NAME sve_product
                    SOURCES scalable_math_product_test.cc
                            scalable_math_dispatch_test.cc
                            scalable_dtype_product_test.cc
                            scalable_bitpack_product_test.cc
                            scalable_sort_product_test.cc
                    DEFINES HYLE_SIMD_HAS_SVE_VARIANT=1)
      hyle_target_simd(TARGET hyle_simd_sve_product_test ISA SVE)
      target_include_directories(hyle_simd_sve_product_test PRIVATE
        "${PROJECT_SOURCE_DIR}")
    endif()
  elseif(HYLE_TARGET_PROCESSOR MATCHES "riscv64|riscv")
    if(HYLE_TEST_SIMD_RVV)
      hyle_add_test(simd NAME rvv_product
                    SOURCES scalable_math_product_test.cc
                            scalable_math_dispatch_test.cc
                            scalable_dtype_product_test.cc
                            scalable_bitpack_product_test.cc
                            scalable_sort_product_test.cc
                    DEFINES HYLE_SIMD_HAS_RVV_VARIANT=1)
      hyle_target_simd(TARGET hyle_simd_rvv_product_test ISA RVV)
      target_include_directories(hyle_simd_rvv_product_test PRIVATE
        "${PROJECT_SOURCE_DIR}")
    endif()
  endif()
endif()
