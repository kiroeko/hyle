# container: owning/concurrent templates plus compiled algorithms. The
# allocation-free intrusive-list surface remains a separate internal factor;
# the consumer-facing hyle::container facade aggregates both factors.
# SPSC / bounded_queue / ws_fifo / ws deque remain header-only here; the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash + wyhash single-headers (best-of-breed byte hasher: XXH3 short-mid, wyhash
# long); those include dirs are PRIVATE so neither leaks to consumers.
# string_algo.cc provides dispatched UTF-8 validate/count entries by BORROWING
# simdutf's verified SIMD core.
# string_view.cc holds std forwards plus count/rfind/ASCII-case dispatchers;
# DynamicBitSet owns its algebra/count dispatcher. Both kernel families are
# generated from one direct-Highway implementation header per family. Container
# owns the pointer/count semantics and one cached whole-buffer call, while
# Highway owns register lowering. No hyle::simd register DSL or simd product
# target enters this component. The proven Apple-NEON CNT+UDOT bit-count body
# remains a Container-owned specialty kernel.
#
# ARCH SPLIT (Apple-Silicon 真机纪律): wide variant TUs gated by HYLE_TARGET_PROCESSOR --
# x86 ships scalar+sse4+avx2+avx512, aarch64 ships scalar+neon. Feeding -mavx2 to arm64 (or the
# x86 intrinsic headers) errors, so non-host-arch TUs are never compiled. SwissTable's NEON
# group lives in probe.h (pulled by swiss_table.cc), so container-04b NEON compiles auto on arm64.
include(HyleDispatch)

set(_container_arch_sources "")
set(_container_highway_targets scalar sse4 avx2 neon)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_CONTAINER_X64 TRUE)
  list(APPEND _container_arch_sources hash_xxh3_avx2.cc)
  if(NOT (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    list(APPEND _container_highway_targets avx512)
    set(HYLE_CONTAINER_HAS_AVX512 TRUE)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_CONTAINER_ARM64 TRUE)
  list(APPEND _container_arch_sources bitset_ops_neon.cc)
endif()
hyle_add_module(container TESTING
  SOURCES hash.cc pod_vector.cc string_buf.cc path.cc swiss_table.cc string_algo.cc
          string_view.cc dynamic_bitset.cc ${_container_arch_sources})
# probe.h exposes an SSSE3 inline route on x86 GCC/Clang-family targets. Keep
# producer and consumer TUs on one ODR/codegen shape, but let Container own the
# permission instead of leaking it through HYLE's global/Core build contract.
# Compiler selection stays in generator expressions because the target is
# exported and a Windows SDK may cross between clang-cl and cl.exe.
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(WIN32)
    target_compile_options(hyle_container PUBLIC
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:/clang:-mssse3>")
  else()
    target_compile_options(hyle_container PUBLIC
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-mssse3>")
  endif()
endif()
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash"
  "${PROJECT_SOURCE_DIR}")
if(HYLE_CONTAINER_HAS_AVX512)
  target_compile_definitions(hyle_container PRIVATE
    HYLE_CONTAINER_HAS_AVX512=1)
endif()

hyle_add_dispatched_kernel(HIGHWAY OWN_TARGET_SET
  TARGET hyle_container
  NAME container_string
  IMPL "src/container/str_search_impl.h"
  TARGETS ${_container_highway_targets})
hyle_add_dispatched_kernel(HIGHWAY OWN_TARGET_SET
  TARGET hyle_container
  NAME container_bitset
  IMPL "src/container/bitset_ops_impl.h"
  TARGETS ${_container_highway_targets})

# Per-variant ISA flags: each str_search kernel TU gets ONLY its target's -march; the
# dispatcher (string_view.cc) stays baseline so it never emits a wide instruction on the
# always-taken path. Gated by arch (the wide TUs only exist in the matching build).
if(HYLE_CONTAINER_X64)
  hyle_set_isa(hash_xxh3_avx2.cc AVX2)
elseif(HYLE_CONTAINER_ARM64)
  hyle_set_isa(bitset_ops_neon.cc NEON)
endif()
# Keep IPO/LTO off for the module so a future global -flto can't inline a variant TU's
# wider codegen into the baseline dispatcher (same isolation rule as simd/, conventions §10).
set_target_properties(hyle_container PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- simdutf-accelerated UTF-8 validate/count (vendored, UNCONDITIONAL, 借核不手搓) --
# utf8_validate BORROWS simdutf's one-pass validation core; utf8_length BORROWS its
# validate+count cores. simdutf is VENDORED in third_party (the single-header
# amalgamation: simdutf.h + simdutf.cpp) and wired
# UNCONDITIONALLY -- a perf library ships its fastest path by default (user directive: no
# opt-in gate; same stance as the hard mimalloc dep). The .cpp compiles PRIVATE into
# hyle_container; simdutf does its OWN runtime ISA dispatch internally, so there is no
# SIGILL-safety gate needed here (it picks the host's best kernel at runtime). string_algo.cc
# still keeps the scalar Utf8Decoder as the constexpr path + non-simdutf fallback.
target_compile_definitions(hyle_container PRIVATE HYLE_SIMD_SIMDUTF)
target_include_directories(hyle_container SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/simdutf")
target_sources(hyle_container PRIVATE "${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp")
# Third-party amalgamation: keep our -Wall/-Wextra/-Wpedantic (+ -Werror under CI) OFF it,
# exactly like mimalloc's static lib gets /w|-w (src/mem). Without this, simdutf.cpp's
# own warnings (e.g. -Wpedantic zero-size-array) would break a HYLE_WERROR=ON build. The
# SYSTEM include dir above only silences warnings from its HEADERS, not from compiling the
# .cpp TU itself -- that needs a per-source flag.
if(MSVC)
  set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp"
    PROPERTIES COMPILE_OPTIONS "/w")
else()
  set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp"
    PROPERTIES COMPILE_OPTIONS "-w")
endif()

hyle_register_package_dependency(
  NAME container_vendors
  OWNER container
  LOCKED_DEPENDENCIES xxhash wyhash simdutf
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/xxhash/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|xxhash-LICENSE"
    "${PROJECT_SOURCE_DIR}/third_party/wyhash/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|wyhash-LICENSE"
    "${PROJECT_SOURCE_DIR}/third_party/simdutf/LICENSE-MIT|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|simdutf-LICENSE-MIT"
    "${PROJECT_SOURCE_DIR}/third_party/simdutf/LICENSE-APACHE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|simdutf-LICENSE-APACHE")
