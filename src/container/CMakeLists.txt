# container: owning/concurrent templates plus compiled algorithms. The
# allocation-free intrusive-list surface is aggregated transitively through the
# public hyle::container_intrusive dependency declared in HyleComponents.cmake.
# SPSC / bounded_queue / ws_fifo / ws deque remain header-only here; the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash + wyhash single-headers (best-of-breed byte hasher: XXH3 short-mid, wyhash
# long); those include dirs are PRIVATE so neither leaks to consumers.
# string_algo.cc provides dispatched UTF-8 validate/count entries by BORROWING
# simdutf's verified SIMD core.
# string_ref.cc holds the std forwards (substring find/rfind + char-set find_*_of, where
# MSVC STL's vectorized asm wins) + count/rfind/ASCII-case DISPATCHERS. These kernels use
# simd/vec.h (2026-07): kernels live in str_search_{scalar,sse4,avx2,avx512,neon}.cc (shared
# body str_search_impl.h, written against the D-tag abstraction), and string_ref.cc picks the
# widest by best_isa() through a cached fn ptr -- same variant-TU + runtime-dispatch model as
# simd/ops. DynamicBitSet uses the same isolation for AND/OR/XOR/and-not/not/popcount in
# bitset_ops_{scalar,sse4,avx2,avx512,neon}.cc. REPLACED the old hand-rolled
# string_ref_avx2.cc (256b, x86-only) + inline SSE2; now every ISA incl. aarch64 NEON +
# AVX-512 is covered from one source (arm64 previously fell to scalar). Each variant TU is
# its own -march, entered only when best_isa() confirms the ISA.
#
# ARCH SPLIT (Apple-Silicon 真机纪律): wide variant TUs gated by HYLE_TARGET_PROCESSOR --
# x86 ships scalar+sse4+avx2+avx512, aarch64 ships scalar+neon. Feeding -mavx2 to arm64 (or the
# x86 intrinsic headers) errors, so non-host-arch TUs are never compiled. SwissTable's NEON
# group lives in probe.h (pulled by swiss_table.cc), so container-04b NEON compiles auto on arm64.
set(_container_arch_sources "")
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_CONTAINER_X64 TRUE)
  list(APPEND _container_arch_sources
    hash_xxh3_avx2.cc str_search_sse4.cc str_search_avx2.cc str_search_avx512.cc
    bitset_ops_sse4.cc bitset_ops_avx2.cc bitset_ops_avx512.cc)
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_CONTAINER_ARM64 TRUE)
  list(APPEND _container_arch_sources str_search_neon.cc bitset_ops_neon.cc)
endif()
hyle_add_module(container
  SOURCES hash.cc carray.cc string_buf.cc path.cc swiss_table.cc string_algo.cc
          string_ref.cc dynamic_bitset.cc c_container.cc c_btree.cc str_search_scalar.cc
          bitset_ops_scalar.cc ${_container_arch_sources})
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash"
  "${PROJECT_SOURCE_DIR}")  # str_search_*.cc #include "src/container/str_search_impl.h"

# Per-variant ISA flags: each str_search kernel TU gets ONLY its target's -march; the
# dispatcher (string_ref.cc) stays baseline so it never emits a wide instruction on the
# always-taken path. Gated by arch (the wide TUs only exist in the matching build).
if(HYLE_CONTAINER_X64)
  hyle_set_isa(hash_xxh3_avx2.cc AVX2)
  hyle_set_isa(str_search_sse4.cc SSE42)
  hyle_set_isa(str_search_avx2.cc AVX2)
  hyle_set_isa(str_search_avx512.cc AVX512)
  hyle_set_isa(bitset_ops_sse4.cc SSE42)
  hyle_set_isa(bitset_ops_avx2.cc AVX2)
  hyle_set_isa(bitset_ops_avx512.cc AVX512)
elseif(HYLE_CONTAINER_ARM64)
  hyle_set_isa(str_search_neon.cc NEON)
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
