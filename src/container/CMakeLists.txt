# container: header-only templates (bounded_queue / ws_fifo / ws deque) + the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash + wyhash single-headers (best-of-breed byte hasher: XXH3 short-mid, wyhash
# long); those include dirs are PRIVATE so neither leaks to consumers. string_algo.cc
# is the dispatched utf8_length entry, which BORROWS simdutf's verified SIMD core.
# string_ref.cc holds the out-of-line SIMD search cores + std forwards. It is the
# BASELINE (SSE2) dispatcher for the self-implemented ops (count/rfind char via AVX2
# variant string_ref_avx2.cc); the ops where MSVC STL's vectorized asm wins (substring
# find/rfind, char-set find_*_of) are forwarded to std::string_view inside the .cc
# (the public header pulls in no std string headers). Each SIMD variant is its own
# -march TU, entered only when platform::cpu_features() confirms the ISA (runtime CPUID).
#
# ARCH SPLIT (Apple-Silicon 真机暴露): string_ref_avx2.cc is an x86-only variant TU (-mavx2).
# string_ref.cc already #if-gates all references to its symbols behind HYLE_ARCH_X64, so on
# aarch64 the dispatcher forwards to the scalar/std path and the AVX2 TU must NOT be compiled
# (clang rejects -mavx2 on arm64). SwissTable's NEON group lives in probe.h (a header pulled
# by swiss_table.cc), so container-04b NEON compiles automatically on arm64 -- no separate TU.
set(_container_arch_sources "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_CONTAINER_X64 TRUE)
  list(APPEND _container_arch_sources string_ref_avx2.cc)
endif()
hyle_add_module(container SOURCES hash.cc carray.cc string_buf.cc path.cc swiss_table.cc string_algo.cc string_ref.cc ${_container_arch_sources} PUBLIC_DEPS hyle_mem hyle_platform)
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash")

# The AVX2 variant TU gets 256-bit codegen; string_ref.cc (the dispatcher) stays
# baseline so it never emits an AVX2 instruction on the always-taken path. (x86 only.)
if(HYLE_CONTAINER_X64)
  hyle_set_isa(string_ref_avx2.cc AVX2)
endif()
# Keep IPO/LTO off for the module so a future global -flto can't inline the AVX2 TU's
# wider codegen into the baseline dispatcher (same isolation rule as simd/, conventions §10).
set_target_properties(hyle_container PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- simdutf-accelerated utf8_length (vendored, UNCONDITIONAL, 借核不手搓) ------
# utf8_length BORROWS simdutf's verified validate+count cores. simdutf is VENDORED in
# third_party (the single-header amalgamation: simdutf.h + simdutf.cpp) and wired
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
