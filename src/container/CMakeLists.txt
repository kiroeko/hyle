# container: header-only templates (bounded_queue / ws_fifo / ws deque) + the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash + wyhash single-headers (best-of-breed byte hasher: XXH3 short-mid, wyhash
# long); those include dirs are PRIVATE so neither leaks to consumers. string_algo.cc
# is the dispatched utf8_length entry, which BORROWS simdutf's verified SIMD core.
# string_ref.cc holds the out-of-line SIMD search cores. It is the BASELINE (SSE2)
# dispatcher; string_ref_avx2.cc is the 256-bit variant compiled at its own -march
# (AVX2), entered only when platform::cpu_features().avx2 (runtime CPUID from the
# probe now in platform/). Public header stays free of <emmintrin.h>.
hyle_add_module(container SOURCES hash.cc carray.cc string_buf.cc path.cc swiss_table.cc string_algo.cc string_ref.cc string_ref_avx2.cc PUBLIC_DEPS hyle_mem hyle_platform)
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash")

# The AVX2 variant TU gets 256-bit codegen; string_ref.cc (the dispatcher) stays
# baseline so it never emits an AVX2 instruction on the always-taken path.
hyle_set_isa(string_ref_avx2.cc AVX2)
# Keep IPO/LTO off for the module so a future global -flto can't inline the AVX2 TU's
# 256-bit codegen into the baseline dispatcher (same isolation rule as simd/, conventions §10).
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
