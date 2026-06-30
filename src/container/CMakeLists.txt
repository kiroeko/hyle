# container: header-only templates (bounded_queue / ws_fifo / ws deque) + the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash single-header (XXH_INLINE_ALL); that include dir is PRIVATE so xxhash
# never leaks to consumers. string_algo.cc is the dispatched utf8_length entry,
# which BORROWS simdutf's verified SIMD validate+count core (wired below).
hyle_add_module(container SOURCES hash.cc carray.cc string_buf.cc path.cc swiss_table.cc string_algo.cc PUBLIC_DEPS hyle_mem)
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash")

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
