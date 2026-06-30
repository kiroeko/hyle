# container: header-only templates (bounded_queue / ws_fifo / ws deque) + the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash single-header (XXH_INLINE_ALL); that include dir is PRIVATE so xxhash
# never leaks to consumers. string_algo.cc is the dispatched utf8_length entry
# (scalar by default; borrows simdutf when HYLE_SIMD_SIMDUTF, see below).
hyle_add_module(container SOURCES hash.cc carray.cc string_buf.cc path.cc swiss_table.cc string_algo.cc PUBLIC_DEPS hyle_mem)
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash")

# --- simdutf-accelerated utf8_length (opt-in borrow, 借核不手搓) ---------------
# utf8_length's vectorized tier BORROWS simdutf's verified validate+count cores
# (the same opt-in vendor-core borrow simd/math uses for SLEEF/Highway/xsimd).
# Off by default (simdutf is a heavy ~2.8MB amalgamation; the scalar Utf8Decoder
# serves the default, keeping the container module light). Enable with
# -DHYLE_SIMD_SIMDUTF=ON + -DHYLE_SIMDUTF_DIR=<simdutf/singleheader> (containing
# simdutf.h + simdutf.cpp, produced by simdutf's singleheader/amalgamate.py). The
# .cpp is compiled PRIVATE into hyle_container; simdutf does its own runtime ISA
# dispatch, so there is no SIGILL-safety gate needed here. string_algo.cc falls
# back to the scalar core when this is not set.
option(HYLE_SIMD_SIMDUTF "borrow simdutf for vectorized utf8_length (needs HYLE_SIMDUTF_DIR)" OFF)
if(HYLE_SIMD_SIMDUTF)
  set(HYLE_SIMDUTF_DIR "${PROJECT_SOURCE_DIR}/../simdutf/singleheader" CACHE PATH "simdutf singleheader dir")
  if(NOT EXISTS "${HYLE_SIMDUTF_DIR}/simdutf.h")
    message(FATAL_ERROR "HYLE_SIMD_SIMDUTF=ON but no simdutf.h at ${HYLE_SIMDUTF_DIR}")
  endif()
  target_compile_definitions(hyle_container PRIVATE HYLE_SIMD_SIMDUTF)
  target_include_directories(hyle_container SYSTEM PRIVATE "${HYLE_SIMDUTF_DIR}")
  target_sources(hyle_container PRIVATE "${HYLE_SIMDUTF_DIR}/simdutf.cpp")
endif()
