# container: header-only templates (bounded_queue / ws_fifo / ws deque) + the
# compiled hash unit (container-02). hash.cc privately includes the vendored
# xxhash single-header (XXH_INLINE_ALL); that include dir is PRIVATE so xxhash
# never leaks to consumers.
hyle_add_module(container SOURCES hash.cc carray.cc string_buf.cc path.cc PUBLIC_DEPS hyle_mem)
target_include_directories(hyle_container PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash")
