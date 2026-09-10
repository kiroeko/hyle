# G01 Base64 evidence: a narrow same-provider adapter-tax contract. The peer
# includes source-pinned declarations and resolves the byte-identical simdutf
# implementation already carried by the static container_text factor.
option(HYLE_CONTAINER_TEXT_BASE64_P1_BENCH
  "build the Container Text v3 Base64 evidence adapter" OFF)
if(HYLE_CONTAINER_TEXT_BASE64_P1_BENCH)
  if(NOT HYLE_STATIC)
    message(FATAL_ERROR
      "HYLE_CONTAINER_TEXT_BASE64_P1_BENCH requires HYLE_STATIC")
  endif()
  set(HYLE_SIMDUTF_PEER_DIR "${HYLE_SOURCE_DIR}/../simdutf" CACHE PATH
      "source-pinned simdutf checkout for Container Text Base64 evidence")
  if(NOT EXISTS "${HYLE_SIMDUTF_PEER_DIR}/include/simdutf.h")
    message(FATAL_ERROR
      "HYLE_CONTAINER_TEXT_BASE64_P1_BENCH requires a source-pinned simdutf "
      "checkout at HYLE_SIMDUTF_PEER_DIR")
  endif()

  function(_hyle_configure_container_text_base64_bench target)
    target_link_libraries(${target} PRIVATE hyle_container_text nanobench)
    target_include_directories(${target} SYSTEM BEFORE PRIVATE
      "${HYLE_SIMDUTF_PEER_DIR}/include")
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_container_text_base64_p1_bench
    base64_product_bench.cc)
  _hyle_configure_container_text_base64_bench(
    hyle_container_text_base64_p1_bench)
  add_executable(hyle_container_text_base64_p1_bench_placement
    base64_product_bench.cc)
  _hyle_configure_container_text_base64_bench(
    hyle_container_text_base64_p1_bench_placement)
  if(MSVC)
    target_link_options(hyle_container_text_base64_p1_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      hyle_container_text_base64_p1_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Container Text Base64 placement perturbation is unsupported")
  endif()

  hyle_register_benches(container_text
    hyle_container_text_base64_p1_bench
    hyle_container_text_base64_p1_bench_placement)
endif()
