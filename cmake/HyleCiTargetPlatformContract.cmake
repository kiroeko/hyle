cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
  # These values are embedded into generated CMake source below. Native
  # Windows backslashes would otherwise become escapes such as \h, even though
  # the same script happens to work when called through the master contract.
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()
file(MAKE_DIRECTORY "${HYLE_SMOKE_ROOT}")

set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND
   NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

function(_hyle_target_contract_run label expect_success)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  set(HYLE_TARGET_CONTRACT_LAST_OUTPUT "${_output}" PARENT_SCOPE)
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly failed:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded")
  endif()
endfunction()

function(_hyle_pointer_width_probe width expect_success)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -DCMAKE_SIZEOF_VOID_P=${width}
      -P "${HYLE_SOURCE_DIR}/cmake/HyleTargetPlatform.cmake"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "64-bit target-platform contract unexpectedly failed:\n${_output}")
    endif()
  elseif(_result EQUAL 0)
    message(FATAL_ERROR
      "${width}-byte target pointer unexpectedly passed the 64-bit contract")
  elseif(NOT _output MATCHES "requires a 64-bit target ABI")
    message(FATAL_ERROR
      "32-bit target rejection lost its actionable diagnostic:\n${_output}")
  endif()
endfunction()

_hyle_pointer_width_probe(8 TRUE)
_hyle_pointer_width_probe(4 FALSE)

# Visual Studio's multi-platform generator reports the host through
# CMAKE_SYSTEM_PROCESSOR even when `-A` selected a different compiler target.
# Keep the normalized project identity under a script-level regression gate so
# later source/component decisions cannot silently fall back to the host ISA.
function(_hyle_processor_identity_probe label compiler_arch generator_arch
         system_arch expected)
  string(MAKE_C_IDENTIFIER "${label}" _label_id)
  set(_probe "${HYLE_SMOKE_ROOT}/target-identity-${_label_id}.cmake")
  file(WRITE "${_probe}"
"cmake_minimum_required(VERSION 3.25)\n"
"set(CMAKE_SIZEOF_VOID_P 8)\n"
"set(MSVC_CXX_ARCHITECTURE_ID \"${compiler_arch}\")\n"
"set(CMAKE_GENERATOR_PLATFORM \"${generator_arch}\")\n"
"set(CMAKE_SYSTEM_PROCESSOR \"${system_arch}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleTargetPlatform.cmake\")\n"
"if(NOT HYLE_TARGET_PROCESSOR STREQUAL \"${expected}\")\n"
"  message(FATAL_ERROR \"expected target '${expected}', got '\${HYLE_TARGET_PROCESSOR}'\")\n"
"endif()\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -P "${_probe}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "target identity '${label}' failed:\n${_stdout}\n${_stderr}")
  endif()
endfunction()

_hyle_processor_identity_probe(
  "vs-arm64-over-amd64-host" "ARM64" "ARM64" "AMD64" "aarch64")
_hyle_processor_identity_probe(
  "vs-x64" "AMD64" "x64" "AMD64" "x86_64")
_hyle_processor_identity_probe(
  "generator-arm64" "" "ARM64" "AMD64" "aarch64")
_hyle_processor_identity_probe(
  "toolchain-riscv64" "" "" "riscv64" "riscv64")

# A platform without fcontext still supports the sched-free io proactor and
# host/CUDA/Vulkan compute.  Coroutine-only protocol/ggml features default OFF,
# AUTO TLS skips OpenSSL discovery, and an explicit impossible ON must fail
# rather than being written into installed feature metadata.
set(_hyle_unsupported_feature_script
  "${HYLE_SMOKE_ROOT}/unsupported-fcontext-feature-contract.cmake")
file(WRITE "${_hyle_unsupported_feature_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_FCONTEXT_SUPPORTED FALSE)\n"
"set(CMAKE_SYSTEM_NAME ContractOS)\n"
"set(CMAKE_SYSTEM_PROCESSOR rv64)\n"
"include(HyleFeatureOption)\n"
"if(HYLE_FEATURE_CASE STREQUAL \"http2-on\")\n"
"  set(HYLE_IO_HTTP2 ON CACHE BOOL \"\")\n"
"elseif(HYLE_FEATURE_CASE STREQUAL \"tls-on\")\n"
"  set(HYLE_IO_TLS ON CACHE STRING \"\")\n"
"elseif(HYLE_FEATURE_CASE STREQUAL \"ggml-on\")\n"
"  set(HYLE_COMPUTE_GGML ON CACHE BOOL \"\")\n"
"endif()\n"
"set(HYLE_COMPUTE_CUDA OFF CACHE STRING \"\")\n"
"set(HYLE_COMPUTE_VULKAN OFF CACHE BOOL \"\")\n"
"include(HyleIoDependencies)\n"
"include(HyleComputeDependencies)\n"
"if(HYLE_FEATURE_CASE STREQUAL \"defaults\")\n"
"  if(HYLE_IO_HTTP2 OR HYLE_COMPUTE_GGML)\n"
"    message(FATAL_ERROR \"unsupported fcontext features did not default OFF\")\n"
"  endif()\n"
"  hyle_configure_io_dependencies()\n"
"  hyle_configure_compute_dependencies()\n"
"  if(HYLE_IO_TLS_ENABLED)\n"
"    message(FATAL_ERROR \"TLS AUTO was advertised without fcontext\")\n"
"  endif()\n"
"elseif(HYLE_FEATURE_CASE MATCHES \"^(http2-on|tls-on)$\")\n"
"  hyle_configure_io_dependencies()\n"
"elseif(HYLE_FEATURE_CASE STREQUAL \"ggml-on\")\n"
"  hyle_configure_compute_dependencies()\n"
"else()\n"
"  message(FATAL_ERROR \"unknown feature contract case\")\n"
"endif()\n")

function(_hyle_unsupported_feature_probe case expect_success expected_message)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -DHYLE_FEATURE_CASE=${case}
      -P "${_hyle_unsupported_feature_script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR
      "unsupported-fcontext '${case}' unexpectedly failed:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR
      "unsupported-fcontext '${case}' unexpectedly succeeded")
  elseif(NOT expect_success AND NOT _output MATCHES "${expected_message}")
    message(FATAL_ERROR
      "unsupported-fcontext '${case}' failed for the wrong reason:\n${_output}")
  endif()
endfunction()

_hyle_unsupported_feature_probe(defaults TRUE "")
_hyle_unsupported_feature_probe(http2-on FALSE
  "HYLE_IO_HTTP2=ON requires sched/fcontext")
_hyle_unsupported_feature_probe(tls-on FALSE
  "HYLE_IO_TLS=ON requires sched/fcontext")
_hyle_unsupported_feature_probe(ggml-on FALSE
  "HYLE_COMPUTE_GGML=ON requires sched/fcontext")

# The public io header tree is intentionally installed as one surface.  Prove
# that raw headers+library consumption derives the supported native capability,
# that the no-transport target contract leaves standalone surfaces usable, and
# that a sched-dependent header fails immediately for an incompatible/0-valued
# product definition instead of deferring to a missing linker symbol.
set(_hyle_io_header_source
  "${HYLE_SMOKE_ROOT}/io-capability-header-source")
set(_hyle_io_header_build
  "${HYLE_SMOKE_ROOT}/io-capability-header-build")
file(REMOVE_RECURSE "${_hyle_io_header_source}" "${_hyle_io_header_build}")
file(MAKE_DIRECTORY "${_hyle_io_header_source}")
file(WRITE "${_hyle_io_header_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_io_capability_header_contract LANGUAGES C CXX)\n"
"add_library(raw_supported OBJECT raw_supported.cc)\n"
"target_compile_features(raw_supported PRIVATE cxx_std_23)\n"
"target_include_directories(raw_supported PRIVATE \"${HYLE_SOURCE_DIR}/include\")\n"
"add_library(no_transport_surfaces OBJECT no_transport_surfaces.cc)\n"
"target_compile_features(no_transport_surfaces PRIVATE cxx_std_23)\n"
"target_compile_definitions(no_transport_surfaces PRIVATE HYLE_IO_HAS_COROUTINE_TRANSPORT=0)\n"
"target_include_directories(no_transport_surfaces PRIVATE \"${HYLE_SOURCE_DIR}/include\")\n"
"add_library(raw_unsupported EXCLUDE_FROM_ALL OBJECT raw_unsupported.cc)\n"
"target_compile_features(raw_unsupported PRIVATE cxx_std_23)\n"
"target_compile_definitions(raw_unsupported PRIVATE HYLE_IO_HAS_COROUTINE_TRANSPORT=0)\n"
"target_include_directories(raw_unsupported PRIVATE \"${HYLE_SOURCE_DIR}/include\")\n")
file(WRITE "${_hyle_io_header_source}/raw_supported.cc"
"#include <hyle/io/capabilities.h>\n"
"#include <hyle/io/net.h>\n"
"static_assert(HYLE_IO_HAS_COROUTINE_TRANSPORT == 1);\n"
"using CompleteFn = void (*)(const hyle::io::Completion&) noexcept;\n"
"CompleteFn raw_supported_complete = &hyle::io::complete;\n")
file(WRITE "${_hyle_io_header_source}/no_transport_surfaces.cc"
"#include <hyle/c/io.h>\n"
"#include <hyle/io/proactor.h>\n"
"#include <hyle/io/shared_memory.h>\n"
"static_assert(sizeof(hyle::io::Completion) != 0);\n"
"static_assert(sizeof(hyle_io_completion) != 0);\n")
file(WRITE "${_hyle_io_header_source}/raw_unsupported.cc"
"#include <hyle/io/net.h>\n")

_hyle_target_contract_run("configure io raw-header capability contract" TRUE
  "${CMAKE_COMMAND}" -S "${_hyle_io_header_source}"
  -B "${_hyle_io_header_build}" ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)
_hyle_target_contract_run("build supported/raw standalone io headers" TRUE
  "${CMAKE_COMMAND}" --build "${_hyle_io_header_build}" --config Release
  --parallel 2 --target raw_supported no_transport_surfaces)
_hyle_target_contract_run("reject sched-dependent header for no-transport product" FALSE
  "${CMAKE_COMMAND}" --build "${_hyle_io_header_build}" --config Release
  --parallel 2 --target raw_unsupported)
if(NOT HYLE_TARGET_CONTRACT_LAST_OUTPUT MATCHES
   "coroutine-transport target/header capability mismatch")
  message(FATAL_ERROR
    "unsupported io header failed for the wrong reason:\n"
    "${HYLE_TARGET_CONTRACT_LAST_OUTPUT}")
endif()
message(STATUS "HYLE target-platform contract passed")
