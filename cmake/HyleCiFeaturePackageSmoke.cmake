# Linux correctness + package/install/downstream smoke for locked optional
# product sources. Unlike the cross-platform CPU package smoke, this enables
# the private nghttp2 closure once per CI revision and exercises its real IO
# adapter before producing static/shared SDKs. A separate small
# obs-only shape compiles and consumes the exact locked Tracy trace-on surface.
# The ordinary shared-source resolver prepares every official archive and
# expanded tree. This smoke deliberately does not carry a second downloader:
# product configure and CI must exercise the same supply-chain implementation.
# The default scope is `all`; local diagnosis may select `adapters` or `trace`
# without weakening the hosted all-scope gate.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR
      "HyleCiFeaturePackageSmoke requires -D${_required}=...")
  endif()
endforeach()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)
file(MAKE_DIRECTORY "${HYLE_SMOKE_ROOT}")

if(NOT DEFINED HYLE_FEATURE_SMOKE_SCOPE OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "")
  set(HYLE_FEATURE_SMOKE_SCOPE all)
endif()
set(_hyle_feature_smoke_scopes all adapters trace)
if(NOT HYLE_FEATURE_SMOKE_SCOPE IN_LIST _hyle_feature_smoke_scopes)
  message(FATAL_ERROR
    "HYLE_FEATURE_SMOKE_SCOPE must be all, adapters, or trace; got "
    "'${HYLE_FEATURE_SMOKE_SCOPE}'")
endif()

include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyCache.cmake")
hyle_initialize_dependency_source_cache()

function(_hyle_feature_smoke_run label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    COMMAND_ECHO STDOUT)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} failed with exit code ${_result}")
  endif()
endfunction()

set(_hyle_feature_smoke_dependencies "")
if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "adapters")
  list(APPEND _hyle_feature_smoke_dependencies
    highway nghttp2 googletest)
endif()
if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "trace")
  list(APPEND _hyle_feature_smoke_dependencies tracy)
endif()
foreach(_dependency IN LISTS _hyle_feature_smoke_dependencies)
  hyle_dependency_cache_resolve(
    "${_dependency}" "_${_dependency}_source")
endforeach()

set(_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND
   NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

# Package-only probes prove closure/export mechanics, but they cannot detect an
# adapter that still compiles while misusing the upgraded library. Build and
# run the real feature-owned test once; do not replicate the
# complete heavy test tree for both static and shared package shapes.
function(_hyle_feature_smoke_correctness)
  set(_build "${HYLE_SMOKE_ROOT}/feature-correctness-build")
  file(REMOVE_RECURSE "${_build}")

  _hyle_feature_smoke_run("configure locked feature adapter tests"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DHYLE_COMPONENTS=io
    -DHYLE_BUILD_TESTS=ON
    -DHYLE_TEST_INSTRUMENTATION=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=OFF
    -DHYLE_STATIC=OFF
    -DHYLE_WERROR=ON
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=ON
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=${_highway_source}
    -DFETCHCONTENT_SOURCE_DIR_NGHTTP2=${_nghttp2_source}
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${_googletest_source})
  _hyle_feature_smoke_run("build locked feature adapter tests"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2
    --target hyle_io_http2_test)
  _hyle_feature_smoke_run("run locked feature adapter tests"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_build}" -C Release
    -R "^hyle_io_http2_test$"
    --output-on-failure)
endfunction()

function(_hyle_feature_smoke_shape shape is_static)
  set(_build "${HYLE_SMOKE_ROOT}/feature-package-${shape}-build")
  set(_prefix "${HYLE_SMOKE_ROOT}/feature-package-${shape}-prefix")
  set(_consumer_source "${HYLE_SMOKE_ROOT}/feature-consumer-${shape}-source")
  set(_consumer_build "${HYLE_SMOKE_ROOT}/feature-consumer-${shape}-build")
  file(REMOVE_RECURSE
    "${_build}" "${_prefix}" "${_consumer_source}" "${_consumer_build}")

  _hyle_feature_smoke_run("configure locked ${shape} feature SDK"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_prefix}
    -DHYLE_COMPONENTS=io
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${is_static}
    -DHYLE_WERROR=ON
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=ON
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=${_highway_source}
    -DFETCHCONTENT_SOURCE_DIR_NGHTTP2=${_nghttp2_source})

  _hyle_feature_smoke_run("build locked ${shape} feature SDK"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2)
  _hyle_feature_smoke_run("install locked ${shape} feature SDK"
    "${CMAKE_COMMAND}" --install "${_build}" --config Release)

  set(_relocated_prefix
    "${HYLE_SMOKE_ROOT}/feature-package-${shape}-relocated")
  file(REMOVE_RECURSE "${_relocated_prefix}")
  file(RENAME "${_prefix}" "${_relocated_prefix}")
  set(_prefix "${_relocated_prefix}")

  set(_required_installs
    "share/licenses/Hyle/nghttp2-LICENSE")
  set(_static_development_installs
    "include/nghttp2/nghttp2.h"
    "include/nghttp2/nghttp2ver.h")
  if(is_static)
    list(APPEND _required_installs ${_static_development_installs})
  endif()
  foreach(_required_install IN LISTS _required_installs)
    if(NOT EXISTS "${_prefix}/${_required_install}")
      message(FATAL_ERROR
        "Locked ${shape} feature SDK omitted '${_required_install}'")
    endif()
  endforeach()
  if(NOT is_static)
    foreach(_private_install IN LISTS _static_development_installs)
      if(EXISTS "${_prefix}/${_private_install}")
        message(FATAL_ERROR
          "Shared feature SDK leaked private development asset '${_private_install}'")
      endif()
    endforeach()
  endif()
  file(MAKE_DIRECTORY "${_consumer_source}")
  file(WRITE "${_consumer_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_locked_feature_consumer LANGUAGES CXX)\n"
"find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS io)\n"
"if(NOT Hyle_STATIC STREQUAL \"${is_static}\")\n"
"  message(FATAL_ERROR \"installed linkage mismatch: Hyle_STATIC=\${Hyle_STATIC}\")\n"
"endif()\n"
"foreach(_feature IN ITEMS io_http2)\n"
"  if(NOT _feature IN_LIST Hyle_AVAILABLE_FEATURES)\n"
"    message(FATAL_ERROR \"missing installed HYLE feature: \${_feature}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_target IN ITEMS hyle::io)\n"
"  if(NOT TARGET \${_target})\n"
"    message(FATAL_ERROR \"missing installed HYLE target: \${_target}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_target IN ITEMS hyle::_nghttp2_static)\n"
"  if(${is_static} AND NOT TARGET \${_target})\n"
"    message(FATAL_ERROR \"static SDK omitted private closure target: \${_target}\")\n"
"  elseif(NOT ${is_static} AND TARGET \${_target})\n"
"    message(FATAL_ERROR \"shared SDK exported private closure target: \${_target}\")\n"
"  endif()\n"
"endforeach()\n"
"add_executable(hyle_locked_feature_consumer main.cc)\n"
"target_compile_features(hyle_locked_feature_consumer PRIVATE cxx_std_23)\n"
"target_link_libraries(hyle_locked_feature_consumer PRIVATE hyle::io)\n"
"hyle_stage_runtime_dependencies(hyle_locked_feature_consumer)\n"
"enable_testing()\n"
"add_test(NAME hyle_locked_feature_consumer COMMAND hyle_locked_feature_consumer)\n")
  file(WRITE "${_consumer_source}/main.cc"
"#include <hyle/io/http2.h>\n"
"#include <hyle/mem/allocator.h>\n"
"#if !defined(HYLE_IO_HTTP2)\n"
"#error installed io_http2 feature did not propagate its public compile contract\n"
"#endif\n"
"int main() {\n"
"  auto client = hyle::io::Http2ClientSession::create(\n"
"      hyle::mem::system_allocator(), hyle::io::TcpStream{});\n"
"  if (client.has_value()) return 1;\n"
"  return 0;\n"
"}\n")

  _hyle_feature_smoke_run("configure locked ${shape} feature SDK consumer"
    "${CMAKE_COMMAND}" -S "${_consumer_source}" -B "${_consumer_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_prefix})
  _hyle_feature_smoke_run("build locked ${shape} feature SDK consumer"
    "${CMAKE_COMMAND}" --build "${_consumer_build}" --config Release
    --parallel 2)
  _hyle_feature_smoke_run("run locked ${shape} feature SDK consumer"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}" -C Release
    --output-on-failure)
endfunction()

function(_hyle_feature_smoke_tracy shape is_static)
  set(_build "${HYLE_SMOKE_ROOT}/trace-package-${shape}-build")
  set(_prefix "${HYLE_SMOKE_ROOT}/trace-package-${shape}-prefix")
  set(_consumer_source "${HYLE_SMOKE_ROOT}/trace-consumer-${shape}-source")
  set(_consumer_build "${HYLE_SMOKE_ROOT}/trace-consumer-${shape}-build")
  file(REMOVE_RECURSE
    "${_build}" "${_prefix}" "${_consumer_source}" "${_consumer_build}")

  _hyle_feature_smoke_run("configure locked Tracy ${shape} SDK"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_prefix}
    -DHYLE_COMPONENTS=obs
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${is_static}
    -DHYLE_WERROR=ON
    -DHYLE_TRACE=ON
    -DHYLE_TRACY_DIR=${_tracy_source}
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=OFF)
  _hyle_feature_smoke_run("build locked Tracy ${shape} SDK"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2)
  _hyle_feature_smoke_run("install locked Tracy ${shape} SDK"
    "${CMAKE_COMMAND}" --install "${_build}" --config Release)

  foreach(_required_install IN ITEMS
      "share/licenses/Hyle/tracy-LICENSE"
      "include/hyle/vendor/tracy-public/tracy/Tracy.hpp")
    if(NOT EXISTS "${_prefix}/${_required_install}")
      message(FATAL_ERROR
        "Locked Tracy ${shape} SDK omitted '${_required_install}'")
    endif()
  endforeach()

  set(_relocated_prefix
    "${HYLE_SMOKE_ROOT}/trace-package-${shape}-relocated")
  file(REMOVE_RECURSE "${_relocated_prefix}")
  file(RENAME "${_prefix}" "${_relocated_prefix}")
  set(_prefix "${_relocated_prefix}")

  file(MAKE_DIRECTORY "${_consumer_source}")
  file(WRITE "${_consumer_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_locked_trace_consumer LANGUAGES CXX)\n"
"find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS obs)\n"
"if(NOT Hyle_STATIC STREQUAL \"${is_static}\")\n"
"  message(FATAL_ERROR \"installed trace linkage mismatch: Hyle_STATIC=\${Hyle_STATIC}\")\n"
"endif()\n"
"if(NOT trace IN_LIST Hyle_AVAILABLE_FEATURES)\n"
"  message(FATAL_ERROR \"installed package omitted trace feature\")\n"
"endif()\n"
"if(NOT TARGET hyle::_tracy_client)\n"
"  message(FATAL_ERROR \"installed package omitted Tracy closure target\")\n"
"endif()\n"
"add_executable(hyle_locked_trace_consumer main.cc)\n"
"target_compile_features(hyle_locked_trace_consumer PRIVATE cxx_std_23)\n"
"target_link_libraries(hyle_locked_trace_consumer PRIVATE hyle::obs)\n"
"hyle_stage_runtime_dependencies(hyle_locked_trace_consumer)\n"
"enable_testing()\n"
"add_test(NAME hyle_locked_trace_consumer COMMAND hyle_locked_trace_consumer)\n")
  file(WRITE "${_consumer_source}/main.cc"
"#include <hyle/obs/trace.h>\n"
"#if !defined(HYLE_TRACE) || !defined(HYLE_TRACE_TRACY)\n"
"#error trace-on package did not propagate its public compile contract\n"
"#endif\n"
"int main() {\n"
"  int value = 7;\n"
"  HYLE_TRACE_ZONE();\n"
"  HYLE_TRACE_MESSAGE_LITERAL(\"installed Tracy consumer\");\n"
"  HYLE_TRACE_PLOT(\"installed.value\", value);\n"
"  HYLE_TRACE_COUNTER(\"installed.counter\", value);\n"
"  HYLE_TRACE_ALLOC(&value, sizeof(value));\n"
"  HYLE_TRACE_FREE(&value);\n"
"  return 0;\n"
"}\n")

  _hyle_feature_smoke_run("configure locked Tracy ${shape} SDK consumer"
    "${CMAKE_COMMAND}" -S "${_consumer_source}" -B "${_consumer_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_prefix})
  _hyle_feature_smoke_run("build locked Tracy ${shape} SDK consumer"
    "${CMAKE_COMMAND}" --build "${_consumer_build}" --config Release
    --parallel 2)
  _hyle_feature_smoke_run("run locked Tracy ${shape} SDK consumer"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}" -C Release
    --output-on-failure)
endfunction()

if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "adapters")
  _hyle_feature_smoke_correctness()
  _hyle_feature_smoke_shape(static ON)
  _hyle_feature_smoke_shape(shared OFF)
endif()
if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "trace")
  _hyle_feature_smoke_tracy(static ON)
  _hyle_feature_smoke_tracy(shared OFF)
endif()

message(STATUS
  "HYLE locked feature package smoke scope '${HYLE_FEATURE_SMOKE_SCOPE}' passed")
