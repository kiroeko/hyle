# Linux correctness + package/install/downstream smoke for locked optional
# product sources.  Unlike the cross-platform CPU package smoke, this enables
# the private ggml/nghttp2/Vulkan closure once per CI revision and exercises its
# real module adapters before producing static/shared SDKs.  A separate small
# obs-only shape compiles and consumes the exact locked Tracy trace-on surface.
# Official archives are cached by locked commit, verified before extraction,
# then their no-.git expanded trees are verified again before use.  The default
# scope is `all`; local diagnosis may select `adapters` or `trace` without
# weakening the hosted all-scope gate.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT HYLE_ARCHIVE_CACHE)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR
      "HyleCiFeaturePackageSmoke requires -D${_required}=...")
  endif()
endforeach()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)
get_filename_component(HYLE_ARCHIVE_CACHE "${HYLE_ARCHIVE_CACHE}" ABSOLUTE)
file(MAKE_DIRECTORY "${HYLE_SMOKE_ROOT}" "${HYLE_ARCHIVE_CACHE}")

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

function(_hyle_feature_smoke_run label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    COMMAND_ECHO STDOUT)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} failed with exit code ${_result}")
  endif()
endfunction()

function(_hyle_feature_prepare_source dependency out_var)
  if(NOT dependency IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES AND
     NOT dependency IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
    message(FATAL_ERROR
      "Feature package smoke names dependency '${dependency}' whose source "
      "class cannot be prepared from the locked archive")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  set(_archive
    "${HYLE_ARCHIVE_CACHE}/${dependency}-${${_prefix}_COMMIT}.tar.gz")

  if(EXISTS "${_archive}")
    file(SHA256 "${_archive}" _archive_sha256)
    if(NOT _archive_sha256 STREQUAL "${${_prefix}_ARCHIVE_SHA256}")
      message(FATAL_ERROR
        "Cached archive '${_archive}' does not match the dependency lock.\n"
        "  expected: ${${_prefix}_ARCHIVE_SHA256}\n"
        "  actual:   ${_archive_sha256}\n"
        "Remove only this cache entry and rerun the smoke.")
    endif()
  else()
    set(_partial "${_archive}.partial")
    file(REMOVE "${_partial}")
    file(DOWNLOAD "${${_prefix}_ARCHIVE_URL}" "${_partial}"
      EXPECTED_HASH "SHA256=${${_prefix}_ARCHIVE_SHA256}"
      TLS_VERIFY ON
      STATUS _download_status
      LOG _download_log)
    list(GET _download_status 0 _download_code)
    if(NOT _download_code EQUAL 0)
      list(GET _download_status 1 _download_message)
      file(REMOVE "${_partial}")
      message(FATAL_ERROR
        "Cannot download locked '${dependency}' archive: "
        "${_download_message}\n${_download_log}")
    endif()
    file(RENAME "${_partial}" "${_archive}")
  endif()

  set(_source_root "${HYLE_SMOKE_ROOT}/locked-sources/${dependency}")
  set(_extract_root
    "${HYLE_SMOKE_ROOT}/locked-sources/${dependency}.extracting")
  file(REMOVE_RECURSE "${_source_root}" "${_extract_root}")
  file(MAKE_DIRECTORY "${_extract_root}")
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_root}")
  file(GLOB _archive_roots LIST_DIRECTORIES TRUE
    RELATIVE "${_extract_root}" "${_extract_root}/*")
  list(LENGTH _archive_roots _archive_root_count)
  if(NOT _archive_root_count EQUAL 1)
    message(FATAL_ERROR
      "Locked '${dependency}' archive must contain exactly one root directory; "
      "found: ${_archive_roots}")
  endif()
  list(GET _archive_roots 0 _archive_root)
  if(NOT IS_DIRECTORY "${_extract_root}/${_archive_root}" OR
     IS_SYMLINK "${_extract_root}/${_archive_root}")
    message(FATAL_ERROR
      "Locked '${dependency}' archive root is not a real directory: "
      "${_archive_root}")
  endif()
  file(RENAME "${_extract_root}/${_archive_root}" "${_source_root}")
  file(REMOVE_RECURSE "${_extract_root}")
  hyle_verify_locked_expanded_source("${dependency}" "${_source_root}")
  set(${out_var} "${_source_root}" PARENT_SCOPE)
endfunction()

set(_hyle_feature_smoke_dependencies "")
if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "adapters")
  list(APPEND _hyle_feature_smoke_dependencies
    ggml nghttp2 vulkan_headers volk glslang googletest)
endif()
if(HYLE_FEATURE_SMOKE_SCOPE STREQUAL "all" OR
   HYLE_FEATURE_SMOKE_SCOPE STREQUAL "trace")
  list(APPEND _hyle_feature_smoke_dependencies tracy)
endif()
foreach(_dependency IN LISTS _hyle_feature_smoke_dependencies)
  _hyle_feature_prepare_source("${_dependency}" "_${_dependency}_source")
endforeach()

set(_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND
   NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

# Package-only probes prove closure/export mechanics, but they cannot detect an
# adapter that still compiles while misusing the upgraded library.  Build and
# run exactly the three real feature-owned tests once; do not replicate the
# complete heavy test tree for both static and shared package shapes.
function(_hyle_feature_smoke_correctness)
  set(_build "${HYLE_SMOKE_ROOT}/feature-correctness-build")
  file(REMOVE_RECURSE "${_build}")

  _hyle_feature_smoke_run("configure locked feature adapter tests"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_build}"
    ${_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DHYLE_COMPONENTS=io,compute
    -DHYLE_BUILD_TESTS=ON
    -DHYLE_TEST_INSTRUMENTATION=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=OFF
    -DHYLE_STATIC=OFF
    -DHYLE_WERROR=ON
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=ON
    -DHYLE_COMPUTE_CUDA=OFF
    -DHYLE_COMPUTE_GGML=ON
    -DHYLE_COMPUTE_GGML_NATIVE=OFF
    -DHYLE_COMPUTE_VULKAN=ON
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    -DFETCHCONTENT_SOURCE_DIR_GGML=${_ggml_source}
    -DFETCHCONTENT_SOURCE_DIR_NGHTTP2=${_nghttp2_source}
    -DFETCHCONTENT_SOURCE_DIR_VULKAN_HEADERS=${_vulkan_headers_source}
    -DFETCHCONTENT_SOURCE_DIR_VOLK=${_volk_source}
    -DFETCHCONTENT_SOURCE_DIR_GLSLANG=${_glslang_source}
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${_googletest_source})
  _hyle_feature_smoke_run("build locked feature adapter tests"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2
    --target hyle_io_http2_test hyle_compute_ggml_bridge_test
             hyle_compute_vulkan_test)
  _hyle_feature_smoke_run("run locked feature adapter tests"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_build}" -C Release
    -R "^(hyle_io_http2_test|hyle_compute_ggml_bridge_test|hyle_compute_vulkan_test)$"
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
    -DHYLE_COMPONENTS=io,compute
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${is_static}
    -DHYLE_WERROR=ON
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=ON
    -DHYLE_COMPUTE_CUDA=OFF
    -DHYLE_COMPUTE_GGML=ON
    -DHYLE_COMPUTE_GGML_NATIVE=OFF
    -DHYLE_COMPUTE_VULKAN=ON
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    -DFETCHCONTENT_SOURCE_DIR_GGML=${_ggml_source}
    -DFETCHCONTENT_SOURCE_DIR_NGHTTP2=${_nghttp2_source}
    -DFETCHCONTENT_SOURCE_DIR_VULKAN_HEADERS=${_vulkan_headers_source}
    -DFETCHCONTENT_SOURCE_DIR_VOLK=${_volk_source})

  if(IS_DIRECTORY "${_build}/_deps/glslang-src")
    message(FATAL_ERROR
      "Feature package smoke fetched test-only glslang in a production SDK build")
  endif()

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
    "bin/hyle_shader.py"
    "share/licenses/Hyle/ggml-LICENSE"
    "share/licenses/Hyle/nghttp2-LICENSE"
    "share/licenses/Hyle/volk-LICENSE.md"
    "share/licenses/Hyle/Vulkan-Headers/LICENSE.md"
    "share/licenses/Hyle/Vulkan-Headers/Apache-2.0.txt"
    "share/licenses/Hyle/Vulkan-Headers/MIT.txt")
  set(_static_development_installs
    "include/ggml.h"
    "include/nghttp2/nghttp2.h"
    "include/nghttp2/nghttp2ver.h"
    "include/hyle/vendor/volk/volk.h"
    "include/hyle/vendor/volk/vulkan/vulkan.h")
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
  file(GLOB_RECURSE _installed_entries LIST_DIRECTORIES TRUE
    RELATIVE "${_prefix}" "${_prefix}/*")
  foreach(_installed_entry IN LISTS _installed_entries)
    string(TOLOWER "${_installed_entry}" _installed_entry_lower)
    if(_installed_entry_lower MATCHES "glslang")
      message(FATAL_ERROR
        "Test-only glslang leaked into the production SDK: ${_installed_entry}")
    endif()
  endforeach()

  file(MAKE_DIRECTORY "${_consumer_source}")
  file(WRITE "${_consumer_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_locked_feature_consumer LANGUAGES C CXX)\n"
"find_package(Hyle 0.1 REQUIRED COMPONENTS io compute)\n"
"if(NOT Hyle_STATIC STREQUAL \"${is_static}\")\n"
"  message(FATAL_ERROR \"installed linkage mismatch: Hyle_STATIC=\${Hyle_STATIC}\")\n"
"endif()\n"
"foreach(_feature IN ITEMS io_http2 compute_ggml compute_vulkan)\n"
"  if(NOT _feature IN_LIST Hyle_AVAILABLE_FEATURES)\n"
"    message(FATAL_ERROR \"missing installed HYLE feature: \${_feature}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_target IN ITEMS hyle::io hyle::compute)\n"
"  if(NOT TARGET \${_target})\n"
"    message(FATAL_ERROR \"missing installed HYLE target: \${_target}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_target IN ITEMS hyle::_nghttp2_static hyle::_ggml hyle::_ggml_base hyle::_ggml_cpu hyle::_volk)\n"
"  if(${is_static} AND NOT TARGET \${_target})\n"
"    message(FATAL_ERROR \"static SDK omitted private closure target: \${_target}\")\n"
"  elseif(NOT ${is_static} AND TARGET \${_target})\n"
"    message(FATAL_ERROR \"shared SDK exported private closure target: \${_target}\")\n"
"  endif()\n"
"endforeach()\n"
"add_executable(hyle_locked_feature_consumer main.cc)\n"
"target_link_libraries(hyle_locked_feature_consumer PRIVATE hyle::io hyle::compute)\n"
"hyle_stage_runtime_dependencies(hyle_locked_feature_consumer)\n"
"enable_testing()\n"
"add_test(NAME hyle_locked_feature_consumer COMMAND hyle_locked_feature_consumer)\n")
  file(WRITE "${_consumer_source}/main.cc"
"#include <hyle/c/io.h>\n"
"#include <hyle/compute/device.h>\n"
"#include <hyle/compute/ggml_bridge.h>\n"
"#include <cstddef>\n"
"#include <cstring>\n"
"int main() {\n"
"  if (hyle_io_http2_available() != 1) return 1;\n"
"  hyle_io_http2_session* session = hyle_io_http2_session_create(\n"
"      hyle_allocator{}, HYLE_IO_HTTP2_CLIENT);\n"
"  if (session == nullptr || !hyle_io_http2_session_valid(session)) return 2;\n"
"  const void* settings = nullptr;\n"
"  const std::size_t settings_size = hyle_io_http2_next_output(session, &settings);\n"
"  const bool settings_ok = settings_size != static_cast<std::size_t>(-1) &&\n"
"                           settings_size != 0 && settings != nullptr;\n"
"  hyle_io_http2_session_destroy(session);\n"
"  if (!settings_ok) return 3;\n"
"  hyle::compute::Device devices[32];\n"
"  const auto count = hyle::compute::enumerate(devices, 32);\n"
"  if (count == 0 || hyle::compute::enumerate(nullptr, 7) != count) return 4;\n"
"  const hyle::compute::Device* host = nullptr;\n"
"  for (std::size_t i = 0; i < count && i < 32; ++i) {\n"
"    if (std::strcmp(devices[i].backend(), \"host\") == 0) host = &devices[i];\n"
"  }\n"
"  if (host == nullptr) return 5;\n"
"  auto backend = hyle::compute::GgmlBackend::create(*host, 1);\n"
"  if (!backend.has_value() || !backend->valid()) return 6;\n"
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
    -DHYLE_TRACE_BACKEND=tracy
    -DHYLE_TRACY_DIR=${_tracy_source}
    -DHYLE_IO_TLS=OFF
    -DHYLE_IO_HTTP2=OFF
    -DHYLE_COMPUTE_CUDA=OFF
    -DHYLE_COMPUTE_GGML=OFF
    -DHYLE_COMPUTE_VULKAN=OFF)
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
"find_package(Hyle 0.1 REQUIRED COMPONENTS obs)\n"
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
"#include <hyle/c/obs.h>\n"
"#include <hyle/obs/trace.h>\n"
"#if !defined(HYLE_TRACE) || !defined(HYLE_TRACE_TRACY)\n"
"#error trace-on package did not propagate its public compile contract\n"
"#endif\n"
"int main() {\n"
"  int value = 7;\n"
"  HYLE_TRACE_ZONE();\n"
"  HYLE_TRACE_MESSAGE_LITERAL(\"installed Tracy consumer\");\n"
"  HYLE_TRACE_PLOT(\"installed.value\", value);\n"
"  HYLE_TRACE_ALLOC(&value, sizeof(value));\n"
"  HYLE_TRACE_FREE(&value);\n"
"  hyle_obs_trace_message(\"installed FFI-candidate trace fixture\");\n"
"  hyle_obs_trace_counter(\"installed.counter\", value);\n"
"  return hyle_obs_trace_enabled() == 1 ? 0 : 1;\n"
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
