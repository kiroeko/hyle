# Source/static/shared smoke for one product root. This is the bounded
# hand-off gate used by an individual module review; the all-product package
# smoke remains a separate whole-graph regression.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT HYLE_SMOKE_COMPONENT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=...")
  endif()
endforeach()
if(NOT HYLE_SMOKE_COMPONENT MATCHES "^[a-z][a-z0-9_]*$")
  message(FATAL_ERROR
    "HYLE_SMOKE_COMPONENT must be one canonical product name")
endif()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)
include("${HYLE_SOURCE_DIR}/cmake/HylePlatformRequirements.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleCiSharedExportHygiene.cmake")
set(_sample
  "${HYLE_SOURCE_DIR}/samples/package/${HYLE_SMOKE_COMPONENT}/cpp")
if(NOT EXISTS "${_sample}/CMakeLists.txt" OR
   NOT EXISTS "${_sample}/main.cc")
  message(FATAL_ERROR
    "product '${HYLE_SMOKE_COMPONENT}' has no installed C++ consumer under "
    "samples/package/${HYLE_SMOKE_COMPONENT}/cpp; add the real consumer "
    "journey before claiming independent delivery")
endif()

function(_hyle_selected_run label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    COMMAND_ECHO STDOUT)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} failed with exit code ${_result}")
  endif()
endfunction()

set(_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

# The source-embedding shape deliberately behaves like a real parent project.
# On macOS that parent must choose the deployment floor before add_subdirectory;
# use the same central value for all three delivery shapes so the smoke cannot
# accidentally test different loader contracts.
set(_platform_args "")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  list(APPEND _platform_args
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN})
endif()

set(_root
  "${HYLE_SMOKE_ROOT}/selected-${HYLE_SMOKE_COMPONENT}")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

# First prove the source-embedding journey. The generated parent owns its
# generator and product root and links only the requested public target.
set(_source_parent "${_root}/source-parent")
set(_source_build "${_root}/source-build")
file(MAKE_DIRECTORY "${_source_parent}")
set(_source_expert_contract "")
set(_source_extra_tests "")
set(_source_feature_contract "")
set(_selected_feature_args "")
set(_compute_onednn_selected FALSE)
set(_compute_cuda_selected FALSE)
if(HYLE_SMOKE_COMPONENT STREQUAL "simd")
  string(APPEND _source_expert_contract
    "add_library(hyle_selected_source_provider OBJECT \"${_sample}/expert_probe.cc\")\n"
    "hyle_target_simd(TARGET hyle_selected_source_provider ISA SCALAR)\n"
    "target_sources(hyle_selected_source_consumer PRIVATE $<TARGET_OBJECTS:hyle_selected_source_provider>)\n"
    "hyle_add_fat_kernel(TARGET hyle_selected_source_consumer NAME package_bias_relu IMPL \"${_sample}/fat_kernel-inl.h\" TARGETS scalar sse4 avx2 avx512 neon sve rvv)\n")
elseif(HYLE_SMOKE_COMPONENT STREQUAL "core")
  string(APPEND _source_expert_contract
    "add_executable(hyle_selected_core_no_exceptions \"${_sample}/no_exceptions.cc\")\n"
    "target_link_libraries(hyle_selected_core_no_exceptions PRIVATE hyle::core)\n"
    "if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL \"MSVC\")\n"
    "  target_compile_options(hyle_selected_core_no_exceptions PRIVATE /EHs-c-)\n"
    "else()\n"
    "  target_compile_options(hyle_selected_core_no_exceptions PRIVATE -fno-exceptions)\n"
    "endif()\n")
  string(APPEND _source_extra_tests
    "add_test(NAME hyle_selected_core_no_exceptions COMMAND hyle_selected_core_no_exceptions)\n")
elseif(HYLE_SMOKE_COMPONENT MATCHES "^(compute|graph)$")
  # Exercise one explicit capability set in every delivery shape. Preserve
  # the existing x64 oneDNN smoke unless the caller chooses a narrower route.
  set(_compute_cpu_selected TRUE)
  if(DEFINED HYLE_SMOKE_COMPUTE_CPU)
    set(_compute_cpu_selected "${HYLE_SMOKE_COMPUTE_CPU}")
  endif()
  if(DEFINED HYLE_SMOKE_COMPUTE_CUDA AND HYLE_SMOKE_COMPUTE_CUDA)
    set(_compute_cuda_selected TRUE)
  endif()
  if(DEFINED HYLE_SMOKE_COMPUTE_ONEDNN)
    set(_compute_onednn_selected "${HYLE_SMOKE_COMPUTE_ONEDNN}")
  elseif(_compute_cpu_selected AND NOT _compute_cuda_selected)
    cmake_host_system_information(
      RESULT _hyle_smoke_host_processor QUERY OS_PLATFORM)
    string(TOLOWER "${_hyle_smoke_host_processor}"
           _hyle_smoke_host_processor)
    if(_hyle_smoke_host_processor MATCHES "^(amd64|x86_64|x64)$")
      set(_compute_onednn_selected TRUE)
    endif()
    unset(_hyle_smoke_host_processor)
  endif()
  foreach(_feature IN ITEMS cpu cuda onednn)
    string(TOUPPER "${_feature}" _feature_upper)
    if(_compute_${_feature}_selected)
      set(_feature_value ON)
    else()
      set(_feature_value OFF)
    endif()
    string(APPEND _source_feature_contract
      "set(HYLE_COMPUTE_${_feature_upper} ${_feature_value})\n")
    list(APPEND _selected_feature_args
      -DHYLE_COMPUTE_${_feature_upper}=${_feature_value})
  endforeach()
endif()
file(WRITE "${_source_parent}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_selected_source_consumer LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS ${HYLE_SMOKE_COMPONENT})\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"${_source_feature_contract}"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"add_executable(hyle_selected_source_consumer \"${_sample}/main.cc\")\n"
"target_link_libraries(hyle_selected_source_consumer PRIVATE hyle::${HYLE_SMOKE_COMPONENT})\n"
"${_source_expert_contract}"
"hyle_stage_runtime_dependencies(hyle_selected_source_consumer)\n"
"enable_testing()\n"
"add_test(NAME hyle_selected_source_consumer COMMAND hyle_selected_source_consumer)\n"
"${_source_extra_tests}")
_hyle_selected_run("configure ${HYLE_SMOKE_COMPONENT} source consumer"
  "${CMAKE_COMMAND}" -S "${_source_parent}" -B "${_source_build}"
  ${_generator_args} ${_platform_args} -DCMAKE_BUILD_TYPE=Release)
_hyle_selected_run("build ${HYLE_SMOKE_COMPONENT} source consumer"
  "${CMAKE_COMMAND}" --build "${_source_build}" --config Release --parallel 2)
_hyle_selected_run("run ${HYLE_SMOKE_COMPONENT} source consumer"
  "${CMAKE_CTEST_COMMAND}" --test-dir "${_source_build}" -C Release
  --timeout 30 --output-on-failure)

foreach(_linkage IN ITEMS shared static)
  if(_linkage STREQUAL "static")
    set(_static ON)
  else()
    set(_static OFF)
  endif()
  set(_sdk_build "${_root}/${_linkage}-sdk-build")
  set(_staging "${_root}/${_linkage}-sdk-staging")
  set(_prefix "${_root}/${_linkage}-sdk-relocated")
  set(_consumer "${_root}/${_linkage}-consumer")

  _hyle_selected_run("configure ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_sdk_build}"
    ${_generator_args} ${_platform_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_staging}
    -DHYLE_COMPONENTS=${HYLE_SMOKE_COMPONENT}
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${_static}
    ${_selected_feature_args})
  _hyle_selected_run("build ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK"
    "${CMAKE_COMMAND}" --build "${_sdk_build}" --config Release --parallel 2)
  _hyle_selected_run("install ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK"
    "${CMAKE_COMMAND}" --install "${_sdk_build}" --config Release)
  if(_linkage STREQUAL "shared")
    hyle_ci_check_shared_export_hygiene("${_staging}"
      ALLOW_EMPTY
      LABEL "selected ${HYLE_SMOKE_COMPONENT} shared SDK")
  endif()
  if(HYLE_SMOKE_COMPONENT STREQUAL "simd")
    foreach(_expert_simd_header IN ITEMS
        hyle/simd/fat_kernel.h
        hyle/simd/target.h)
      if(NOT EXISTS "${_staging}/include/${_expert_simd_header}")
        message(FATAL_ERROR
          "installed ${_linkage} SIMD SDK lacks expert header "
          "${_expert_simd_header}")
      endif()
    endforeach()
    if(EXISTS "${_staging}/include/hyle/simd/provider")
      message(FATAL_ERROR
        "installed ${_linkage} SIMD SDK leaked internal provider routes")
    endif()
    foreach(_expert_asset IN ITEMS
        share/Hyle/dependencies/highway/hwy/highway.h
        share/Hyle/dependencies/xsimd/include/xsimd/xsimd.hpp
        lib/cmake/Hyle/HyleIsa.cmake
        lib/cmake/Hyle/HyleDispatch.cmake)
      if(NOT EXISTS "${_staging}/${_expert_asset}")
        message(FATAL_ERROR
          "installed ${_linkage} SIMD SDK lacks expert asset ${_expert_asset}")
      endif()
    endforeach()
  elseif(HYLE_SMOKE_COMPONENT MATCHES "^(compute|graph)$")
    file(READ "${_staging}/lib/cmake/Hyle/HyleConfig.cmake"
      _installed_hyle_config)
    file(READ "${_staging}/lib/cmake/Hyle/HyleTargets.cmake"
      _installed_hyle_targets)
    string(REGEX MATCH "set\\(Hyle_AVAILABLE_FEATURES \"([^\"]*)\"\\)"
      _feature_declaration "${_installed_hyle_config}")
    set(_installed_features "${CMAKE_MATCH_1}")
    foreach(_feature IN ITEMS cpu cuda onednn)
      if(_compute_${_feature}_selected)
        if(NOT "compute_${_feature}" IN_LIST _installed_features)
          message(FATAL_ERROR "SDK missing selected compute_${_feature} feature")
        endif()
      elseif("compute_${_feature}" IN_LIST _installed_features)
        message(FATAL_ERROR "SDK leaked unselected compute_${_feature} feature")
      endif()
    endforeach()
    foreach(_entry IN ITEMS
        "cpu|compute_backend_cpu|hyle/compute/backends/cpu.h"
        "cpu|compute_provider_native|hyle/compute/providers/native.h"
        "cuda|compute_backend_cuda|hyle/compute/backends/cuda.h"
        "onednn|compute_provider_onednn|hyle/compute/providers/onednn.h")
      string(REPLACE "|" ";" _fields "${_entry}")
      list(GET _fields 0 _feature)
      list(GET _fields 1 _factor)
      list(GET _fields 2 _header)
      if(_compute_${_feature}_selected)
        if(NOT EXISTS "${_staging}/include/${_header}" OR
           NOT _installed_hyle_targets MATCHES "_factor_${_factor}([ ;)]|$)")
          message(FATAL_ERROR "SDK lacks selected ${_factor} header/target")
        endif()
      elseif(EXISTS "${_staging}/include/${_header}" OR
             _installed_hyle_targets MATCHES "_factor_${_factor}([ ;)]|$)")
        message(FATAL_ERROR "SDK leaked absent ${_factor} header/target")
      endif()
    endforeach()
    if(NOT _compute_cpu_selected AND
       EXISTS "${_staging}/include/hyle/compute/cpu.h")
      message(FATAL_ERROR "SDK leaked unselected CPU convenience header")
    endif()
    if(NOT _compute_onednn_selected AND
       (EXISTS "${_staging}/include/hyle/graph/providers/onednn.h" OR
        _installed_hyle_targets MATCHES "_factor_graph_provider_onednn"))
      message(FATAL_ERROR "SDK leaked unselected Graph oneDNN provider")
    endif()
    if(NOT _compute_cuda_selected AND
       _installed_hyle_config MATCHES "CUDAToolkit")
      message(FATAL_ERROR "SDK leaked unselected CUDA SDK lookup")
    endif()
    if(EXISTS "${_staging}/include/hyle/compute/detail/provider.h")
      message(FATAL_ERROR
        "installed ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK leaked its private provider SPI")
    endif()
    if(_compute_onednn_selected)
      if(NOT EXISTS "${_staging}/share/licenses/Hyle/oneDNN-LICENSE")
        message(FATAL_ERROR
          "installed ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK lacks oneDNN license payload")
      endif()
      if(_static AND
         (NOT EXISTS "${_staging}/include/oneapi/dnnl/dnnl.h" OR
          NOT EXISTS "${_staging}/include/oneapi/dnnl/dnnl_ukernel.h"))
        message(FATAL_ERROR
          "installed static ${HYLE_SMOKE_COMPONENT} SDK lacks oneDNN ukernel development closure")
      endif()
    endif()
    if((NOT _compute_onednn_selected OR NOT _static) AND
       EXISTS "${_staging}/include/oneapi")
      message(FATAL_ERROR
        "installed ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK leaked private oneDNN headers")
    endif()
    if(_compute_cuda_selected)
      if(NOT EXISTS "${_staging}/include/hyle/compute/backends/cuda.h")
        message(FATAL_ERROR
          "installed ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK lacks its CUDA façade")
      endif()
      if(NOT _installed_hyle_config MATCHES
         "CUDAToolkit\\|13\\.3")
        message(FATAL_ERROR
          "installed ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK does not propagate CUDAToolkit 13.3")
      endif()
    endif()
  endif()
  file(RENAME "${_staging}" "${_prefix}" RESULT _relocate_result)
  if(NOT _relocate_result STREQUAL "0")
    message(FATAL_ERROR
      "could not relocate ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK: "
      "${_relocate_result}")
  endif()

  _hyle_selected_run("configure ${_linkage} ${HYLE_SMOKE_COMPONENT} consumer"
    "${CMAKE_COMMAND}" -S "${_sample}" -B "${_consumer}"
    ${_generator_args} ${_platform_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_prefix})
  _hyle_selected_run("build ${_linkage} ${HYLE_SMOKE_COMPONENT} consumer"
    "${CMAKE_COMMAND}" --build "${_consumer}" --config Release --parallel 2)
  _hyle_selected_run("run ${_linkage} ${HYLE_SMOKE_COMPONENT} consumer"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer}" -C Release
    --timeout 30 --output-on-failure)
endforeach()

message(STATUS
  "HYLE selected product smoke passed: ${HYLE_SMOKE_COMPONENT} "
  "(source, shared, static, relocated consumers)")
