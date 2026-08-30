# Source/static/shared smoke for one component root. This is the bounded
# hand-off gate used by an individual module review; the all-component package
# smoke remains a separate whole-graph regression.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT HYLE_SMOKE_COMPONENT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=...")
  endif()
endforeach()
if(NOT HYLE_SMOKE_COMPONENT MATCHES "^[a-z][a-z0-9_]*$")
  message(FATAL_ERROR
    "HYLE_SMOKE_COMPONENT must be one canonical component name")
endif()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)
include("${HYLE_SOURCE_DIR}/cmake/HylePlatformRequirements.cmake")
set(_sample
  "${HYLE_SOURCE_DIR}/samples/package/${HYLE_SMOKE_COMPONENT}/cpp")
if(NOT EXISTS "${_sample}/CMakeLists.txt" OR
   NOT EXISTS "${_sample}/main.cc")
  message(FATAL_ERROR
    "component '${HYLE_SMOKE_COMPONENT}' has no installed C++ consumer under "
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
file(WRITE "${_source_parent}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_selected_source_consumer LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS ${HYLE_SMOKE_COMPONENT})\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"add_executable(hyle_selected_source_consumer \"${_sample}/main.cc\")\n"
"target_link_libraries(hyle_selected_source_consumer PRIVATE hyle::${HYLE_SMOKE_COMPONENT})\n"
"hyle_stage_runtime_dependencies(hyle_selected_source_consumer)\n"
"enable_testing()\n"
"add_test(NAME hyle_selected_source_consumer COMMAND hyle_selected_source_consumer)\n")
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
    -DHYLE_STATIC=${_static})
  _hyle_selected_run("build ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK"
    "${CMAKE_COMMAND}" --build "${_sdk_build}" --config Release --parallel 2)
  _hyle_selected_run("install ${_linkage} ${HYLE_SMOKE_COMPONENT} SDK"
    "${CMAKE_COMMAND}" --install "${_sdk_build}" --config Release)
  if(HYLE_SMOKE_COMPONENT STREQUAL "simd")
    foreach(_source_only_simd_header IN ITEMS
        hyle/simd/fat_kernel.h
        hyle/simd/target.h)
      if(EXISTS "${_staging}/include/${_source_only_simd_header}")
        message(FATAL_ERROR
          "installed ${_linkage} SIMD SDK leaked source-only header "
          "${_source_only_simd_header}")
      endif()
    endforeach()
    foreach(_source_only_simd_directory IN ITEMS
        hyle/simd/provider)
      if(EXISTS "${_staging}/include/${_source_only_simd_directory}")
        message(FATAL_ERROR
          "installed ${_linkage} SIMD SDK leaked source-only directory "
          "${_source_only_simd_directory}")
      endif()
    endforeach()
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
  "HYLE selected component smoke passed: ${HYLE_SMOKE_COMPONENT} "
  "(source, shared, static, relocated consumers)")
