# Configure-only checks for the component-selection and dependency-graph
# contracts.  These deliberately use fresh child projects so cache shadowing
# and source-embedding behavior are exercised exactly as consumers see them.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()

set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

set(_hyle_contract_root "${HYLE_SMOKE_ROOT}/component-contract")
file(REMOVE_RECURSE "${_hyle_contract_root}")
file(MAKE_DIRECTORY "${_hyle_contract_root}")

function(_hyle_contract_run label expect_success)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${_result}:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded")
  endif()
  set(HYLE_CONTRACT_LAST_OUTPUT "${_output}" PARENT_SCOPE)
endfunction()

# An explicitly empty -D value is almost always a lost shell/CI variable.  The
# ordinary unspecified default remains `all`, but explicit empty must fail.
_hyle_contract_run("explicit-empty component selection" FALSE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_hyle_contract_root}/explicit-empty"
  ${_hyle_generator_args}
  -DHYLE_COMPONENTS:STRING=
  -DHYLE_BUILD_TESTS=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_IO_TLS=OFF
  -DHYLE_COMPUTE_CUDA=OFF)
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "HYLE_COMPONENTS is explicitly empty")
  message(FATAL_ERROR
    "explicit-empty selection failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# Core is a self-contained, header-only product component. Selecting and
# installing it alone must never manufacture a version-only runtime artifact.
set(_core_source "${_hyle_contract_root}/core-source")
file(MAKE_DIRECTORY "${_core_source}")
file(WRITE "${_core_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_core_component LANGUAGES C CXX)\n"
"set(HYLE_COMPONENTS core)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"if(NOT TARGET hyle::core)\n"
"  message(FATAL_ERROR \"missing hyle::core\")\n"
"endif()\n"
"get_target_property(_type hyle_core TYPE)\n"
"if(NOT _type STREQUAL \"INTERFACE_LIBRARY\")\n"
"  message(FATAL_ERROR \"core is not header-only: \${_type}\")\n"
"endif()\n"
"foreach(_forbidden IN ITEMS container_intrusive util_core error obs math platform time mem simd container util sched io compute graph)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("standalone core closure" TRUE
  "${CMAKE_COMMAND}" -S "${_core_source}"
  -B "${_hyle_contract_root}/core-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

foreach(_core_linkage IN ITEMS shared static)
  if(_core_linkage STREQUAL "static")
    set(_core_static ON)
  else()
    set(_core_static OFF)
  endif()
  set(_core_sdk_build
    "${_hyle_contract_root}/core-${_core_linkage}-sdk-build")
  set(_core_sdk_prefix
    "${_hyle_contract_root}/core-${_core_linkage}-sdk-prefix")
  _hyle_contract_run("configure isolated ${_core_linkage} Core SDK" TRUE
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
    -B "${_core_sdk_build}"
    ${_hyle_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_core_sdk_prefix}
    -DHYLE_COMPONENTS=core
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${_core_static}
    -DHYLE_IO_TLS=OFF
    -DHYLE_COMPUTE_CUDA=OFF)
  _hyle_contract_run("install isolated ${_core_linkage} Core SDK" TRUE
    "${CMAKE_COMMAND}" --install "${_core_sdk_build}" --config Release)
  file(GLOB_RECURSE _core_binary_artifacts LIST_DIRECTORIES FALSE
    "${_core_sdk_prefix}/*.a"
    "${_core_sdk_prefix}/*.dll"
    "${_core_sdk_prefix}/*.dylib"
    "${_core_sdk_prefix}/*.lib"
    "${_core_sdk_prefix}/*.so"
    "${_core_sdk_prefix}/*.so.*")
  if(_core_binary_artifacts)
    message(FATAL_ERROR
      "header-only ${_core_linkage} Core SDK installed binary artifacts: "
      "${_core_binary_artifacts}")
  endif()
  set(_core_consumer_build
    "${_hyle_contract_root}/core-${_core_linkage}-consumer-build")
  _hyle_contract_run("configure isolated ${_core_linkage} Core consumer" TRUE
    "${CMAKE_COMMAND}"
    -S "${HYLE_SOURCE_DIR}/samples/package/core/cpp"
    -B "${_core_consumer_build}"
    ${_hyle_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_core_sdk_prefix})
  _hyle_contract_run("build isolated ${_core_linkage} Core consumer" TRUE
    "${CMAKE_COMMAND}" --build "${_core_consumer_build}"
    --config Release --parallel 2)
  _hyle_contract_run("run isolated ${_core_linkage} Core consumer" TRUE
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_core_consumer_build}"
    -C Release --output-on-failure)
endforeach()
unset(_core_binary_artifacts)
unset(_core_consumer_build)
unset(_core_linkage)
unset(_core_sdk_build)
unset(_core_sdk_prefix)
unset(_core_static)

# A caller that wants only the allocation-free intrusive container gets Core's
# header-only source contracts as its sole dependency and no compiled HYLE
# runtime closure. This prevents an
# installed-all-headers layout from hiding a missing or accidentally widened
# component edge.
set(_intrusive_source "${_hyle_contract_root}/intrusive-source")
file(MAKE_DIRECTORY "${_intrusive_source}")
file(WRITE "${_intrusive_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_intrusive_component LANGUAGES C CXX)\n"
"set(HYLE_COMPONENTS container_intrusive)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"if(NOT TARGET hyle::container_intrusive)\n"
"  message(FATAL_ERROR \"missing hyle::container_intrusive\")\n"
"endif()\n"
"foreach(_required IN ITEMS core container_intrusive)\n"
"  if(NOT TARGET hyle::\${_required})\n"
"    message(FATAL_ERROR \"missing intrusive-closure target hyle::\${_required}\")\n"
"  endif()\n"
"  get_target_property(_type hyle_\${_required} TYPE)\n"
"  if(NOT _type STREQUAL \"INTERFACE_LIBRARY\")\n"
"    message(FATAL_ERROR \"\${_required} is not header-only: \${_type}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS error time mem platform container)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("standalone container_intrusive closure" TRUE
  "${CMAKE_COMMAND}" -S "${_intrusive_source}"
  -B "${_hyle_contract_root}/intrusive-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# The target is also an independently deliverable installed component.  Build
# an isolated SDK rather than testing against an all-component prefix: the
# latter installs every public header and can hide a missing export or an
# accidental binary dependency.
set(_intrusive_sdk_build "${_hyle_contract_root}/intrusive-sdk-build")
set(_intrusive_sdk_prefix "${_hyle_contract_root}/intrusive-sdk-prefix")
_hyle_contract_run("configure isolated container_intrusive SDK" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_intrusive_sdk_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=${_intrusive_sdk_prefix}
  -DHYLE_COMPONENTS=container_intrusive
  -DHYLE_BUILD_TESTS=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_INSTALL=ON
  -DHYLE_IO_TLS=OFF
  -DHYLE_COMPUTE_CUDA=OFF)
_hyle_contract_run("install isolated container_intrusive SDK" TRUE
  "${CMAKE_COMMAND}" --install "${_intrusive_sdk_build}" --config Release)
file(GLOB_RECURSE _intrusive_binary_artifacts LIST_DIRECTORIES FALSE
  "${_intrusive_sdk_prefix}/*.a"
  "${_intrusive_sdk_prefix}/*.dll"
  "${_intrusive_sdk_prefix}/*.dylib"
  "${_intrusive_sdk_prefix}/*.lib"
  "${_intrusive_sdk_prefix}/*.so"
  "${_intrusive_sdk_prefix}/*.so.*")
if(_intrusive_binary_artifacts)
  message(FATAL_ERROR
    "header-only container_intrusive SDK installed binary artifacts: "
    "${_intrusive_binary_artifacts}")
endif()
set(_intrusive_consumer_build
    "${_hyle_contract_root}/intrusive-consumer-build")
_hyle_contract_run("configure isolated container_intrusive consumer" TRUE
  "${CMAKE_COMMAND}"
  -S "${HYLE_SOURCE_DIR}/samples/package/container_intrusive/cpp"
  -B "${_intrusive_consumer_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH=${_intrusive_sdk_prefix})
_hyle_contract_run("build isolated container_intrusive consumer" TRUE
  "${CMAKE_COMMAND}" --build "${_intrusive_consumer_build}"
  --config Release --parallel 2)
_hyle_contract_run("run isolated container_intrusive consumer" TRUE
  "${CMAKE_CTEST_COMMAND}" --test-dir "${_intrusive_consumer_build}"
  -C Release --output-on-failure)
unset(_intrusive_binary_artifacts)
unset(_intrusive_consumer_build)
unset(_intrusive_sdk_build)
unset(_intrusive_sdk_prefix)

# time publicly exposes TimerList, so selecting time must carry the lightweight
# intrusive target but must not pull the compiled container/mem/platform stack.
set(_time_source "${_hyle_contract_root}/time-source")
file(MAKE_DIRECTORY "${_time_source}")
file(WRITE "${_time_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_time_component LANGUAGES C CXX)\n"
"set(HYLE_COMPONENTS time)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"foreach(_required IN ITEMS core error container_intrusive time)\n"
"  if(NOT TARGET hyle::\${_required})\n"
"    message(FATAL_ERROR \"missing time-closure target hyle::\${_required}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS mem platform container)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected time-closure target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("time uses lightweight intrusive closure" TRUE
  "${CMAKE_COMMAND}" -S "${_time_source}"
  -B "${_hyle_contract_root}/time-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Reproduce the normal-variable form used by add_subdirectory consumers.  The
# generated target graph and the cache/GUI manifest must agree.
set(_embed_source "${_hyle_contract_root}/embed-source")
file(MAKE_DIRECTORY "${_embed_source}")
file(WRITE "${_embed_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_component_embed LANGUAGES C CXX)\n"
"if(APPLE)\n"
"  set(CMAKE_OSX_DEPLOYMENT_TARGET 14.4 CACHE STRING \"\" FORCE)\n"
"endif()\n"
"set(HYLE_COMPONENTS container)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_COMPUTE_CUDA OFF)\n"
"set(HYLE_DEPS_DIR \"\${CMAKE_BINARY_DIR}/hyle-dependencies\")\n"
"set(FETCHCONTENT_BASE_DIR \"\${CMAKE_BINARY_DIR}/parent-dependencies\")\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"if(NOT FETCHCONTENT_BASE_DIR STREQUAL \"\${CMAKE_BINARY_DIR}/parent-dependencies\")\n"
"  message(FATAL_ERROR \"HYLE rewrote its embedding parent's FetchContent cache\")\n"
"endif()\n"
"get_property(_cached CACHE HYLE_COMPONENTS PROPERTY VALUE)\n"
"if(NOT _cached STREQUAL \"container\")\n"
"  message(FATAL_ERROR \"cache says '\${_cached}', expected container\")\n"
"endif()\n"
"foreach(_required IN ITEMS core error platform mem container_intrusive container)\n"
"  if(NOT TARGET hyle::\${_required})\n"
"    message(FATAL_ERROR \"missing selected-closure target hyle::\${_required}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS io compute graph simd)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("source-embedded component/cache agreement" TRUE
  "${CMAKE_COMMAND}" -S "${_embed_source}"
  -B "${_hyle_contract_root}/embed-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Prove that the end-of-configuration audit catches an internal link added
# after the declaration-time helper has returned.
set(_bypass_source "${_hyle_contract_root}/bypass-source")
file(MAKE_DIRECTORY "${_bypass_source}")
file(WRITE "${_bypass_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_dependency_bypass LANGUAGES NONE)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core error)\n"
"set(_HYLE_DEPS_CORE \"\")\n"
"set(_HYLE_DEPS_ERROR \"\")\n"
"include(HyleModule)\n"
"add_library(hyle_core INTERFACE)\n"
"add_library(hyle_error INTERFACE)\n"
"target_link_libraries(hyle_error INTERFACE hyle_core)\n"
"hyle_validate_final_module_graph()\n")
_hyle_contract_run("late internal dependency bypass" FALSE
  "${CMAKE_COMMAND}" -S "${_bypass_source}"
  -B "${_hyle_contract_root}/bypass-build"
  ${_hyle_generator_args})
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "direct internal dependencies disagree")
  message(FATAL_ERROR
    "late dependency bypass failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# A late PUBLIC edge can also silently widen a package usage contract even when
# the dependency set itself is unchanged.  Prove that visibility is audited,
# not only graph membership.
set(_visibility_source "${_hyle_contract_root}/visibility-source")
file(MAKE_DIRECTORY "${_visibility_source}")
file(WRITE "${_visibility_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_dependency_visibility LANGUAGES NONE)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core error)\n"
"set(_HYLE_DEPS_CORE \"\")\n"
"set(_HYLE_PRIVATE_DEPS_CORE \"\")\n"
"set(_HYLE_PUBLIC_DEPS_CORE \"\")\n"
"set(_HYLE_DEPS_ERROR core)\n"
"set(_HYLE_PRIVATE_DEPS_ERROR core)\n"
"set(_HYLE_PUBLIC_DEPS_ERROR \"\")\n"
"include(HyleModule)\n"
"add_library(hyle_core INTERFACE)\n"
"add_library(hyle_error INTERFACE)\n"
"target_link_libraries(hyle_error INTERFACE hyle_core)\n"
"hyle_validate_final_module_graph()\n")
_hyle_contract_run("late internal dependency visibility widening" FALSE
  "${CMAKE_COMMAND}" -S "${_visibility_source}"
  -B "${_hyle_contract_root}/visibility-build"
  ${_hyle_generator_args})
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "public internal dependencies disagree")
  message(FATAL_ERROR
    "late dependency visibility widening failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# FetchContent documents that the first declaration of a dependency name wins
# across the complete project hierarchy.  A source-embedding parent must not be
# able to predeclare a same-named arbitrary source and silently make HYLE's
# locked FetchContent_Declare() a no-op.  The parent is executable/trusted CMake,
# but the resulting HYLE product must still fail identity verification.
set(_first_declare_source
  "${_hyle_contract_root}/fetchcontent-first-declare-source")
set(_first_declare_parent
  "${_hyle_contract_root}/fetchcontent-first-declare-parent")
file(MAKE_DIRECTORY "${_first_declare_source}" "${_first_declare_parent}")
file(WRITE "${_first_declare_source}/LICENSE"
  "not the locked GoogleTest license\n")
file(WRITE "${_first_declare_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(not_locked_googletest LANGUAGES NONE)\n")
file(WRITE "${_first_declare_parent}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_fetchcontent_first_declare LANGUAGES C CXX)\n"
"include(FetchContent)\n"
"FetchContent_Declare(googletest SOURCE_DIR \"${_first_declare_source}\")\n"
"set(HYLE_COMPONENTS core)\n"
"set(HYLE_BUILD_TESTS ON)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_IO_HTTP2 OFF)\n"
"set(HYLE_COMPUTE_CUDA OFF)\n"
"set(HYLE_COMPUTE_GGML OFF)\n"
"set(HYLE_COMPUTE_VULKAN OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n")
_hyle_contract_run("FetchContent first-declaration lock bypass" FALSE
  "${CMAKE_COMMAND}" -S "${_first_declare_parent}"
  -B "${_hyle_contract_root}/fetchcontent-first-declare-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "Locked dependency 'googletest'.*(license|source tree)")
  message(FATAL_ERROR
    "FetchContent first-declaration bypass failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# Lock projection must not require an unused in-tree snapshot.  A selected
# dependency and the complete-audit script entry point must still fail closed.
set(_vendor_fixture "${_hyle_contract_root}/vendor-closure-fixture")
file(MAKE_DIRECTORY "${_vendor_fixture}/cmake")
configure_file("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake"
  "${_vendor_fixture}/cmake/HyleDependencyVersions.cmake" COPYONLY)
file(WRITE "${_vendor_fixture}/cmake/HyleDependencyLock.json"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [{
    \"name\": \"demo\", \"class\": \"in_tree\",
    \"repository\": \"https://github.com/example/demo.git\",
    \"version\": \"fixture\",
    \"revision\": \"1111111111111111111111111111111111111111\",
    \"git_tree\": \"2222222222222222222222222222222222222222\",
    \"archive\": {
      \"url\": \"https://codeload.github.com/example/demo/tar.gz/1111111111111111111111111111111111111111\",
      \"sha256\": \"3333333333333333333333333333333333333333333333333333333333333333\"
    },
    \"vendored\": {
      \"path\": \"third_party/demo\",
      \"tree_sha256\": \"4444444444444444444444444444444444444444444444444444444444444444\"
    },
    \"license\": { \"expression\": \"MIT\", \"files\": [{
      \"root\": \"hyle\", \"path\": \"third_party/demo/LICENSE\",
      \"sha256\": \"5555555555555555555555555555555555555555555555555555555555555555\"
    }] }
  }]
}
")
set(_vendor_unused "${_vendor_fixture}/unused.cmake")
file(WRITE "${_vendor_unused}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_in_tree_dependencies()\n")
_hyle_contract_run("unused in-tree dependency is not required" TRUE
  "${CMAKE_COMMAND}" -P "${_vendor_unused}")

set(_vendor_selected "${_vendor_fixture}/selected.cmake")
file(WRITE "${_vendor_selected}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_in_tree_dependencies(demo)\n")
_hyle_contract_run("selected in-tree dependency is required" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_selected}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "required by the selected closure is missing")
  message(FATAL_ERROR
    "selected vendor failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

_hyle_contract_run("complete in-tree audit requires every snapshot" FALSE
  "${CMAKE_COMMAND}" -P
  "${_vendor_fixture}/cmake/HyleDependencyVersions.cmake")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "required by the selected closure is missing")
  message(FATAL_ERROR
    "complete vendor audit failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# License roots are class semantics, not an optional verifier hint.  A bad
# root must fail during projection even when that dependency is not selected;
# otherwise one audit path could silently skip the license entirely.
file(READ "${_vendor_fixture}/cmake/HyleDependencyLock.json"
  _vendor_lock_json)
string(REPLACE "\"root\": \"hyle\"" "\"root\": \"dependency\""
  _vendor_bad_root_json "${_vendor_lock_json}")
set(_vendor_bad_root_lock "${_vendor_fixture}/bad-license-root.json")
file(WRITE "${_vendor_bad_root_lock}" "${_vendor_bad_root_json}")
set(_vendor_bad_root_script "${_vendor_fixture}/bad-license-root.cmake")
file(WRITE "${_vendor_bad_root_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_vendor_bad_root_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_vendor_bad_root_lock}\")\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("class-specific license root" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_bad_root_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "class 'in_tree' requires license root 'hyle'")
  message(FATAL_ERROR
    "bad license root failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

string(REPLACE
  "third_party/demo/LICENSE" "third_party/other/LICENSE"
  _vendor_bad_owner_json "${_vendor_lock_json}")
set(_vendor_bad_owner_lock "${_vendor_fixture}/bad-license-owner.json")
file(WRITE "${_vendor_bad_owner_lock}" "${_vendor_bad_owner_json}")
set(_vendor_bad_owner_script "${_vendor_fixture}/bad-license-owner.cmake")
file(WRITE "${_vendor_bad_owner_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_vendor_bad_owner_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_vendor_bad_owner_lock}\")\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("in-tree license ownership" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_bad_owner_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "outside its vendored path")
  message(FATAL_ERROR
    "cross-vendor license path failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

file(READ "${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json"
  _main_dependency_lock_json)
string(JSON _main_dependency_count LENGTH
  "${_main_dependency_lock_json}" dependencies)
math(EXPR _main_dependency_last "${_main_dependency_count} - 1")
unset(_highway_port_index)
foreach(_main_dependency_index RANGE 0 ${_main_dependency_last})
  string(JSON _main_dependency_name GET
    "${_main_dependency_lock_json}"
    dependencies ${_main_dependency_index} name)
  if(_main_dependency_name STREQUAL "highway_port")
    set(_highway_port_index ${_main_dependency_index})
  endif()
endforeach()
if(NOT DEFINED _highway_port_index)
  message(FATAL_ERROR "central lock has no highway_port integrated source")
endif()
string(REPLACE
  "third_party/ported/LICENSE-HIGHWAY"
  "third_party/foreign/LICENSE-HIGHWAY"
  _integrated_bad_owner_json "${_main_dependency_lock_json}")
set(_integrated_bad_owner_lock
  "${_hyle_contract_root}/integrated-bad-license-owner.json")
file(WRITE "${_integrated_bad_owner_lock}" "${_integrated_bad_owner_json}")
set(_integrated_bad_owner_script
  "${_hyle_contract_root}/integrated-bad-license-owner.cmake")
file(WRITE "${_integrated_bad_owner_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_bad_owner_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_bad_owner_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")

# A command-line cache variable is data, not an authority to replace the
# central ledger. Trusted fixture code can attest one exact alternate path via
# a GLOBAL property, but a plain -D override must be rejected before parsing.
set(_untrusted_lock_override_script
  "${_hyle_contract_root}/untrusted-lock-override.cmake")
file(WRITE "${_untrusted_lock_override_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("untrusted central lock override" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEPENDENCY_LOCK_FILE=${_integrated_bad_owner_lock}
  -P "${_untrusted_lock_override_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "Refusing untrusted HYLE_DEPENDENCY_LOCK_FILE override")
  message(FATAL_ERROR
    "untrusted central lock override failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()
_hyle_contract_run("integrated-source license ownership" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_bad_owner_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "must be under 'third_party/ported/'")
  message(FATAL_ERROR
    "integrated license ownership failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

string(JSON _integrated_bad_content_json SET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated content_sha256
  "\"0000000000000000000000000000000000000000000000000000000000000000\"")
set(_integrated_bad_content_lock
  "${_hyle_contract_root}/integrated-bad-content.json")
file(WRITE "${_integrated_bad_content_lock}" "${_integrated_bad_content_json}")
set(_integrated_bad_content_script
  "${_hyle_contract_root}/integrated-bad-content.cmake")
file(WRITE "${_integrated_bad_content_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_bad_content_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_bad_content_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_integrated_source_dependencies(highway_port)\n")
_hyle_contract_run("integrated-source adapted content" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_bad_content_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "adapted content does not match the lock")
  message(FATAL_ERROR
    "integrated content mismatch failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# Repeating an adapted path would make the hash projection ambiguous: one lock
# could appear to cover more surfaces while hashing the same bytes twice.
string(JSON _highway_integrated_path_count LENGTH
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths)
string(JSON _highway_first_integrated_path GET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths 0)
string(JSON _integrated_duplicate_path_json SET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths
  ${_highway_integrated_path_count}
  "\"${_highway_first_integrated_path}\"")
set(_integrated_duplicate_path_lock
  "${_hyle_contract_root}/integrated-duplicate-path.json")
file(WRITE "${_integrated_duplicate_path_lock}"
  "${_integrated_duplicate_path_json}")
set(_integrated_duplicate_path_script
  "${_hyle_contract_root}/integrated-duplicate-path.cmake")
file(WRITE "${_integrated_duplicate_path_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_duplicate_path_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_duplicate_path_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("integrated-source duplicate path" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_duplicate_path_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "has duplicate HYLE path")
  message(FATAL_ERROR
    "integrated duplicate path failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# External implementation/package facts also have one owner and one
# declaration.  Exercise the registry in script mode so a duplicate name or a
# stale exported target cannot degrade into a partially relocatable SDK.
set(_package_registry_projection
  "${_hyle_contract_root}/package-registry-projection.cmake")
file(WRITE "${_package_registry_projection}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_STATIC ON)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation FEATURES demo_feature\n"
"  STATIC_FIND_DEPENDENCIES Threads OpenSSL|3.0)\n"
"hyle_package_collect_registry(_targets _features _find_dependencies)\n"
"if(_targets OR NOT _features STREQUAL \"demo_feature\" OR\n"
"   NOT _find_dependencies STREQUAL \"Threads;OpenSSL|3.0\")\n"
"  message(FATAL_ERROR \"unexpected registry projection\")\n"
"endif()\n")
_hyle_contract_run("package dependency registry projection" TRUE
  "${CMAKE_COMMAND}" -P "${_package_registry_projection}")

set(_package_registry_duplicate
  "${_hyle_contract_root}/package-registry-duplicate.cmake")
file(WRITE "${_package_registry_duplicate}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(NAME demo OWNER foundation)\n"
"hyle_register_package_dependency(NAME demo OWNER foundation)\n")
_hyle_contract_run("duplicate package dependency registration" FALSE
  "${CMAKE_COMMAND}" -P "${_package_registry_duplicate}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "registered more than once")
  message(FATAL_ERROR
    "duplicate package dependency failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

set(_package_registry_missing_target
  "${_hyle_contract_root}/package-registry-missing-target.cmake")
file(WRITE "${_package_registry_missing_target}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_STATIC ON)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation STATIC_EXPORT_TARGETS absent_vendor)\n"
"hyle_package_collect_registry(_targets _features _find_dependencies)\n")
_hyle_contract_run("missing package dependency export target" FALSE
  "${CMAKE_COMMAND}" -P "${_package_registry_missing_target}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "requires missing export target")
  message(FATAL_ERROR
    "missing package dependency target failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

_hyle_contract_run("locked package/license registry contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiPackageRegistryContract.cmake")

_hyle_contract_run("64-bit target-platform contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiTargetPlatformContract.cmake")

# Exercise the strict offline preload in a self-contained local Git fixture:
# allow-listing, HEAD/tree/clean checks, license hashes, no-network cache gates,
# and the explicit non-Git escape hatch all have positive/negative coverage.
_hyle_contract_run("verified offline dependency contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiOfflineDependencyContract.cmake")

# HYLE_DEPS_DIR is the one build-tree population location for every HYLE-owned
# FetchContent dependency.  Exercise the test-only path through a self-locked
# GoogleTest-shaped fixture; arbitrary fake direct overrides are intentionally
# rejected by the production provenance wrapper.
set(_fake_gtest_source "${_hyle_contract_root}/fake-googletest")
set(_fake_gtest_lock "${_hyle_contract_root}/fake-googletest-lock.json")
set(_error_test_build "${_hyle_contract_root}/error-test-build")
set(_error_dependency_cache "${_hyle_contract_root}/error-dependencies")
file(MAKE_DIRECTORY "${_fake_gtest_source}")
file(WRITE "${_fake_gtest_source}/LICENSE" "fixture license\n")
file(WRITE "${_fake_gtest_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(fake_googletest LANGUAGES CXX)\n"
"add_library(gtest INTERFACE)\n"
"add_library(GTest::gtest ALIAS gtest)\n"
"add_library(gtest_main INTERFACE)\n"
"add_library(GTest::gtest_main ALIAS gtest_main)\n")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
hyle_compute_expanded_source_tree_sha256(
  "${_fake_gtest_source}" _fake_gtest_expanded_tree)
string(SHA256 _fake_gtest_license_sha256 "fixture license\n")
file(WRITE "${_fake_gtest_lock}"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [{
    \"name\": \"googletest\", \"class\": \"fetchcontent\",
    \"repository\": \"${HYLE_DEP_GOOGLETEST_REPOSITORY}\",
    \"version\": \"fixture-${HYLE_DEP_GOOGLETEST_VERSION}\",
    \"revision\": \"${HYLE_DEP_GOOGLETEST_COMMIT}\",
    \"git_tree\": \"${HYLE_DEP_GOOGLETEST_GIT_TREE}\",
    \"archive\": {
      \"url\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_URL}\",
      \"sha256\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_SHA256}\"
    },
    \"expanded_tree_sha256\": \"${_fake_gtest_expanded_tree}\",
    \"license\": { \"expression\": \"BSD-3-Clause\", \"files\": [{
      \"root\": \"dependency\", \"path\": \"LICENSE\",
      \"sha256\": \"${_fake_gtest_license_sha256}\"
    }] }
  }]
}
")
set(_fake_gtest_lock_attestation
  "${_hyle_contract_root}/fake-gtest-lock-attestation.cmake")
file(WRITE "${_fake_gtest_lock_attestation}"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_fake_gtest_lock}\")\n")
_hyle_contract_run("error-only dependency cache ownership" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_error_test_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_COMPONENTS=error
  -DHYLE_BUILD_TESTS=ON
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_DEPS_DIR=${_error_dependency_cache}
  -DHYLE_DEPENDENCY_LOCK_FILE=${_fake_gtest_lock}
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${_fake_gtest_lock_attestation}
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${_fake_gtest_source})
if(NOT IS_DIRECTORY "${_error_dependency_cache}/googletest-build")
  message(FATAL_ERROR
    "error-only tests ignored HYLE_DEPS_DIR='${_error_dependency_cache}'")
endif()
if(IS_DIRECTORY "${_error_test_build}/_deps")
  message(FATAL_ERROR
    "error-only tests leaked FetchContent state into '${_error_test_build}/_deps'")
endif()

message(STATUS "HYLE component contracts passed")
