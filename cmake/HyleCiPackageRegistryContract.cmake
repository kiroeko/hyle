cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "HyleCiPackageRegistryContract requires -D${_required}=...")
  endif()
endforeach()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)
file(TO_CMAKE_PATH "${HYLE_SOURCE_DIR}" _source_dir)
set(_root "${HYLE_SMOKE_ROOT}/package-registry-contract")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

set(_locked_license "${_root}/LOCKED-LICENSE")
set(_other_license "${_root}/OTHER-LICENSE")
file(WRITE "${_locked_license}" "demo license\r\n")
file(WRITE "${_other_license}" "other license\n")
string(SHA256 _locked_sha256 "demo license\n")
file(TO_CMAKE_PATH "${_locked_license}" _locked_license_cmake)
file(TO_CMAKE_PATH "${_other_license}" _other_license_cmake)

function(_hyle_registry_contract_run label expect_success expected_pattern script)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -P "${script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "${label} failed unexpectedly:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR "${label} unexpectedly succeeded")
    endif()
    if(NOT "${_output}" MATCHES "${expected_pattern}")
      message(FATAL_ERROR
        "${label} failed for the wrong reason; expected '${expected_pattern}':\n"
        "${_output}")
    endif()
  endif()
endfunction()

string(CONCAT _fixture_prefix
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_DEPENDENCY_NAMES demo)\n"
"set(HYLE_DEP_DEMO_LICENSE_FILE_SPECS \"hyle|third_party/demo/LICENSE|${_locked_sha256}\")\n"
"include(HylePackageRegistry)\n")

set(_positive "${_root}/positive.cmake")
file(WRITE "${_positive}" "${_fixture_prefix}"
"set(HYLE_DEP_DEMO_CLASS in_tree)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS \"${_locked_license_cmake}|share/licenses/Hyle|demo-LICENSE\")\n"
"_hyle_package_dependency_property(demo LOCKED_DEPENDENCIES _locked)\n"
"if(NOT _locked STREQUAL \"demo\")\n"
"  message(FATAL_ERROR \"locked dependency was not retained\")\n"
"endif()\n")
_hyle_registry_contract_run(
  "locked package registry projection" TRUE "" "${_positive}")

set(_duplicate_owner "${_root}/duplicate-lock-owner.cmake")
file(WRITE "${_duplicate_owner}" "${_fixture_prefix}"
"set(HYLE_DEP_DEMO_CLASS in_tree)\n"
"hyle_register_package_dependency(\n"
"  NAME first OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS \"${_locked_license_cmake}|share/licenses/Hyle|first-LICENSE\")\n"
"hyle_register_package_dependency(\n"
"  NAME second OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS \"${_locked_license_cmake}|share/licenses/Hyle|second-LICENSE\")\n")
_hyle_registry_contract_run(
  "one lock record has one package registry owner" FALSE
  "already owned by package registry.*first" "${_duplicate_owner}")

set(_missing_license "${_root}/missing-license.cmake")
file(WRITE "${_missing_license}" "${_fixture_prefix}"
"set(HYLE_DEP_DEMO_CLASS fetchcontent)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS \"${_other_license_cmake}|share/licenses/Hyle|demo-LICENSE\")\n")
_hyle_registry_contract_run(
  "missing locked runtime license" FALSE "does not install locked license hash"
  "${_missing_license}")

set(_benchmark "${_root}/benchmark.cmake")
file(WRITE "${_benchmark}" "${_fixture_prefix}"
"set(HYLE_DEP_DEMO_CLASS benchmark)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS \"${_locked_license_cmake}|share/licenses/Hyle|demo-LICENSE\")\n")
_hyle_registry_contract_run(
  "benchmark lock in production registry" FALSE "benchmark-only lock"
  "${_benchmark}")

set(_duplicate_hash "${_root}/duplicate-same-hash.cmake")
file(WRITE "${_duplicate_hash}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_DEPENDENCY_NAMES demo twin)\n"
"set(HYLE_DEP_DEMO_CLASS in_tree)\n"
"set(HYLE_DEP_DEMO_LICENSE_FILE_SPECS \"hyle|third_party/demo/LICENSE|${_locked_sha256}\")\n"
"set(HYLE_DEP_TWIN_CLASS in_tree)\n"
"set(HYLE_DEP_TWIN_LICENSE_FILE_SPECS \"hyle|third_party/twin/LICENSE|${_locked_sha256}\")\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME combined OWNER foundation LOCKED_DEPENDENCIES demo twin\n"
"  RUNTIME_FILE_SPECS \"${_locked_license_cmake}|share/licenses/Hyle|shared-LICENSE\")\n")
_hyle_registry_contract_run(
  "one runtime asset cannot satisfy two same-hash locks" FALSE
  "does not install locked license hash.*distinct runtime asset.*twin"
  "${_duplicate_hash}")

set(_extra_runtime "${_root}/extra-runtime.cmake")
file(WRITE "${_extra_runtime}" "${_fixture_prefix}"
"set(HYLE_DEP_DEMO_CLASS in_tree)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation LOCKED_DEPENDENCIES demo\n"
"  RUNTIME_FILE_SPECS\n"
"    \"${_locked_license_cmake}|share/licenses/Hyle|demo-LICENSE\"\n"
"    \"${_other_license_cmake}|share/licenses/Hyle|unlocked-NOTICE\")\n")
_hyle_registry_contract_run(
  "locked registry rejects extra runtime assets" FALSE
  "contains unlocked runtime asset" "${_extra_runtime}")

set(_unaccounted "${_root}/unaccounted.cmake")
file(WRITE "${_unaccounted}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation\n"
"  RUNTIME_FILE_SPECS \"${_other_license_cmake}|share/licenses/Hyle|demo-LICENSE\")\n")
_hyle_registry_contract_run(
  "unaccounted runtime asset" FALSE "runtime assets without LOCKED_DEPENDENCIES"
  "${_unaccounted}")

set(_unlocked_escape "${_root}/unlocked-escape.cmake")
file(WRITE "${_unlocked_escape}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME attribution OWNER foundation\n"
"  UNLOCKED_RUNTIME_ASSET_REASON \"license for incorporated source\"\n"
"  RUNTIME_FILE_SPECS \"${_other_license_cmake}|share/licenses/Hyle|NOTICE\")\n")
_hyle_registry_contract_run(
  "generic unlocked runtime escape is not accepted" FALSE "unknown arguments"
  "${_unlocked_escape}")

set(_unsafe_destination "${_root}/unsafe-destination.cmake")
file(WRITE "${_unsafe_destination}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"include(HylePackageRegistry)\n"
"_hyle_package_parse_asset(demo file\n"
"  \"${_other_license_cmake}|share/../escape|NOTICE\" _src _dst _name)\n")
_hyle_registry_contract_run(
  "package destination traversal" FALSE "must be a safe"
  "${_unsafe_destination}")

set(_unsafe_rename "${_root}/unsafe-rename.cmake")
file(WRITE "${_unsafe_rename}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${_source_dir}/cmake\")\n"
"include(HylePackageRegistry)\n"
"_hyle_package_parse_asset(demo file\n"
"  \"${_other_license_cmake}|share/licenses/Hyle|../NOTICE\" _src _dst _name)\n")
_hyle_registry_contract_run(
  "package rename traversal" FALSE "RENAME must"
  "${_unsafe_rename}")

message(STATUS "HYLE package registry lock contract passed")
