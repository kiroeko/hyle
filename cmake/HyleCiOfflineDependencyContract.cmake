# Positive and negative contracts for locked source overrides and the verified
# offline dependency preload.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()
find_package(Git REQUIRED)
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")

set(_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

set(_root "${HYLE_SMOKE_ROOT}/offline-dependency-contract")
set(_bundle "${_root}/bundle")
set(_source "${_bundle}/deps/demo")
set(_mapped_project "${_root}/mapped-project")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY
  "${_bundle}/cmake" "${_source}/nested" "${_mapped_project}")
configure_file("${HYLE_SOURCE_DIR}/hyle-offline-deps.cmake"
               "${_bundle}/hyle-offline-deps.cmake" COPYONLY)
configure_file("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake"
               "${_bundle}/cmake/HyleDependencyVersions.cmake" COPYONLY)

# Every production/test FetchContent input must pass through the one strict
# wrapper.  This makes a new direct override impossible to forget locally.
set(_locked_consumer_specs
  "cmake/HyleSimdDependencies.cmake|highway"
  "cmake/HyleSimdDependencies.cmake|xsimd"
  "cmake/HyleIoDependencies.cmake|nghttp2"
  "tests/CMakeLists.txt|googletest")
foreach(_spec IN LISTS _locked_consumer_specs)
  string(REPLACE "|" ";" _fields "${_spec}")
  list(GET _fields 0 _consumer_file)
  list(GET _fields 1 _dependency)
  file(READ "${HYLE_SOURCE_DIR}/${_consumer_file}" _consumer_content)
  string(FIND "${_consumer_content}"
    "hyle_fetchcontent_make_available_locked(${_dependency})" _locked_call)
  string(FIND "${_consumer_content}"
    "FetchContent_MakeAvailable(${_dependency})" _raw_call)
  if(_locked_call LESS 0 OR NOT _raw_call LESS 0)
    message(FATAL_ERROR
      "${_consumer_file} must consume '${_dependency}' only through "
      "hyle_fetchcontent_make_available_locked()")
  endif()
endforeach()

function(_run_checked label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${_result}:\n${_stdout}\n${_stderr}")
  endif()
endfunction()

function(_write_fixture_lock revision tree expanded_tree)
  string(SHA256 _license_sha256 "demo license\n")
  file(WRITE "${_bundle}/cmake/HyleDependencyLock.json"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [
    {
      \"name\": \"demo\", \"class\": \"fetchcontent\",
      \"repository\": \"https://github.com/example/demo.git\",
      \"version\": \"fixture\",
      \"revision\": \"${revision}\", \"git_tree\": \"${tree}\",
      \"archive\": {
        \"url\": \"https://codeload.github.com/example/demo/tar.gz/${revision}\",
        \"sha256\": \"1111111111111111111111111111111111111111111111111111111111111111\"
      },
      \"expanded_tree_sha256\": \"${expanded_tree}\",
      \"license\": { \"expression\": \"MIT\", \"files\": [
        { \"root\": \"dependency\", \"path\": \"LICENSE\", \"sha256\": \"${_license_sha256}\" }
      ] }
    }
  ]
}
")
endfunction()

function(_check_configure_result label expect_success expected_pattern
         result stdout stderr)
  set(_output "${stdout}\n${stderr}")
  if(expect_success AND NOT result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${result}:\n${_output}")
  elseif(NOT expect_success AND result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded")
  elseif(NOT expect_success AND NOT _output MATCHES "${expected_pattern}")
    message(FATAL_ERROR
      "${label} failed for the wrong reason; expected '${expected_pattern}':\n"
      "${_output}")
  endif()
endfunction()

function(_configure_offline label expect_success expected_pattern suffix)
  file(REMOVE_RECURSE "${_root}/build-offline-${suffix}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" ${ARGN}
      -C "${_bundle}/hyle-offline-deps.cmake"
      -S "${_project}" -B "${_root}/build-offline-${suffix}"
      ${_generator_args}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  _check_configure_result(
    "${label}" ${expect_success} "${expected_pattern}"
    ${_result} "${_stdout}" "${_stderr}")
endfunction()

function(_configure_direct label expect_success expected_pattern suffix)
  file(REMOVE_RECURSE "${_root}/build-direct-${suffix}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" ${ARGN}
      "-DFETCHCONTENT_SOURCE_DIR_DEMO=${_source}"
      -S "${_mapped_project}" -B "${_root}/build-direct-${suffix}"
      ${_generator_args}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  _check_configure_result(
    "${label}" ${expect_success} "${expected_pattern}"
    ${_result} "${_stdout}" "${_stderr}")
endfunction()

# One fixture is both a committed Git checkout and, with .git moved outside,
# an exact expanded archive tree.  The safe relative link mirrors real ggml
# archive behavior and proves that links are not flattened on Windows.
file(WRITE "${_source}/LICENSE" "demo license\n")
file(WRITE "${_source}/nested/payload.txt" "locked payload\n")
file(WRITE "${_source}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.25)\n"
  "project(demo LANGUAGES NONE)\n"
  "add_library(demo INTERFACE)\n")
file(WRITE "${_source}/.gitignore" "ignored-generated.txt\n")
file(CREATE_LINK "nested/payload.txt" "${_source}/payload-link"
     SYMBOLIC RESULT _safe_link_result)
if(NOT _safe_link_result STREQUAL "0")
  message(FATAL_ERROR
    "Cannot create fixture safe symbolic link: ${_safe_link_result}")
endif()
hyle_compute_expanded_source_tree_sha256("${_source}" _expanded_tree)

_run_checked("fixture git init" "${GIT_EXECUTABLE}" init --quiet "${_source}")
_run_checked("fixture git autocrlf"
  "${GIT_EXECUTABLE}" -C "${_source}" config core.autocrlf false)
_run_checked("fixture git symlinks"
  "${GIT_EXECUTABLE}" -C "${_source}" config core.symlinks true)
_run_checked("fixture git user"
  "${GIT_EXECUTABLE}" -C "${_source}" config user.name HYLE-CI)
_run_checked("fixture git email"
  "${GIT_EXECUTABLE}" -C "${_source}" config user.email hyle-ci@example.invalid)
_run_checked("fixture git add"
  "${GIT_EXECUTABLE}" -C "${_source}" add -A)
_run_checked("fixture git commit"
  "${GIT_EXECUTABLE}" -C "${_source}" commit --quiet -m fixture)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_source}"
  rev-parse HEAD OUTPUT_VARIABLE _revision OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_source}"
  rev-parse "HEAD^{tree}" OUTPUT_VARIABLE _tree OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
_write_fixture_lock("${_revision}" "${_tree}" "${_expanded_tree}")

file(WRITE "${_mapped_project}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)
project(hyle_locked_source_contract LANGUAGES NONE)
list(APPEND CMAKE_MODULE_PATH \"${_bundle}/cmake\")
include(HyleDependencyVersions)
include(FetchContent)
FetchContent_Declare(demo
  URL https://codeload.github.com/example/demo/tar.gz/${_revision}
  URL_HASH SHA256=1111111111111111111111111111111111111111111111111111111111111111)
hyle_fetchcontent_make_available_locked(demo)
if(NOT TARGET demo)
  message(FATAL_ERROR \"locked demo source was not made available\")
endif()
")

set(_project "${_mapped_project}")
_configure_offline("exact clean Git checkout" TRUE "" git-exact)
_configure_direct("direct exact Git override" TRUE "" direct-git-exact)

# `git status --untracked-files=all` intentionally omits ignored files.  They
# are still unsafe in an exact source input because upstream CMake may glob or
# prefer an injected generated header/cache entry.
file(WRITE "${_source}/ignored-generated.txt" "ignored but consumed\n")
_configure_offline("ignored file in Git checkout" FALSE
  "checkout contains ignored files" git-ignored)
_configure_direct("direct Git override with ignored file" FALSE
  "checkout contains ignored files" direct-git-ignored)
file(REMOVE "${_source}/ignored-generated.txt")

file(MAKE_DIRECTORY "${_bundle}/deps/typo")
_configure_offline("unknown dependency directory" FALSE
  "is not a product source name allowed by" unknown)
file(REMOVE_RECURSE "${_bundle}/deps/typo")

file(APPEND "${_source}/CMakeLists.txt" "# second commit\n")
_run_checked("fixture second git add"
  "${GIT_EXECUTABLE}" -C "${_source}" add CMakeLists.txt)
_run_checked("fixture second git commit"
  "${GIT_EXECUTABLE}" -C "${_source}" commit --quiet -m second)
_configure_offline("wrong Git HEAD" FALSE "HEAD mismatch" git-head)
_run_checked("restore fixture revision"
  "${GIT_EXECUTABLE}" -C "${_source}" switch --quiet --detach "${_revision}")

_write_fixture_lock("${_revision}"
  "0000000000000000000000000000000000000000" "${_expanded_tree}")
_configure_offline("wrong Git tree" FALSE "Git tree mismatch" git-tree)
_write_fixture_lock("${_revision}" "${_tree}" "${_expanded_tree}")

file(WRITE "${_source}/untracked.txt" "dirty\n")
_configure_offline("dirty Git checkout" FALSE "checkout is dirty" git-dirty)
file(REMOVE "${_source}/untracked.txt")

# Remove Git metadata from the dependency tree, not merely rename it inside the
# tree (which would correctly count as an added archive entry).
set(_saved_git "${_root}/demo.git.saved")
file(RENAME "${_source}/.git" "${_saved_git}")
_configure_offline("exact official-archive expansion" TRUE "" expanded-exact)
_configure_direct("direct exact expanded override" TRUE "" direct-expanded)

file(WRITE "${_source}/nested/payload.txt" "tampered payload\n")
_configure_direct("direct override modified file" FALSE
  "expanded source tree mismatch" direct-modified)
_configure_direct("direct override cannot self-attest" FALSE
  "expanded source tree mismatch" direct-self-attest
  -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON)
_configure_direct("direct override cannot spoof preload attestation" FALSE
  "expanded source tree mismatch" direct-spoof-attestation
  -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON
  "-DHYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_DEMO=${_source}")
file(WRITE "${_source}/nested/payload.txt" "locked payload\n")

file(WRITE "${_source}/added.txt" "added\n")
_configure_direct("direct override added file" FALSE
  "expanded source tree mismatch" direct-added)
file(REMOVE "${_source}/added.txt")

file(REMOVE "${_source}/nested/payload.txt")
_configure_direct("direct override deleted file" FALSE
  "expanded source tree mismatch" direct-deleted)
file(WRITE "${_source}/nested/payload.txt" "locked payload\n")

file(MAKE_DIRECTORY "${_source}/added-empty-directory")
_configure_direct("direct override added empty directory" FALSE
  "expanded source tree mismatch" direct-added-directory)
file(REMOVE_RECURSE "${_source}/added-empty-directory")

# The target resolves to the same file, but exact archive identity retains the
# original link spelling and therefore rejects this tamper.
file(REMOVE "${_source}/payload-link")
file(CREATE_LINK "nested/./payload.txt" "${_source}/payload-link"
     SYMBOLIC RESULT _spelling_link_result)
if(NOT _spelling_link_result STREQUAL "0")
  message(FATAL_ERROR
    "Cannot recreate fixture symbolic link: ${_spelling_link_result}")
endif()
_configure_direct("direct override link spelling tamper" FALSE
  "expanded source tree mismatch" direct-link-spelling)
file(REMOVE "${_source}/payload-link")
file(CREATE_LINK "nested/payload.txt" "${_source}/payload-link"
     SYMBOLIC RESULT _restore_link_result)
if(NOT _restore_link_result STREQUAL "0")
  message(FATAL_ERROR
    "Cannot restore fixture symbolic link: ${_restore_link_result}")
endif()

file(WRITE "${_root}/outside.txt" "outside\n")
file(CREATE_LINK "../outside.txt" "${_source}/escape-link"
     SYMBOLIC RESULT _escape_link_result)
if(NOT _escape_link_result STREQUAL "0")
  message(FATAL_ERROR
    "Cannot create fixture escaping symbolic link: ${_escape_link_result}")
endif()
_configure_direct("direct override escaping symbolic link" FALSE
  "symbolic link escaping its root" direct-escape-link)
file(REMOVE "${_source}/escape-link")

file(WRITE "${_source}/externally-attested.marker" "external proof\n")
_configure_offline("unsafe mismatch rejects every symbolic link" FALSE
  "contains symbolic links"
  expanded-external-proof-link
  -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON)
file(REMOVE "${_source}/payload-link")
_configure_offline("explicit externally-attested mismatch" TRUE ""
  expanded-external-proof
  -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON)
file(REMOVE "${_source}/externally-attested.marker")

file(WRITE "${_source}/LICENSE" "modified license\n")
_configure_offline("unsafe escape still rejects bad license" FALSE
  "license hash mismatch" expanded-bad-license
  -DHYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES=ON)
file(WRITE "${_source}/LICENSE" "demo license\n")
file(CREATE_LINK "nested/payload.txt" "${_source}/payload-link"
     SYMBOLIC RESULT _restore_after_unsafe_link_result)
if(NOT _restore_after_unsafe_link_result STREQUAL "0")
  message(FATAL_ERROR
    "Cannot restore fixture symbolic link after unsafe contract: "
    "${_restore_after_unsafe_link_result}")
endif()
file(RENAME "${_saved_git}" "${_source}/.git")

# A source-free component closure needs no deps/ directory at all, while an
# actually selected but absent FetchContent source fails before any network use.
set(_source_free_project "${_root}/source-free-project")
set(_missing_project "${_root}/missing-project")
file(MAKE_DIRECTORY "${_source_free_project}" "${_missing_project}")
file(WRITE "${_source_free_project}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)
project(hyle_offline_source_free_contract LANGUAGES NONE)
if(NOT FETCHCONTENT_FULLY_DISCONNECTED OR
   NOT FETCHCONTENT_UPDATES_DISCONNECTED)
  message(FATAL_ERROR \"offline network gates are not enabled\")
endif()
")
file(WRITE "${_missing_project}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)
project(hyle_offline_missing_source_contract LANGUAGES NONE)
list(APPEND CMAKE_MODULE_PATH \"${_bundle}/cmake\")
include(HyleDependencyVersions)
include(FetchContent)
FetchContent_Declare(demo
  URL https://codeload.github.com/example/demo/tar.gz/${_revision}
  URL_HASH SHA256=1111111111111111111111111111111111111111111111111111111111111111)
hyle_fetchcontent_make_available_locked(demo)
")
file(RENAME "${_bundle}/deps" "${_root}/deps.saved")
set(_project "${_source_free_project}")
_configure_offline("source-free selection without deps directory" TRUE ""
  source-free)
set(_project "${_missing_project}")
_configure_offline("selected dependency missing offline source" FALSE
  "Expanded dependency source must be a real directory" missing-source)
file(RENAME "${_root}/deps.saved" "${_bundle}/deps")

message(STATUS
  "HYLE locked/offline dependency contract passed positive and negative cases")
