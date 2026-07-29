include(FetchContent)

# GoogleTest (charter: unit test framework). Pinned tag, FetchContent GIT form (same
# open-box-auto-download contract as ggml): first `cmake` fetches it under the
# root-owned HYLE_DEPS_DIR (build-tree `_deps` by default). No pre-placed mirror.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE)
# Use shared CRT on MSVC to match our build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_custom_target(hyle_tests)
set_target_properties(hyle_tests PROPERTIES FOLDER "hyle/validation")

function(hyle_ensure_module_test_target module)
  if(NOT TARGET hyle_${module}_tests)
    add_custom_target(hyle_${module}_tests)
    set_target_properties(hyle_${module}_tests PROPERTIES FOLDER "hyle/validation")
    add_dependencies(hyle_tests hyle_${module}_tests)
  endif()
endfunction()

function(hyle_add_c_test module source)
  if(NOT TARGET hyle_${module})
    return()
  endif()
  set(target hyle_${module}_c_test)
  add_executable(${target} "${source}")
  set_target_properties(${target} PROPERTIES
    FOLDER "hyle/tests/${module}"
    HYLE_TEST_MODULE "${module}")
  target_link_libraries(${target} PRIVATE hyle_${module} ${ARGN})
  set_source_files_properties("${source}" PROPERTIES LANGUAGE C)
  hyle_apply_warnings(${target})
  hyle_apply_sanitizer(${target})
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "${module};c-abi")
  hyle_ensure_module_test_target(${module})
  add_dependencies(hyle_${module}_tests ${target})
endfunction()

function(hyle_add_c_surface_test module)
  if(NOT TARGET hyle_${module})
    return()
  endif()
  set(_static_arg "")
  if(HYLE_STATIC)
    set(_static_arg --static)
  endif()
  set(_private_symbol_args "")
  if("${module}" STREQUAL "mem" AND NOT HYLE_STATIC)
    list(APPEND _private_symbol_args
      --forbid-prefix mi_
      --forbid-prefix _mi_)
  endif()
  add_test(NAME hyle_${module}_c_surface_test
           COMMAND "${Python3_EXECUTABLE}"
                   "${HYLE_SOURCE_DIR}/scripts/c-surface-check.py"
                   --module "${module}"
                   --library "$<TARGET_FILE:hyle_${module}>"
                   --symbol-tool "${_hyle_c_symbol_tool}"
                   ${_static_arg}
                   ${_private_symbol_args})
  set_tests_properties(hyle_${module}_c_surface_test PROPERTIES
    LABELS "${module};c-abi")
  hyle_ensure_module_test_target(${module})
  add_dependencies(hyle_${module}_tests hyle_${module})
endfunction()

function(hyle_add_header_self_contained_test module)
  cmake_parse_arguments(ARG "NO_C" "HEADER_DIR;C_HEADER" "HEADERS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_header_self_contained_test(${module}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_header_self_contained_test(${module}): keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(ARG_NO_C AND ARG_C_HEADER)
    message(FATAL_ERROR
      "hyle_add_header_self_contained_test(${module}): NO_C and C_HEADER conflict")
  endif()
  if(NOT TARGET hyle_${module})
    return()
  endif()
  set(_hyle_vs_args "")
  if(DEFINED CMAKE_GENERATOR_INSTANCE)
    set(_hyle_vs_value "${CMAKE_GENERATOR_INSTANCE}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-instance "${_hyle_vs_value}")
    endif()
  endif()
  if(DEFINED CMAKE_VS_PLATFORM_NAME)
    set(_hyle_vs_value "${CMAKE_VS_PLATFORM_NAME}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-platform "${_hyle_vs_value}")
    endif()
  endif()
  if(DEFINED CMAKE_VS_PLATFORM_TOOLSET_VERSION)
    set(_hyle_vs_value "${CMAKE_VS_PLATFORM_TOOLSET_VERSION}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-toolset-version "${_hyle_vs_value}")
    endif()
  endif()
  set(_hyle_header_args "")
  if(ARG_HEADER_DIR)
    list(APPEND _hyle_header_args --header-dir "${ARG_HEADER_DIR}")
  endif()
  foreach(_hyle_header IN LISTS ARG_HEADERS)
    list(APPEND _hyle_header_args --header "${_hyle_header}")
  endforeach()
  if(ARG_NO_C)
    list(APPEND _hyle_header_args --no-c-header)
  elseif(ARG_C_HEADER)
    list(APPEND _hyle_header_args --c-header "${ARG_C_HEADER}")
  endif()
  add_test(NAME hyle_${module}_header_self_contained_test
           COMMAND "${Python3_EXECUTABLE}"
                   "${HYLE_SOURCE_DIR}/scripts/header-self-contained.py"
                   --module "${module}"
                   --include-dir "${HYLE_SOURCE_DIR}/include"
                   --cc "${CMAKE_C_COMPILER}"
                   --cxx "${CMAKE_CXX_COMPILER}"
                   --c-compiler-id "${CMAKE_C_COMPILER_ID}"
                   --cxx-compiler-id "${CMAKE_CXX_COMPILER_ID}"
                   --c-frontend "${CMAKE_C_COMPILER_FRONTEND_VARIANT}"
                   --cxx-frontend "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}"
                   ${_hyle_header_args}
                   ${_hyle_vs_args})
  set_tests_properties(hyle_${module}_header_self_contained_test PROPERTIES
    LABELS "${module};headers")
  hyle_ensure_module_test_target(${module})
  unset(_hyle_vs_args)
  unset(_hyle_vs_value)
  unset(_hyle_header_args)
  unset(_hyle_header)
endfunction()
# Keep reviewed C manifests, pure-C consumers, and DLL/.so exports in sync.
# Before first release this is drift detection, not a compatibility promise.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(WIN32)
  get_filename_component(_hyle_linker_dir "${CMAKE_LINKER}" DIRECTORY)
  set(_hyle_c_symbol_tool "${_hyle_linker_dir}/dumpbin.exe")
elseif(CMAKE_NM)
  set(_hyle_c_symbol_tool "${CMAKE_NM}")
else()
  find_program(_hyle_c_symbol_tool NAMES nm REQUIRED)
endif()

# Every module owns its test registration beside its test sources. A real
# subdirectory gives each module independent source/binary/variable scope while
# target names and the common output directory remain stable.
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  set(_hyle_test_fragment "${CMAKE_CURRENT_SOURCE_DIR}/${_component}/CMakeLists.txt")
  if(TARGET hyle_${_component} AND NOT EXISTS "${_hyle_test_fragment}")
    message(FATAL_ERROR
      "Selected HYLE component '${_component}' has no test ownership fragment: "
      "${_hyle_test_fragment}")
  endif()
  if(TARGET hyle_${_component})
    add_subdirectory("${_component}")
    if(NOT TARGET hyle_${_component}_tests)
      message(FATAL_ERROR
        "Test ownership fragment for '${_component}' registered no module aggregate")
    endif()
    get_property(_hyle_module_targets
      DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_component}"
      PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_hyle_module_target IN LISTS _hyle_module_targets)
      get_target_property(_hyle_module_target_type
        ${_hyle_module_target} TYPE)
      if(_hyle_module_target_type STREQUAL "EXECUTABLE")
        get_target_property(_hyle_test_owner
          ${_hyle_module_target} HYLE_TEST_MODULE)
        if(NOT _hyle_test_owner STREQUAL "${_component}")
          message(FATAL_ERROR
            "Test executable '${_hyle_module_target}' in tests/${_component} is not "
            "registered through the ${_component} test aggregate")
        endif()
      endif()
    endforeach()
  endif()
endforeach()
unset(_hyle_test_owner)
unset(_hyle_module_target_type)
unset(_hyle_module_target)
unset(_hyle_module_targets)
unset(_hyle_test_fragment)
unset(_component)
