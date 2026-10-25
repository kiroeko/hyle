include(FetchContent)

# GoogleTest is a validation implementation detail. An embedding parent may
# already own the canonical target; reuse it instead of populating a second
# copy with colliding gtest/gtest_main target names. Otherwise fetch HYLE's
# pinned archive through HYLE_DEP_SOURCE_CACHE, keep its binary/subbuild under
# HYLE_DEP_BUILD_DIR, and keep generic GoogleTest options out of
# the parent cache/install graph. HYLE does not use GMock.
set(_hyle_owns_googletest FALSE)
if(TARGET GTest::gtest_main)
  message(STATUS "HYLE tests: reusing parent-provided GTest::gtest_main")
elseif(TARGET gtest OR TARGET gtest_main OR TARGET GTest::gtest)
  message(FATAL_ERROR
    "HYLE_BUILD_TESTS found a partial GoogleTest target set without "
    "GTest::gtest_main. Provide the canonical imported target or configure "
    "HYLE_BUILD_TESTS=OFF.")
else()
  FetchContent_Declare(
    googletest
    URL      ${HYLE_DEP_GOOGLETEST_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_GOOGLETEST_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    set(gtest_force_shared_crt ON)
    set(BUILD_GMOCK OFF)
    set(INSTALL_GTEST OFF)
    set(GTEST_HAS_ABSL OFF)
    set(gtest_build_samples OFF)
    set(gtest_build_tests OFF)
    set(gtest_disable_pthreads OFF)
    set(gtest_hide_internal_symbols OFF)
    hyle_fetchcontent_make_available_locked(googletest)
  endblock()
  set(_hyle_owns_googletest TRUE)
endif()

# MSVC's ASan-enabled STL records container-annotation mismatch metadata in
# every object. A source-built GoogleTest must therefore use the same compiler
# instrumentation as HYLE's test executables; otherwise the linker correctly
# rejects the mixed annotated/unannotated ABI. Do not mutate a parent-owned
# target: an embedding parent remains responsible for supplying an
# ASan-compatible GoogleTest when it asks HYLE to build tests in that mode.
if(_hyle_owns_googletest AND MSVC AND HYLE_SANITIZER STREQUAL "asan")
  hyle_apply_sanitizer(gtest)
  hyle_apply_sanitizer(gtest_main)
endif()
unset(_hyle_owns_googletest)

add_custom_target(hyle_tests)
set_target_properties(hyle_tests PROPERTIES FOLDER "hyle/validation")

function(hyle_ensure_module_test_target module)
  if(NOT TARGET hyle_${module}_tests)
    add_custom_target(hyle_${module}_tests)
    set_target_properties(hyle_${module}_tests PROPERTIES FOLDER "hyle/validation")
    add_dependencies(hyle_tests hyle_${module}_tests)
  endif()
endfunction()

function(hyle_add_header_self_contained_test module)
  cmake_parse_arguments(ARG "" "HEADER_DIR" "HEADERS;EXTRA_INCLUDE_DIRS" ${ARGN})
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
  foreach(_hyle_extra_include_dir IN LISTS ARG_EXTRA_INCLUDE_DIRS)
    list(APPEND _hyle_header_args
      --extra-include-dir "${_hyle_extra_include_dir}")
  endforeach()
  add_test(NAME hyle_${module}_header_self_contained_test
           COMMAND "${Python3_EXECUTABLE}"
                   "${HYLE_SOURCE_DIR}/scripts/header-self-contained.py"
                   --module "${module}"
                   --include-dir "${HYLE_SOURCE_DIR}/include"
                   --cxx "${CMAKE_CXX_COMPILER}"
                   --cxx-compiler-id "${CMAKE_CXX_COMPILER_ID}"
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
  unset(_hyle_extra_include_dir)
endfunction()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyle_engine_components_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/engine_components_test.py")
set_tests_properties(hyle_engine_components_test PROPERTIES
  LABELS "tooling;build;components")
add_test(NAME hyle_product_macro_contract_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/product_macro_contract_test.py")
set_tests_properties(hyle_product_macro_contract_test PROPERTIES
  LABELS "tooling;build;package")
add_test(NAME hyle_visibility_generation_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/visibility_generation_test.py"
                 --compiler "${CMAKE_CXX_COMPILER}")
set_tests_properties(hyle_visibility_generation_test PROPERTIES
  LABELS "tooling;build;headers")
add_test(NAME hyle_shared_export_hygiene_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/shared_export_hygiene_test.py")
set_tests_properties(hyle_shared_export_hygiene_test PROPERTIES
  LABELS "tooling;build;package")

# Consumer input grammar is host-only and does not need a model, Compute
# product, native SDK, or GoogleTest linkage. Keep it in the normal build/test
# aggregate so the regression is exercised by scripts/test.py as well as CTest.
add_executable(hyle_qwen_generation_schedule_test tools/qwen_generation_schedule_test.cc)
target_link_libraries(hyle_qwen_generation_schedule_test PRIVATE hyle_build_contract)
target_compile_features(hyle_qwen_generation_schedule_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_qwen_generation_schedule_test)
hyle_apply_sanitizer(hyle_qwen_generation_schedule_test)
set_target_properties(hyle_qwen_generation_schedule_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_qwen_generation_schedule_test)
add_test(NAME hyle_qwen_generation_schedule_test
  COMMAND hyle_qwen_generation_schedule_test
    "${CMAKE_CURRENT_BINARY_DIR}/qwen-generation-schedule/$<CONFIG>")
set_tests_properties(hyle_qwen_generation_schedule_test PROPERTIES
  LABELS "tooling;generation;parser" TIMEOUT 30)

add_executable(hyle_inference_version_admission_test tools/inference_version_admission_test.cc)
target_link_libraries(hyle_inference_version_admission_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_version_admission_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_version_admission_test)
hyle_apply_sanitizer(hyle_inference_version_admission_test)
set_target_properties(hyle_inference_version_admission_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_version_admission_test)
add_test(NAME hyle_inference_version_admission_test COMMAND hyle_inference_version_admission_test)
set_tests_properties(hyle_inference_version_admission_test PROPERTIES
  LABELS "tooling;generation;cache" TIMEOUT 30)

add_executable(hyle_inference_request_window_test tools/inference_request_window_test.cc)
target_link_libraries(hyle_inference_request_window_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_request_window_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_request_window_test)
hyle_apply_sanitizer(hyle_inference_request_window_test)
set_target_properties(hyle_inference_request_window_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_request_window_test)
add_test(NAME hyle_inference_request_window_test COMMAND hyle_inference_request_window_test)
set_tests_properties(hyle_inference_request_window_test PROPERTIES
  LABELS "tooling;generation;admission" TIMEOUT 30)

add_executable(hyle_inference_request_progress_test tools/inference_request_progress_test.cc)
target_link_libraries(hyle_inference_request_progress_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_request_progress_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_request_progress_test)
hyle_apply_sanitizer(hyle_inference_request_progress_test)
set_target_properties(hyle_inference_request_progress_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_request_progress_test)
add_test(NAME hyle_inference_request_progress_test COMMAND hyle_inference_request_progress_test)
set_tests_properties(hyle_inference_request_progress_test PROPERTIES
  LABELS "tooling;generation;progress" TIMEOUT 30)

add_executable(hyle_inference_output_delivery_test tools/inference_output_delivery_test.cc)
target_link_libraries(hyle_inference_output_delivery_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_output_delivery_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_output_delivery_test)
hyle_apply_sanitizer(hyle_inference_output_delivery_test)
set_target_properties(hyle_inference_output_delivery_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_output_delivery_test)
add_test(NAME hyle_inference_output_delivery_test COMMAND hyle_inference_output_delivery_test)
set_tests_properties(hyle_inference_output_delivery_test PROPERTIES
  LABELS "tooling;generation;backpressure" TIMEOUT 30)

# Retirement observations use the existing Status implementation; do not pull
# an unselected product into a standalone header-only/util test configuration.
if(TARGET hyle_error)
  add_executable(hyle_inference_session_cache_test tools/inference_session_cache_test.cc)
  target_link_libraries(hyle_inference_session_cache_test PRIVATE hyle_build_contract hyle_error)
  target_compile_features(hyle_inference_session_cache_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_session_cache_test)
  hyle_apply_sanitizer(hyle_inference_session_cache_test)
  set_target_properties(hyle_inference_session_cache_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_session_cache_test)
  add_test(NAME hyle_inference_session_cache_test COMMAND hyle_inference_session_cache_test)
  set_tests_properties(hyle_inference_session_cache_test PROPERTIES
    LABELS "tooling;generation;cache;retirement" TIMEOUT 30)
endif()

# Engine-owned policy checks use the real host storage primitives only when
# those factors are already selected; a util-only test build stays independent.
if(TARGET hyle_compute_batch)
  add_executable(hyle_qwen_prefix_cache_test tools/qwen_prefix_cache_test.cc)
  target_link_libraries(hyle_qwen_prefix_cache_test PRIVATE hyle_compute_batch)
  target_compile_definitions(hyle_qwen_prefix_cache_test PRIVATE HYLE_PREFIX_CACHE_STANDALONE)
  target_compile_features(hyle_qwen_prefix_cache_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_qwen_prefix_cache_test)
  hyle_apply_sanitizer(hyle_qwen_prefix_cache_test)
  set_target_properties(hyle_qwen_prefix_cache_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_prefix_cache_test)
  add_test(NAME hyle_qwen_prefix_cache_test COMMAND hyle_qwen_prefix_cache_test)
  set_tests_properties(hyle_qwen_prefix_cache_test PROPERTIES
    LABELS "tooling;generation;cache" TIMEOUT 30)
endif()

add_executable(hyle_qwen_prepared_roles_test tools/qwen_prepared_roles_test.cc)
target_link_libraries(hyle_qwen_prepared_roles_test PRIVATE hyle_build_contract)
target_compile_features(hyle_qwen_prepared_roles_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_qwen_prepared_roles_test)
hyle_apply_sanitizer(hyle_qwen_prepared_roles_test)
set_target_properties(hyle_qwen_prepared_roles_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_qwen_prepared_roles_test)
add_test(NAME hyle_qwen_prepared_roles_test
  COMMAND hyle_qwen_prepared_roles_test
    "${CMAKE_CURRENT_BINARY_DIR}/qwen-prepared-roles/$<CONFIG>")
set_tests_properties(hyle_qwen_prepared_roles_test PROPERTIES
  LABELS "tooling;mtp;parser;quantization" TIMEOUT 30)

add_executable(hyle_qwen_precision_profile_test tools/qwen_precision_profile_test.cc)
target_link_libraries(hyle_qwen_precision_profile_test PRIVATE hyle_build_contract)
target_compile_features(hyle_qwen_precision_profile_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_qwen_precision_profile_test)
hyle_apply_sanitizer(hyle_qwen_precision_profile_test)
set_target_properties(hyle_qwen_precision_profile_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_qwen_precision_profile_test)
add_test(NAME hyle_qwen_precision_profile_test COMMAND hyle_qwen_precision_profile_test)
set_tests_properties(hyle_qwen_precision_profile_test PROPERTIES
  LABELS "tooling;generation;parser;quantization" TIMEOUT 30)

add_test(NAME hyle_qwen_gguf_roles_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tooling/qwen_gguf_roles_test.py")
set_tests_properties(hyle_qwen_gguf_roles_test PROPERTIES
  LABELS "tooling;mtp;parser;quantization" TIMEOUT 30)

add_test(NAME hyle_qwen_hf_roles_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tooling/qwen_hf_roles_test.py")
set_tests_properties(hyle_qwen_hf_roles_test PROPERTIES
  LABELS "tooling;mtp;parser;quantization" TIMEOUT 30)

if(_HYLE_COMPUTE_HAVE_CUDA AND _HYLE_COMPUTE_REQUEST_CPU
   AND TARGET hyle_graph_route_cpu_cuda AND TARGET hyle_graph_route_cpu
   AND TARGET hyle_graph_route_cuda)
  hyle_add_test(graph_route_cpu_cuda NAME qwen_backend
    SOURCES tools/qwen_backend_cpu_cuda_test.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_cuda hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_cpu_cuda_qwen_backend_test)
endif()

if(TARGET hyle_engine AND _HYLE_COMPUTE_HAVE_CUDA)
  # Actual Model/Session regression, reusing the application's SDK contract.
  # Missing local model input is a skip; an explicitly invalid input is an error.
  add_executable(hyle_qwen_request_lifecycle_test tools/qwen_request_lifecycle_test.cc)
  hyle_configure_engine_target(hyle_qwen_request_lifecycle_test)
  set_target_properties(hyle_qwen_request_lifecycle_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_request_lifecycle_test)
  add_test(NAME hyle_qwen_request_lifecycle_test COMMAND hyle_qwen_request_lifecycle_test)
  add_test(NAME hyle_qwen_chunked_prefill_test
    COMMAND hyle_qwen_request_lifecycle_test --chunked-prefill)
  add_test(NAME hyle_qwen_dual_model_test
    COMMAND hyle_qwen_request_lifecycle_test --dual-model)
  add_test(NAME hyle_qwen_sampling_policy_test
    COMMAND hyle_qwen_request_lifecycle_test --sampling-policies)
  if(TARGET hyle_graph_route_cpu)
    add_test(NAME hyle_qwen_request_ledger_test
      COMMAND hyle_qwen_request_lifecycle_test --request-ledger)
    set_tests_properties(hyle_qwen_request_ledger_test PROPERTIES
      LABELS "tooling;generation;commit;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    add_test(NAME hyle_qwen_sampling_budget_test
      COMMAND hyle_qwen_request_lifecycle_test --sampling-budget)
    set_tests_properties(hyle_qwen_sampling_budget_test PROPERTIES
      LABELS "tooling;generation;sampling;admission;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    add_test(NAME hyle_qwen_request_stop_policy_test
      COMMAND hyle_qwen_request_lifecycle_test --stop-policies)
    set_tests_properties(hyle_qwen_request_stop_policy_test PROPERTIES
      LABELS "tooling;generation;stopping;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  endif()
  add_test(NAME hyle_qwen_retirement_status_test
    COMMAND hyle_qwen_request_lifecycle_test --retirement-status-only)
  set_tests_properties(hyle_qwen_retirement_status_test PROPERTIES
    LABELS "tooling;generation;retirement" TIMEOUT 30)
  set_tests_properties(hyle_qwen_request_lifecycle_test hyle_qwen_chunked_prefill_test hyle_qwen_dual_model_test hyle_qwen_sampling_policy_test PROPERTIES
    LABELS "tooling;generation;admission;cuda;local-model" TIMEOUT 180
    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_graph_route_vulkan
   AND TARGET hyle_compute_stream_vulkan AND TARGET hyle_graph_progress)
  hyle_add_test(graph_route_vulkan NAME qwen_resource_retirement
    SOURCES tools/qwen_resource_retirement_test.cc
      "${HYLE_SOURCE_DIR}/engine/reporting/execution_report.cc"
    DEPS hyle_compute_stream_vulkan hyle_graph_progress hyle_graph_batch
      hyle_compute_provider_ggml_vulkan hyle_compute_ggml_runtime ggml)
  target_include_directories(hyle_graph_route_vulkan_qwen_resource_retirement_test SYSTEM PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_vulkan_qwen_resource_retirement_test)
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

# Cross-product correctness journeys are project-owned integration tests, not
# factor tests.  Register them only after every selected factor fragment has
# run, so helpers owned by a factor (notably the Process child fixture) already
# exist.  The subdirectory independently checks for the required public product
# facades and therefore disappears from narrow component builds without
# inventing another product option or factor owner.
add_subdirectory(correctness)
