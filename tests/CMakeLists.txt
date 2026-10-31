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
add_test(NAME hyle_simd_configure_generation_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/simd_configure_generation_test.py"
                 --cmake "${CMAKE_COMMAND}")
set_tests_properties(hyle_simd_configure_generation_test PROPERTIES
  LABELS "tooling;build;simd")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_HOST_WIN32 AND CMAKE_GENERATOR MATCHES "^Ninja")
  add_test(NAME hyle_ninja_headerdeps_recovery_test
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tooling/ninja_headerdeps_recovery_test.py"
      --cmake "${CMAKE_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
      --compiler "${CMAKE_CXX_COMPILER}")
  set_tests_properties(hyle_ninja_headerdeps_recovery_test PROPERTIES
    LABELS "tooling;build;dependencies")
endif()

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

if(TARGET hyle_container)
  add_executable(hyle_inference_continuation_test tools/inference_continuation_test.cc)
  target_link_libraries(hyle_inference_continuation_test PRIVATE hyle_container)
  target_compile_features(hyle_inference_continuation_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_continuation_test)
  hyle_apply_sanitizer(hyle_inference_continuation_test)
  set_target_properties(hyle_inference_continuation_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_continuation_test)
  add_test(NAME hyle_inference_continuation_test COMMAND hyle_inference_continuation_test)
  set_tests_properties(hyle_inference_continuation_test PROPERTIES
    LABELS "tooling;generation;continuation;identity" TIMEOUT 30)
endif()

if(TARGET hyle_container AND TARGET hyle_compute)
  add_executable(hyle_qwen_generation_ledger_test tools/qwen_generation_ledger_test.cc)
  target_link_libraries(hyle_qwen_generation_ledger_test PRIVATE hyle_container hyle_compute)
  target_compile_features(hyle_qwen_generation_ledger_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_qwen_generation_ledger_test)
  hyle_apply_sanitizer(hyle_qwen_generation_ledger_test)
  set_target_properties(hyle_qwen_generation_ledger_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_generation_ledger_test)
  add_test(NAME hyle_qwen_generation_ledger_test COMMAND hyle_qwen_generation_ledger_test)
  set_tests_properties(hyle_qwen_generation_ledger_test PROPERTIES
    LABELS "tooling;generation;state" TIMEOUT 30)
endif()

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

add_executable(hyle_inference_version_room_test tools/inference_version_room_test.cc)
target_link_libraries(hyle_inference_version_room_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_version_room_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_version_room_test)
hyle_apply_sanitizer(hyle_inference_version_room_test)
set_target_properties(hyle_inference_version_room_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_version_room_test)
add_test(NAME hyle_inference_version_room_test COMMAND hyle_inference_version_room_test)
set_tests_properties(hyle_inference_version_room_test PROPERTIES
  LABELS "tooling;generation;cache;admission" TIMEOUT 30)

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

# BatchIndices uses SmallVector's existing system allocator; do not select an
# otherwise unused memory product just to build this host-only engine check.
if(TARGET hyle_mem)
  add_executable(hyle_inference_batch_dispatch_test tools/inference_batch_dispatch_test.cc)
  target_link_libraries(hyle_inference_batch_dispatch_test PRIVATE hyle_build_contract hyle_mem)
  target_compile_features(hyle_inference_batch_dispatch_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_batch_dispatch_test)
  hyle_apply_sanitizer(hyle_inference_batch_dispatch_test)
  set_target_properties(hyle_inference_batch_dispatch_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_batch_dispatch_test)
  add_test(NAME hyle_inference_batch_dispatch_test COMMAND hyle_inference_batch_dispatch_test)
  set_tests_properties(hyle_inference_batch_dispatch_test PROPERTIES
    LABELS "tooling;generation;batch" TIMEOUT 30)
endif()

add_executable(hyle_inference_prefill_plan_test tools/inference_prefill_plan_test.cc)
target_link_libraries(hyle_inference_prefill_plan_test PRIVATE hyle_build_contract)
target_compile_features(hyle_inference_prefill_plan_test PRIVATE cxx_std_23)
hyle_apply_warnings(hyle_inference_prefill_plan_test)
hyle_apply_sanitizer(hyle_inference_prefill_plan_test)
set_target_properties(hyle_inference_prefill_plan_test PROPERTIES
  FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
add_dependencies(hyle_tests hyle_inference_prefill_plan_test)
add_test(NAME hyle_inference_prefill_plan_test COMMAND hyle_inference_prefill_plan_test)
set_tests_properties(hyle_inference_prefill_plan_test PROPERTIES
  LABELS "tooling;generation;prefill" TIMEOUT 30)

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
  add_executable(hyle_inference_request_execution_failure_test tools/inference_request_execution_failure_test.cc)
  target_link_libraries(hyle_inference_request_execution_failure_test PRIVATE hyle_build_contract hyle_error)
  target_compile_features(hyle_inference_request_execution_failure_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_request_execution_failure_test)
  hyle_apply_sanitizer(hyle_inference_request_execution_failure_test)
  set_target_properties(hyle_inference_request_execution_failure_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_request_execution_failure_test)
  add_test(NAME hyle_inference_request_execution_failure_test COMMAND hyle_inference_request_execution_failure_test)
  set_tests_properties(hyle_inference_request_execution_failure_test PROPERTIES
    LABELS "tooling;generation;failure;identity" TIMEOUT 30)

  add_executable(hyle_inference_request_loop_test tools/inference_request_loop_test.cc)
  target_link_libraries(hyle_inference_request_loop_test PRIVATE hyle_build_contract hyle_error)
  target_compile_features(hyle_inference_request_loop_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_request_loop_test)
  hyle_apply_sanitizer(hyle_inference_request_loop_test)
  set_target_properties(hyle_inference_request_loop_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_request_loop_test)
  add_test(NAME hyle_inference_request_loop_test COMMAND hyle_inference_request_loop_test)
  set_tests_properties(hyle_inference_request_loop_test PROPERTIES
    LABELS "tooling;generation;admission;retirement" TIMEOUT 30)

  add_executable(hyle_inference_step_execution_test tools/inference_step_execution_test.cc)
  target_link_libraries(hyle_inference_step_execution_test PRIVATE hyle_build_contract hyle_error)
  target_compile_features(hyle_inference_step_execution_test PRIVATE cxx_std_23)
  hyle_apply_warnings(hyle_inference_step_execution_test)
  hyle_apply_sanitizer(hyle_inference_step_execution_test)
  set_target_properties(hyle_inference_step_execution_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_step_execution_test)
  add_test(NAME hyle_inference_step_execution_test COMMAND hyle_inference_step_execution_test)
  set_tests_properties(hyle_inference_step_execution_test PROPERTIES
    LABELS "tooling;generation;execution;retirement" TIMEOUT 30)

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

# Only selected Qwen consumers depend on the model-owned cold inventory.
# The Vulkan-only consumer can be selected without the application target.
if(TARGET hyle_engine OR
   (_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_graph_route_vulkan
    AND TARGET hyle_compute_stream_vulkan AND TARGET hyle_graph_progress))
  include("${HYLE_SOURCE_DIR}/engine/models/qwen35/cold_sources.cmake")
endif()

if(TARGET hyle_engine)
  add_executable(hyle_inference_output_pool_capacity_test tools/inference_output_pool_capacity_test.cc)
  hyle_configure_engine_target(hyle_inference_output_pool_capacity_test)
  set_target_properties(hyle_inference_output_pool_capacity_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_output_pool_capacity_test)
  add_test(NAME hyle_inference_output_pool_capacity_test COMMAND hyle_inference_output_pool_capacity_test)
  set_tests_properties(hyle_inference_output_pool_capacity_test PROPERTIES
    LABELS "tooling;inference;capacity;cpu" TIMEOUT 60)
endif()

if(TARGET hyle_engine)
  add_executable(hyle_inference_state_transfer_test tools/inference_state_transfer_test.cc)
  hyle_configure_engine_target(hyle_inference_state_transfer_test)
  set_target_properties(hyle_inference_state_transfer_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_inference_state_transfer_test)
  foreach(_transfer_backend IN ITEMS cpu cuda apple vulkan)
    if(TARGET hyle_graph_route_${_transfer_backend})
      add_test(NAME hyle_inference_state_transfer_${_transfer_backend}_test
        COMMAND hyle_inference_state_transfer_test --${_transfer_backend})
      set_tests_properties(hyle_inference_state_transfer_${_transfer_backend}_test PROPERTIES
        LABELS "tooling;generation;cache;transfer;${_transfer_backend}"
        TIMEOUT 60 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endif()
  endforeach()
  unset(_transfer_backend)
endif()

if(TARGET hyle_engine)
  # The complete-state consumer is available on CPU-only builds as well.
  add_executable(hyle_qwen_host_image_test tools/qwen_host_image_test.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_host_image_test)
  # Private measurement only; production engine sources compile without hooks.
  target_compile_definitions(hyle_qwen_host_image_test PRIVATE HYLE_TEST_INFERENCE_WARM_COST=1)
  if(MSVC)
    # The complete model and its finite cases exceed the ordinary COFF limit.
    target_compile_options(hyle_qwen_host_image_test PRIVATE /bigobj)
  endif()
  set_target_properties(hyle_qwen_host_image_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_host_image_test)
  foreach(_host_image_backend IN ITEMS cpu cuda apple vulkan)
    if(TARGET hyle_graph_route_${_host_image_backend})
      add_test(NAME hyle_qwen_host_image_${_host_image_backend}_test
        COMMAND hyle_qwen_host_image_test --${_host_image_backend})
      add_test(NAME hyle_qwen_host_image_${_host_image_backend}_lag1_test
        COMMAND hyle_qwen_host_image_test --${_host_image_backend} --mtp-lag1)
      set_tests_properties(hyle_qwen_host_image_${_host_image_backend}_test
        hyle_qwen_host_image_${_host_image_backend}_lag1_test PROPERTIES
        LABELS "tooling;generation;cache;transfer;${_host_image_backend};local-model"
        TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
      if(_host_image_backend STREQUAL "cpu" OR _host_image_backend STREQUAL "cuda")
        add_test(NAME hyle_inference_readback_batch_${_host_image_backend}_test
          COMMAND hyle_qwen_host_image_test --${_host_image_backend} --readback-batch)
        set_tests_properties(hyle_inference_readback_batch_${_host_image_backend}_test PROPERTIES
          LABELS "tooling;generation;transfer;${_host_image_backend}"
          TIMEOUT 60 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        add_test(NAME hyle_qwen_input_packet_${_host_image_backend}_test
          COMMAND hyle_qwen_host_image_test --${_host_image_backend} --input-packet)
        set_tests_properties(hyle_qwen_input_packet_${_host_image_backend}_test PROPERTIES
          LABELS "tooling;generation;cache;${_host_image_backend};local-model"
          TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        add_test(NAME hyle_qwen_cold_shape_fifo_${_host_image_backend}_test
          COMMAND hyle_qwen_host_image_test --${_host_image_backend} --cold-shape-fifo)
        set_tests_properties(hyle_qwen_cold_shape_fifo_${_host_image_backend}_test PROPERTIES
          LABELS "tooling;generation;cache;${_host_image_backend};local-model"
          TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
        add_test(NAME hyle_qwen_warm_cost_${_host_image_backend}_test
          COMMAND hyle_qwen_host_image_test --${_host_image_backend} --warm-cost)
        set_tests_properties(hyle_qwen_warm_cost_${_host_image_backend}_test PROPERTIES
          LABELS "tooling;generation;cache;${_host_image_backend};local-model"
          TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
      endif()
    endif()
  endforeach()
  unset(_host_image_backend)
endif()

if(TARGET hyle_engine)
  add_executable(hyle_qwen_mtp_ahead_test tools/qwen_mtp_ahead_test.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_mtp_ahead_test)
  set_target_properties(hyle_qwen_mtp_ahead_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_mtp_ahead_test)
  add_test(NAME hyle_qwen_mtp_ahead_test COMMAND hyle_qwen_mtp_ahead_test)
  set_tests_properties(hyle_qwen_mtp_ahead_test PROPERTIES
    LABELS "tooling;generation;cache;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

  add_executable(hyle_qwen_generation_batch_facts_probe tools/qwen_generation_batch_facts_probe.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_generation_batch_facts_probe)
  set_target_properties(hyle_qwen_generation_batch_facts_probe PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_generation_batch_facts_probe)
  add_test(NAME hyle_qwen_generation_batch_facts_probe COMMAND hyle_qwen_generation_batch_facts_probe)
  set_tests_properties(hyle_qwen_generation_batch_facts_probe PROPERTIES
    LABELS "tooling;generation;cache;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
endif()

if(TARGET hyle_engine AND _HYLE_COMPUTE_HAVE_CUDA)
  # Explicit real-weight diagnostic: callers supply INPUT_DIR and OUTPUT_DIR.
  # Keep its test oracle out of testsOFF and standalone engine source closures.
  add_executable(hyle_kimi_kda_mla_probe EXCLUDE_FROM_ALL tools/kimi_kda_mla_probe.cc)
  target_link_libraries(hyle_kimi_kda_mla_probe PRIVATE hyle_kimi_k3_component)
  hyle_apply_warnings(hyle_kimi_kda_mla_probe)
  hyle_apply_sanitizer(hyle_kimi_kda_mla_probe)
  set_target_properties(hyle_kimi_kda_mla_probe PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  hyle_stage_runtime(hyle_kimi_kda_mla_probe)

  hyle_add_test(graph_route_cuda NAME inference_submission_views
    SOURCES tools/inference_submission_views_test.cc
    DEPS hyle_graph_batch hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_inference_submission_views_test)
  set_tests_properties(hyle_graph_route_cuda_inference_submission_views_test PROPERTIES
    LABELS "tooling;inference;submission;cuda" TIMEOUT 60 RUN_SERIAL TRUE)

  add_executable(hyle_qwen_generation_batch_relay_probe tools/qwen_generation_batch_relay_probe.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_generation_batch_relay_probe)
  set_target_properties(hyle_qwen_generation_batch_relay_probe PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_generation_batch_relay_probe)
  add_test(NAME hyle_qwen_generation_batch_relay_probe COMMAND hyle_qwen_generation_batch_relay_probe)
  set_tests_properties(hyle_qwen_generation_batch_relay_probe PROPERTIES
    LABELS "tooling;generation;cache;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  add_test(NAME hyle_qwen_continuous_generation_probe
    COMMAND hyle_qwen_generation_batch_relay_probe --cuda-continuous)
  set_tests_properties(hyle_qwen_continuous_generation_probe PROPERTIES
    LABELS "tooling;generation;cache;cuda;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

  add_executable(hyle_qwen_generation_relay_probe tools/qwen_generation_relay_probe.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_generation_relay_probe)
  set_target_properties(hyle_qwen_generation_relay_probe PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_generation_relay_probe)
  add_test(NAME hyle_qwen_generation_relay_probe COMMAND hyle_qwen_generation_relay_probe)
  set_tests_properties(hyle_qwen_generation_relay_probe PROPERTIES
    LABELS "tooling;generation;cache;cuda;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  add_executable(hyle_qwen_submission_source_probe tools/qwen_submission_source_probe.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_submission_source_probe)
  set_target_properties(hyle_qwen_submission_source_probe PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_submission_source_probe)
  add_test(NAME hyle_qwen_submission_source_probe COMMAND hyle_qwen_submission_source_probe)
  set_tests_properties(hyle_qwen_submission_source_probe PROPERTIES
    LABELS "tooling;generation;cache;cuda;local-model" TIMEOUT 180 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  add_executable(hyle_qwen_cpu_ahead_test tools/qwen_cpu_ahead_test.cc
    ${_hyle_qwen35_cold_sources})
  hyle_configure_engine_target(hyle_qwen_cpu_ahead_test)
  set_target_properties(hyle_qwen_cpu_ahead_test PROPERTIES
    FOLDER "hyle/tests/tools" HYLE_TEST_MODULE "project")
  add_dependencies(hyle_tests hyle_qwen_cpu_ahead_test)
  add_test(NAME hyle_qwen_cpu_ahead_test COMMAND hyle_qwen_cpu_ahead_test)
  set_tests_properties(hyle_qwen_cpu_ahead_test PROPERTIES
    LABELS "tooling;generation;cuda;local-model" TIMEOUT 120 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  # Actual Model/Session regression, reusing the application's SDK contract.
  # Missing local model input is a skip; an explicitly invalid input is an error.
  add_executable(hyle_qwen_request_lifecycle_test tools/qwen_request_lifecycle_test.cc
    ${_hyle_qwen35_cold_sources})
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
    add_test(NAME hyle_qwen_request_loop_test
      COMMAND hyle_qwen_request_lifecycle_test --request-loop)
    set_tests_properties(hyle_qwen_request_loop_test PROPERTIES
      LABELS "tooling;generation;admission;retirement;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    add_test(NAME hyle_qwen_request_control_test
      COMMAND hyle_qwen_request_lifecycle_test --request-control)
    set_tests_properties(hyle_qwen_request_control_test PROPERTIES
      LABELS "tooling;generation;cancel;delivery;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    add_test(NAME hyle_qwen_e4_cache_test
      COMMAND hyle_qwen_request_lifecycle_test --e4-cache)
    set_tests_properties(hyle_qwen_e4_cache_test PROPERTIES
      LABELS "tooling;generation;cache;quantization;cpu;local-model" TIMEOUT 120
      SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
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
    add_test(NAME hyle_qwen_host_image_request_test
      COMMAND hyle_qwen_request_lifecycle_test --host-image-request)
    set_tests_properties(hyle_qwen_host_image_request_test PROPERTIES
      LABELS "tooling;generation;cache;delivery;retirement;cpu;local-model" TIMEOUT 120
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
      ${_hyle_qwen35_cold_sources}
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
