hyle_ensure_module_test_target(compute_ggml_runtime)
add_test(NAME hyle_ggml_runtime_source_contract_test
  COMMAND "${Python3_EXECUTABLE}"
    "${HYLE_SOURCE_DIR}/tests/compute_provider_ggml_vulkan/source_contract_test.py")
set_tests_properties(hyle_ggml_runtime_source_contract_test PROPERTIES
  LABELS "compute_ggml_runtime")
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_ggml_runtime NAME cuda SOURCES cuda_resource_test.cc)
  target_include_directories(hyle_compute_ggml_runtime_cuda_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_cuda"
    "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
  target_link_libraries(hyle_compute_ggml_runtime_cuda_test PRIVATE ggml CUDA::cudart)
  if(HYLE_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(hyle_compute_ggml_runtime_cuda_test PRIVATE
      HYLE_GGML_CUDA_FAULT_INJECTION=1)
    target_link_options(hyle_compute_ggml_runtime_cuda_test PRIVATE
      LINKER:--wrap=cudaStreamBeginCapture
      LINKER:--wrap=cudaStreamEndCapture
      LINKER:--wrap=cudaGetLastError)
  endif()
  set_tests_properties(hyle_compute_ggml_runtime_cuda_test PROPERTIES TIMEOUT 60)
  hyle_stage_runtime_dependencies(hyle_compute_ggml_runtime_cuda_test)
endif()
if(HYLE_GGML_CPU_ENABLED)
  hyle_add_test(compute_ggml_runtime SOURCES execution_test.cc
    ../compute_provider_ggml_vulkan/metadata_test.cc)
  target_include_directories(hyle_compute_ggml_runtime_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_vulkan"
    "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
  target_link_libraries(hyle_compute_ggml_runtime_test PRIVATE ggml hyle_mem hyle_error)
  hyle_stage_runtime_dependencies(hyle_compute_ggml_runtime_test)

  # Each executable substitutes failures only inside its exact vendor C TU.
  # They do not link the HYLE runtime, install hooks, or exhaust the machine.
  foreach(_fixture IN ITEMS cpu context)
    add_executable(hyle_ggml_${_fixture}_fault_test ${_fixture}_faults.c)
    if(_fixture STREQUAL "cpu")
      set(_vendor_tu_target ggml-cpu)
    else()
      set(_vendor_tu_target ggml-base)
    endif()
    target_compile_definitions(hyle_ggml_${_fixture}_fault_test PRIVATE
      "$<TARGET_PROPERTY:${_vendor_tu_target},COMPILE_DEFINITIONS>")
    target_compile_options(hyle_ggml_${_fixture}_fault_test PRIVATE
      "$<TARGET_PROPERTY:${_vendor_tu_target},COMPILE_OPTIONS>")
    target_include_directories(hyle_ggml_${_fixture}_fault_test PRIVATE
      "${HYLE_GGML_SOURCE_DIR}/src"
      "${HYLE_GGML_SOURCE_DIR}/src/ggml-cpu"
      "${HYLE_BINARY_DIR}/src/compute_ggml_runtime/vendor/src")
    target_link_libraries(hyle_ggml_${_fixture}_fault_test PRIVATE ggml)
    set_target_properties(hyle_ggml_${_fixture}_fault_test PROPERTIES
      C_STANDARD 11 HYLE_TEST_MODULE compute_ggml_runtime)
    hyle_apply_sanitizer(hyle_ggml_${_fixture}_fault_test)
    if(MSVC)
      target_compile_options(hyle_ggml_${_fixture}_fault_test PRIVATE /utf-8)
    else()
      target_compile_definitions(hyle_ggml_${_fixture}_fault_test PRIVATE _GNU_SOURCE)
    endif()
    add_dependencies(hyle_compute_ggml_runtime_tests hyle_ggml_${_fixture}_fault_test)
    add_test(NAME hyle_ggml_${_fixture}_fault_test
      COMMAND hyle_ggml_${_fixture}_fault_test)
    set_tests_properties(hyle_ggml_${_fixture}_fault_test PROPERTIES
      LABELS "compute_ggml_runtime" TIMEOUT 60)
  endforeach()
endif()
