# Source preparation has an SDK-independent contract even when this optional
# provider is absent. Keep that test and its module aggregate in CPU-only builds.
hyle_ensure_module_test_target(compute_provider_ggml_vulkan)
add_test(NAME hyle_ggml_vulkan_source_contract_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract_test.py")
set_tests_properties(hyle_ggml_vulkan_source_contract_test PROPERTIES
  LABELS "compute_provider_ggml_vulkan")

if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_test(compute_provider_ggml_vulkan SOURCES metadata_test.cc region_test.cc native_test.cc work_test.cc storage_test.cc gallocr_faults.c mxfp4_qualification_test.cc)
  if(MSVC)
    # The C fault fixture includes the original UTF-8 vendor source. C++'s
    # normal build-contract encoding flag does not cover this C translation unit.
    set_source_files_properties(gallocr_faults.c PROPERTIES COMPILE_OPTIONS /utf-8)
  endif()
  target_include_directories(hyle_compute_provider_ggml_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_vulkan"
    "${HYLE_GGML_VULKAN_SOURCE_DIR}/src")
  target_link_libraries(hyle_compute_provider_ggml_vulkan_test PRIVATE
    ggml Vulkan-Headers hyle_compute_ggml_runtime hyle_mem hyle_error hyle_platform)
  hyle_stage_runtime_dependencies(hyle_compute_provider_ggml_vulkan_test)
  if(MSVC AND HYLE_TARGET_PROCESSOR STREQUAL "x86_64")
    FetchContent_GetProperties(ggml SOURCE_DIR _locked_ggml_source)
    add_test(NAME hyle_ggml_vulkan_masm_composition_test
      COMMAND "${CMAKE_COMMAND}"
        "-DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}"
        "-DHYLE_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/masm-composition"
        "-DHYLE_GGML_SOURCE=${_locked_ggml_source}"
        "-DHYLE_DEP_SOURCE_CACHE=${HYLE_DEP_SOURCE_CACHE}"
        "-DHYLE_PARENT_GENERATOR=${CMAKE_GENERATOR}"
        "-DHYLE_PARENT_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHYLE_PARENT_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHYLE_PARENT_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHYLE_PARENT_GLSLC=${Vulkan_GLSLC_EXECUTABLE}"
        "-DHYLE_PARENT_VULKAN_INCLUDE=${Vulkan_INCLUDE_DIR}"
        "-DHYLE_PARENT_VULKAN_LIBRARY=${Vulkan_LIBRARY}"
        "-DHYLE_PARENT_SPIRV_HEADERS=${SPIRV-Headers_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/masm_composition_test.cmake")
    set_tests_properties(hyle_ggml_vulkan_masm_composition_test PROPERTIES
      LABELS "compute_provider_ggml_vulkan" TIMEOUT 180)
  endif()
endif()
