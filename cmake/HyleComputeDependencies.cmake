include_guard(GLOBAL)

# Product-shape controls are declared independently from discovery so presets
# and embedding parents can preconfigure compute without probing a GPU SDK.
hyle_auto_feature_option(HYLE_COMPUTE_CUDA
  "Build the compute CUDA backend")
option(HYLE_COMPUTE_GGML
  "compute-05: borrow ggml operators via ggml_bridge" ON)
set(_hyle_ggml_native_default ON)
if(HYLE_INSTALL)
  set(_hyle_ggml_native_default OFF)
endif()
option(HYLE_COMPUTE_GGML_NATIVE
  "Tune ggml CPU kernels for the build CPU (not portable across CPUs)"
  ${_hyle_ggml_native_default})
unset(_hyle_ggml_native_default)
option(HYLE_COMPUTE_VULKAN
  "compute-06: self-authored Vulkan compute SPIR-V kernels" ON)

set(HYLE_COMPUTE_CUDA_ENABLED OFF)
set(HYLE_COMPUTE_VULKAN_ENABLED OFF)
set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED OFF)

function(hyle_configure_compute_dependencies)
  set(_hyle_compute_cuda_enabled OFF)
  set(_hyle_compute_vulkan_enabled OFF)
  set(_hyle_compute_shader_tools_enabled OFF)
  set(_hyle_compute_used_fetchcontent OFF)

  # CUDA has a deterministic AUTO/ON/OFF contract. The distinct *_ENABLED
  # result prevents truth-testing a string option such as "OFF".
  if(NOT HYLE_COMPUTE_CUDA STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
         CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
          "CUDA architectures for HYLE and fetched CUDA kernels")
      endif()
      if(HYLE_INSTALL AND "native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR
          "A CUDA-enabled binary SDK cannot use CMAKE_CUDA_ARCHITECTURES=native; "
          "that would bind the package to the build GPU. Set an explicit list "
          "(for example 80;86;89) or all-major, or select HYLE_COMPUTE_CUDA=OFF.")
      endif()
      enable_language(CUDA)
      set(_hyle_compute_cuda_enabled ON)
      hyle_register_package_dependency(
        NAME cuda_toolkit
        OWNER compute
        FEATURES compute_cuda
        STATIC_FIND_DEPENDENCIES CUDAToolkit)
      message(STATUS
        "HYLE: compute CUDA ${HYLE_COMPUTE_CUDA} -- enabled with Toolkit ${CUDAToolkit_VERSION}")
    elseif(HYLE_COMPUTE_CUDA STREQUAL "ON")
      message(FATAL_ERROR
        "HYLE_COMPUTE_CUDA=ON requires CUDAToolkit, but it was not found. Set "
        "CUDAToolkit_ROOT or CMAKE_TOOLCHAIN_FILE, or select HYLE_COMPUTE_CUDA=AUTO/OFF.")
    else()
      message(STATUS "HYLE: compute CUDA AUTO -- Toolkit not found, host backend only")
    endif()
  else()
    message(STATUS
      "HYLE: compute CUDA OFF -- dependency probe and CUDA language skipped")
  endif()

  # ggml is a pinned implementation kernel, not a public/runtime ABI. HYLE owns
  # outer parallelism and borrows only its operator kernels.
  if(HYLE_COMPUTE_GGML)
    set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}")
    set(_hyle_compute_used_fetchcontent ON)
    include(FetchContent)
    set(GGML_PIN_COMMIT "eced84c86f8b012c752c016f7fe789adea168e1e") # v0.15.3
    FetchContent_Declare(ggml
      GIT_REPOSITORY https://github.com/ggml-org/ggml.git
      GIT_TAG        ${GGML_PIN_COMMIT}
      GIT_SHALLOW    TRUE)
    block()
      # This hook runs in ggml's project() before language detection. Use the
      # HYLE source root, not the caller's CMAKE_SOURCE_DIR, when source-embedded.
      set(HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE
        "${CMAKE_PROJECT_ggml_INCLUDE_BEFORE}")
      set(CMAKE_PROJECT_ggml_INCLUDE_BEFORE
        "${HYLE_SOURCE_DIR}/cmake/HyleGgmlProjectPolicy.cmake")
      set(BUILD_SHARED_LIBS OFF)
      set(CMAKE_POSITION_INDEPENDENT_CODE ON)
      set(GGML_STANDALONE OFF)
      set(GGML_BUILD_TESTS OFF)
      set(GGML_BUILD_EXAMPLES OFF)
      set(GGML_BUILD_SERVER OFF)
      set(GGML_BACKEND_DL OFF)
      set(GGML_OPENMP OFF)
      set(GGML_CCACHE OFF)
      set(GGML_NATIVE ${HYLE_COMPUTE_GGML_NATIVE})
      set(GGML_CPU ON)
      set(GGML_CPU_ALL_VARIANTS OFF)
      set(GGML_ACCELERATE OFF)
      set(GGML_BLAS OFF)
      set(GGML_LLAMAFILE OFF)
      set(GGML_METAL OFF)
      set(GGML_RPC OFF)
      set(GGML_VULKAN OFF)
      set(GGML_CUDA ${_hyle_compute_cuda_enabled})
      set(GGML_CUDA_FA OFF)
      set(GGML_CUDA_NCCL OFF)

      # Portable binary SDKs use the x86-64-v2/SSE4 floor unless an exact HYLE
      # ISA package deliberately raises both HYLE and ggml together.
      if(NOT HYLE_COMPUTE_GGML_NATIVE AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(GGML_SSE42 ON)
        set(GGML_AVX OFF)
        set(GGML_AVX2 OFF)
        set(GGML_BMI2 OFF)
        set(GGML_AVX512 OFF)
        set(GGML_AVX512_VBMI OFF)
        set(GGML_AVX512_VNNI OFF)
        set(GGML_AVX512_BF16 OFF)
        if(NOT MSVC)
          set(GGML_FMA OFF)
          set(GGML_F16C OFF)
        endif()
        if(HYLE_SIMD_EXACT_TARGET STREQUAL "AVX2" OR
           HYLE_SIMD_EXACT_TARGET STREQUAL "AVX512")
          set(GGML_AVX ON)
          set(GGML_AVX2 ON)
          set(GGML_BMI2 ON)
          if(NOT MSVC)
            set(GGML_FMA ON)
            set(GGML_F16C ON)
          endif()
        endif()
        if(HYLE_SIMD_EXACT_TARGET STREQUAL "AVX512")
          set(GGML_AVX512 ON)
        endif()
      endif()
      FetchContent_MakeAvailable(ggml)
    endblock()
    FetchContent_GetProperties(ggml)
    set_property(DIRECTORY "${ggml_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    set(_hyle_ggml_export_targets "")
    foreach(_hyle_ggml_target IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      if(TARGET ${_hyle_ggml_target})
        list(APPEND _hyle_ggml_export_targets ${_hyle_ggml_target})
        string(REPLACE "-" "_" _hyle_ggml_export_name
          "${_hyle_ggml_target}")
        set_target_properties(${_hyle_ggml_target} PROPERTIES
          POSITION_INDEPENDENT_CODE ON
          EXPORT_NAME "_${_hyle_ggml_export_name}")
      endif()
    endforeach()
    unset(_hyle_ggml_export_name)
    set_property(TARGET ggml PROPERTY
      HYLE_DEP_SOURCE_DIR "${ggml_SOURCE_DIR}")
    # HYLE owns the installed static dependency closure and copies ggml's
    # reviewed include directory in HylePackage. Do not inherit upstream's
    # PUBLIC_HEADER install contract into HYLE's single install(TARGETS ...) call.
    set_property(TARGET ggml PROPERTY PUBLIC_HEADER "")
    hyle_register_package_dependency(
      NAME ggml
      OWNER compute
      FEATURES compute_ggml
      STATIC_EXPORT_TARGETS ${_hyle_ggml_export_targets}
      STATIC_FIND_DEPENDENCIES Threads
      RUNTIME_FILE_SPECS
        "${ggml_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-LICENSE"
      STATIC_DEVELOPMENT_DIRECTORY_SPECS
        "${ggml_SOURCE_DIR}/include/|${HYLE_INSTALL_INCLUDEDIR}")
    unset(_hyle_ggml_export_targets)
    message(STATUS
      "HYLE: ggml ${GGML_PIN_COMMIT} -- private PIC static CPU kernel "
      "(native=${HYLE_COMPUTE_GGML_NATIVE}, exact=${HYLE_SIMD_EXACT_TARGET}, "
      "CUDA=${_hyle_compute_cuda_enabled}, cache: ${HYLE_DEPS_DIR})")
  endif()

  # HYLE's Vulkan path owns SPIR-V kernels and uses a pinned volk/headers pair,
  # avoiding a build-time Vulkan loader import library.
  if(HYLE_COMPUTE_VULKAN)
    set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}")
    set(_hyle_compute_used_fetchcontent ON)
    include(FetchContent)
    set(VULKAN_HEADERS_PIN "b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2")
    set(VOLK_PIN           "d979819fd03de3a7606d6e23d6ff4968942599da")
    set(GLSLANG_PIN        "f0bd0257c308b9a26562c1a30c4748a0219cc951")

    set(VULKAN_HEADERS_ENABLE_TESTS OFF)
    set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
    set(VULKAN_HEADERS_ENABLE_MODULE OFF)
    FetchContent_Declare(vulkan_headers
      GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
      GIT_TAG        ${VULKAN_HEADERS_PIN}
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(vulkan_headers)

    set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}")
    set(VOLK_PULL_IN_VULKAN ON)
    set(VOLK_INSTALL OFF)
    FetchContent_Declare(volk
      GIT_REPOSITORY https://github.com/zeux/volk.git
      GIT_TAG        ${VOLK_PIN}
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(volk)
    FetchContent_GetProperties(volk)
    set_property(DIRECTORY "${vulkan_headers_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    set_property(DIRECTORY "${volk_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    set_target_properties(volk PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      EXPORT_NAME "_volk")
    set_property(TARGET volk PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${volk_SOURCE_DIR}>;$<BUILD_INTERFACE:${vulkan_headers_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/volk>")
    set_target_properties(volk PROPERTIES
      HYLE_DEP_SOURCE_DIR "${volk_SOURCE_DIR}"
      HYLE_VULKAN_HEADERS_SOURCE_DIR "${vulkan_headers_SOURCE_DIR}")
    hyle_register_package_dependency(
      NAME vulkan_loader
      OWNER compute
      FEATURES compute_vulkan
      STATIC_EXPORT_TARGETS volk
      RUNTIME_FILE_SPECS
        "${volk_SOURCE_DIR}/LICENSE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|volk-LICENSE.md"
        "${vulkan_headers_SOURCE_DIR}/LICENSE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle/Vulkan-Headers"
        "${vulkan_headers_SOURCE_DIR}/LICENSES/Apache-2.0.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle/Vulkan-Headers"
        "${vulkan_headers_SOURCE_DIR}/LICENSES/MIT.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle/Vulkan-Headers"
      STATIC_DEVELOPMENT_FILE_SPECS
        "${volk_SOURCE_DIR}/volk.h|${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/volk"
      STATIC_DEVELOPMENT_DIRECTORY_SPECS
        "${vulkan_headers_SOURCE_DIR}/include/vulkan|${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/volk"
        "${vulkan_headers_SOURCE_DIR}/include/vk_video|${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/volk"
      DEVELOPMENT_PROGRAM_SPECS
        "${HYLE_SOURCE_DIR}/tools/hyle_shader.py|${CMAKE_INSTALL_BINDIR}")

    # glslang/Python are build-time shader tools only; product sources do not
    # link them, and binary SDK configurations do not fetch them.
    if(HYLE_BUILD_TESTS OR HYLE_BUILD_BENCH)
      block()
        set(ENABLE_OPT OFF)
        set(ENABLE_HLSL OFF)
        set(GLSLANG_TESTS OFF)
        set(GLSLANG_ENABLE_INSTALL OFF)
        set(BUILD_EXTERNAL OFF)
        set(ENABLE_SPIRV ON)
        set(ENABLE_GLSLANG_BINARIES ON)
        FetchContent_Declare(glslang
          GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
          GIT_TAG        ${GLSLANG_PIN}
          GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(glslang)
      endblock()
      FetchContent_GetProperties(glslang)
      set_property(DIRECTORY "${glslang_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
      find_package(Python3 REQUIRED COMPONENTS Interpreter)
      set(_hyle_compute_shader_tools_enabled ON)
      # hyle_embed_shader() runs later from test/benchmark subdirectories.
      # FindPython's normal result variables would otherwise stop at this
      # function boundary even though the imported/tool targets are available.
      set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
    endif()

    set(_hyle_compute_vulkan_enabled ON)
    message(STATUS
      "HYLE: Vulkan compute enabled -- private PIC volk + Vulkan-Headers 1.4.341 "
      "(shader tools=${_hyle_compute_shader_tools_enabled}, cache: ${HYLE_DEPS_DIR}; "
      "loader dlopen'd at runtime)")
  endif()

  # Only product-shape results form the public configure contract. Dependency
  # implementation variables stay local to this module-owned entry point.
  set(HYLE_COMPUTE_CUDA_ENABLED "${_hyle_compute_cuda_enabled}" PARENT_SCOPE)
  set(HYLE_COMPUTE_VULKAN_ENABLED "${_hyle_compute_vulkan_enabled}" PARENT_SCOPE)
  set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED
      "${_hyle_compute_shader_tools_enabled}" PARENT_SCOPE)
  if(_hyle_compute_used_fetchcontent)
    # Preserve the cache location for subsequent HYLE subdirectories. The
    # function caller is HYLE's own directory, so an embedding parent is clean.
    set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}" PARENT_SCOPE)
  endif()
endfunction()
