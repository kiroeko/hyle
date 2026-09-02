include_guard(GLOBAL)

# This file is included only when compute belongs to the selected component
# closure. Declaration remains separate from discovery so OFF never probes.
hyle_auto_feature_option(HYLE_COMPUTE_CUDA
  "Build the compute CUDA backend" DEFAULT OFF)
option(HYLE_COMPUTE_GGML
  "compute-05: borrow ggml operators via ggml_bridge"
  OFF)
if(HYLE_COMPUTE_GGML)
  set(_hyle_ggml_native_default ON)
  if(HYLE_INSTALL)
    set(_hyle_ggml_native_default OFF)
  endif()
  option(HYLE_COMPUTE_GGML_NATIVE
    "Tune ggml CPU kernels for the build CPU (not portable across CPUs)"
    ${_hyle_ggml_native_default})
  unset(_hyle_ggml_native_default)
else()
  set(HYLE_COMPUTE_GGML_NATIVE OFF)
endif()
option(HYLE_COMPUTE_VULKAN
  "compute-06: self-authored Vulkan compute SPIR-V kernels" OFF)

set(HYLE_COMPUTE_CUDA_ENABLED OFF)
set(HYLE_COMPUTE_VULKAN_ENABLED OFF)
set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED OFF)

# enable_language() must execute in the directory scope shared by every target
# that consumes the language. Keeping it inside the dependency function makes
# detection appear to succeed, but its rule variables disappear before target
# generation (notably CMAKE_CUDA_COMPILE_OBJECT). This file is included only for
# a selected compute closure, so prepare the optional language here while OFF
# still performs no Toolkit or CUDA compiler probe.
set(_HYLE_COMPUTE_CUDA_TOOLKIT_AVAILABLE OFF)
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
    set(_HYLE_COMPUTE_CUDA_TOOLKIT_AVAILABLE ON)
  elseif(HYLE_COMPUTE_CUDA STREQUAL "ON")
    message(FATAL_ERROR
      "HYLE_COMPUTE_CUDA=ON requires CUDAToolkit, but it was not found. Set "
      "CUDAToolkit_ROOT or CMAKE_TOOLCHAIN_FILE, or select HYLE_COMPUTE_CUDA=AUTO/OFF.")
  endif()
endif()

function(hyle_configure_compute_dependencies)
  set(_hyle_compute_cuda_enabled OFF)
  set(_hyle_compute_vulkan_enabled OFF)
  set(_hyle_compute_shader_tools_enabled OFF)

  # The current compute product root exposes sched::Waker in public headers.
  # Host/GPU internals may be separable later, but no sched-free component is
  # currently declared or installable.
  if(NOT HYLE_FCONTEXT_SUPPORTED)
    message(FATAL_ERROR
      "The compute component currently requires sched/fcontext. Host/CUDA/"
      "Vulkan internals are not separately deliverable component roots.")
  endif()

  # CUDA has a deterministic AUTO/ON/OFF contract. The distinct *_ENABLED
  # result prevents truth-testing a string option such as "OFF".
  if(_HYLE_COMPUTE_CUDA_TOOLKIT_AVAILABLE)
    set(_hyle_compute_cuda_enabled ON)
    hyle_register_package_dependency(
      NAME cuda_toolkit
      OWNER compute
      FEATURES compute_cuda
      STATIC_FIND_DEPENDENCIES CUDAToolkit)
    message(STATUS
      "HYLE: compute CUDA ${HYLE_COMPUTE_CUDA} -- enabled with Toolkit ${CUDAToolkit_VERSION}")
  elseif(HYLE_COMPUTE_CUDA STREQUAL "AUTO")
    message(STATUS "HYLE: compute CUDA AUTO -- Toolkit not found, host backend only")
  else()
    message(STATUS
      "HYLE: compute CUDA OFF -- dependency probe and CUDA language skipped")
  endif()

  # ggml is a pinned implementation kernel, not a public/runtime ABI. HYLE owns
  # outer parallelism and borrows only its operator kernels.
  if(HYLE_COMPUTE_GGML)
    include(FetchContent)
    FetchContent_Declare(ggml
      URL      ${HYLE_DEP_GGML_ARCHIVE_URL}
      URL_HASH SHA256=${HYLE_DEP_GGML_ARCHIVE_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    block()
      # This hook runs in ggml's project() before language detection. Use the
      # HYLE source root, not the caller's CMAKE_SOURCE_DIR, when source-embedded.
      set(HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE
        "${CMAKE_PROJECT_ggml_INCLUDE_BEFORE}")
      set(CMAKE_PROJECT_ggml_INCLUDE_BEFORE
        "${HYLE_SOURCE_DIR}/cmake/HyleGgmlProjectPolicy.cmake")
      set(BUILD_SHARED_LIBS OFF)
      set(CMAKE_POSITION_INDEPENDENT_CODE ON)
      set(GGML_BUILD_TESTS OFF)
      set(GGML_BUILD_EXAMPLES OFF)
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
         HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
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
      hyle_fetchcontent_make_available_locked(ggml)
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
      LOCKED_DEPENDENCIES ggml
      RUNTIME_FILE_SPECS
        "${ggml_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-LICENSE"
      STATIC_DEVELOPMENT_DIRECTORY_SPECS
        "${ggml_SOURCE_DIR}/include/|${HYLE_INSTALL_INCLUDEDIR}")
    unset(_hyle_ggml_export_targets)
    message(STATUS
      "HYLE: ggml ${HYLE_DEP_GGML_VERSION} (${HYLE_DEP_GGML_COMMIT}) -- "
      "private PIC static CPU kernel "
      "(native=${HYLE_COMPUTE_GGML_NATIVE}, exact=${HYLE_SIMD_EXACT_TARGET}, "
      "CUDA=${_hyle_compute_cuda_enabled}, build state: ${HYLE_DEP_BUILD_DIR})")
  endif()

  # HYLE's Vulkan path owns SPIR-V kernels and uses a pinned volk/headers pair,
  # avoiding a build-time Vulkan loader import library.
  if(HYLE_COMPUTE_VULKAN)
    include(FetchContent)
    set(VULKAN_HEADERS_ENABLE_TESTS OFF)
    set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
    set(VULKAN_HEADERS_ENABLE_MODULE OFF)
    FetchContent_Declare(vulkan_headers
      URL      ${HYLE_DEP_VULKAN_HEADERS_ARCHIVE_URL}
      URL_HASH SHA256=${HYLE_DEP_VULKAN_HEADERS_ARCHIVE_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    hyle_fetchcontent_make_available_locked(vulkan_headers)

    set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}")
    set(VOLK_PULL_IN_VULKAN ON)
    set(VOLK_INSTALL OFF)
    FetchContent_Declare(volk
      URL      ${HYLE_DEP_VOLK_ARCHIVE_URL}
      URL_HASH SHA256=${HYLE_DEP_VOLK_ARCHIVE_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    hyle_fetchcontent_make_available_locked(volk)
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
      LOCKED_DEPENDENCIES volk vulkan_headers
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
          URL      ${HYLE_DEP_GLSLANG_ARCHIVE_URL}
          URL_HASH SHA256=${HYLE_DEP_GLSLANG_ARCHIVE_SHA256}
          DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
        hyle_fetchcontent_make_available_locked(glslang)
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
      "HYLE: Vulkan compute enabled -- private PIC volk + Vulkan-Headers "
      "${HYLE_DEP_VULKAN_SDK_VERSION} "
      "(shader tools=${_hyle_compute_shader_tools_enabled}, build state: ${HYLE_DEP_BUILD_DIR}; "
      "loader dlopen'd at runtime)")
  endif()

  # Only product-shape results form the public configure contract. Dependency
  # implementation variables stay local to this module-owned entry point.
  set(HYLE_COMPUTE_CUDA_ENABLED "${_hyle_compute_cuda_enabled}" PARENT_SCOPE)
  set(HYLE_COMPUTE_VULKAN_ENABLED "${_hyle_compute_vulkan_enabled}" PARENT_SCOPE)
  set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED
      "${_hyle_compute_shader_tools_enabled}" PARENT_SCOPE)
endfunction()
