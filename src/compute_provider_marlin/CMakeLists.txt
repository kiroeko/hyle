if(HYLE_COMPUTE_CUDA_ENABLED)
  # Distributable SM80 PTX plus the qualified SM89 native image. Callers may
  # explicitly select a different architecture set for their CUDA toolchain.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  hyle_add_module(compute_provider_marlin SOURCES
    marlin_provider.cc quantized_weights.cc native_kernel.cu)
  set_target_properties(hyle_compute_provider_marlin PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_include_directories(hyle_compute_provider_marlin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda"
    "${PROJECT_SOURCE_DIR}/third_party/marlin")
  target_link_libraries(hyle_compute_provider_marlin PRIVATE CUDA::cudart)
  target_compile_options(hyle_compute_provider_marlin PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  if(MSVC)
    # CUDA-generated stubs intentionally change a kernel's storage class.
    target_compile_options(hyle_compute_provider_marlin PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()
  target_compile_definitions(hyle_compute_provider_marlin
    PUBLIC HYLE_COMPUTE_CUDA=1)
  hyle_register_package_dependency(
    NAME marlin
    OWNER compute_provider_marlin
    FEATURES compute_cuda
    LOCKED_DEPENDENCIES marlin
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/marlin/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|marlin-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/marlin/NOTICE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|marlin-NOTICE")
else()
  hyle_add_module(compute_provider_marlin OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_marlin PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
