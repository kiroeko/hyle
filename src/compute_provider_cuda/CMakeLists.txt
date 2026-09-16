if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_audit_in_tree_dependencies(sglang_activation)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  hyle_add_module(compute_provider_cuda SOURCES
    cuda_provider.cc native_kernel.cu sglang_kernel.cu)
  set_target_properties(hyle_compute_provider_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_include_directories(hyle_compute_provider_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda"
    "${PROJECT_SOURCE_DIR}/third_party/sglang_activation")
  target_link_libraries(hyle_compute_provider_cuda PRIVATE CUDA::cudart)
  target_compile_options(hyle_compute_provider_cuda PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  if(NOT MSVC AND NOT HYLE_SANITIZER STREQUAL "off")
    # Sanitizer flags instrument the CUDA TU's host adapter, not device code.
    # nvcc forwards the ordinary HYLE host flags during compilation and link.
    target_compile_options(hyle_compute_provider_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
    target_link_options(hyle_compute_provider_cuda PRIVATE
      "$<$<LINK_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
  endif()
  target_compile_definitions(hyle_compute_provider_cuda PUBLIC HYLE_COMPUTE_CUDA=1)
  if(MSVC)
    target_compile_options(hyle_compute_provider_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()
  hyle_register_package_dependency(
    NAME sglang_activation
    OWNER compute_provider_cuda
    FEATURES compute_cuda
    LOCKED_DEPENDENCIES sglang_activation
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-activation-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/NOTICE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-activation-NOTICE")
else()
  hyle_add_module(compute_provider_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
