if(NOT _HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_module(compute_provider_torch_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_torch_cuda PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
include("${HYLE_SOURCE_DIR}/src/compute_provider_native/torch_cpu_fetch.cmake")
hyle_resolve_torch_cpu_sdk("${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json" _cpu_root)
_hyle_prepare_torch_cpu_sdk("${_cpu_root}")
# CUDA-only products still redistribute this common CPU dispatch runtime.
if("compute_provider_torch_cuda" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
  hyle_install_torch_sdk_notices("${HYLE_SOURCE_DIR}" "${_cpu_root}")
endif()
if(NOT TARGET torch OR NOT TARGET torch_cpu OR NOT TARGET c10)
  find_package(Torch 2.8.0 EXACT CONFIG REQUIRED
    PATHS "${_cpu_root}/share/cmake/Torch" NO_DEFAULT_PATH)
  include("${HYLE_SOURCE_DIR}/src/compute_provider_native/HyleTorchCpuSdkConfig.cmake")
  _hyle_promote_torch_cpu_targets()
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/torch_cuda_fetch.cmake")
hyle_resolve_torch_cuda_leaf("${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json" _cuda_root)
hyle_import_torch_cuda_leaf("${_cuda_root}" "${_cpu_root}" GLOBAL)
hyle_audit_in_tree_dependencies(bitsandbytes_cuda awq_cuda compressed_tensors_cuda)
hyle_add_module(compute_provider_torch_cuda SOURCES artifact_provider.cc)
hyle_target_compute_capability(hyle_compute_provider_torch_cuda torch_cuda)
target_include_directories(hyle_compute_provider_torch_cuda PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute" "${HYLE_SOURCE_DIR}/src/compute_backend_cuda"
  "${HYLE_SOURCE_DIR}/third_party/bitsandbytes_cuda"
  "${HYLE_SOURCE_DIR}/third_party/awq_cuda"
  "${HYLE_SOURCE_DIR}/third_party/compressed_tensors_cuda")
target_include_directories(hyle_compute_provider_torch_cuda SYSTEM PRIVATE
  "${_cpu_root}/include" "${_cpu_root}/include/torch/csrc/api/include" "${_cuda_root}/include")
target_link_libraries(hyle_compute_provider_torch_cuda PRIVATE
  torch torch_cpu c10 hyle_vendor_torch_cuda hyle_vendor_c10_cuda CUDA::cudart CUDA::cublas)
if(TORCH_CXX_FLAGS)
  separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(hyle_compute_provider_torch_cuda PRIVATE ${_torch_flags})
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Installed consumers' explicit SDK RPATH covers the private original leaf
  # and unchanged CPU common runtime transitively, as for the CPU adapter.
  target_link_options(hyle_compute_provider_torch_cuda PRIVATE LINKER:--disable-new-dtags)
endif()
get_filename_component(_cuda_sdk_directory "${_cuda_root}" DIRECTORY)
file(GLOB _cuda_licenses "${_cuda_sdk_directory}/*.dist-info")
if(HYLE_INSTALL AND "compute_provider_torch_cuda" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
  foreach(_component IN ITEMS Hyle_Runtime Hyle_Development)
    foreach(_directory IN LISTS _cuda_licenses)
      install(DIRECTORY "${_directory}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle/torch-cuda-sdk"
        COMPONENT "${_component}")
    endforeach()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      # This original wheel locates its notice beside the runtime, not dist-info.
      install(FILES "${_cuda_sdk_directory}/nvidia/cusparselt/LICENSE.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle/torch-cuda-sdk"
        RENAME cusparselt-LICENSE.txt COMPONENT "${_component}")
    endif()
  endforeach()
endif()
hyle_register_package_dependency(NAME torch_cuda_sdk OWNER compute_provider_torch_cuda
  FIND_DEPENDENCIES HyleTorchCudaSdk
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_SOURCE_DIR}/HyleTorchCudaSdkConfig.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|HyleTorchCudaSdkConfig.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_cuda_fetch.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|torch_cuda_fetch.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_cuda_linux.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|torch_cuda_linux.cmake"
    "${HYLE_SOURCE_DIR}/src/compute_provider_native/torch_cpu_fetch.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|torch_cpu_fetch.cmake"
    "${HYLE_SOURCE_DIR}/src/compute_provider_native/HyleTorchCpuSdkConfig.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|HyleTorchCpuSdkConfig.cmake"
    "${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCudaSdk|HyleDependencyLock.json")
foreach(_recipe IN ITEMS bitsandbytes_cuda awq_cuda compressed_tensors_cuda)
  hyle_register_package_dependency(NAME ${_recipe} OWNER compute_provider_torch_cuda
    LOCKED_DEPENDENCIES ${_recipe} RUNTIME_FILE_SPECS
    "${HYLE_SOURCE_DIR}/third_party/${_recipe}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|${_recipe}-LICENSE"
    "${HYLE_SOURCE_DIR}/third_party/${_recipe}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|${_recipe}-NOTICE.md")
endforeach()
