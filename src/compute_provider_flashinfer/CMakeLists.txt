# The bounded native qualification admitted this pin on Linux. Windows uses
# the ggml CUDA attention route; CUDA-OFF never fetches these headers.
if(NOT HYLE_COMPUTE_CUDA_ENABLED OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  hyle_add_module(compute_provider_flashinfer OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_flashinfer PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 89-real)
endif()
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(flashinfer
  URL "${HYLE_DEP_FLASHINFER_ARCHIVE_URL}"
  URL_HASH "SHA256=${HYLE_DEP_FLASHINFER_ARCHIVE_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR hyle-verified-source-only)
hyle_fetchcontent_make_available_locked(flashinfer)
FetchContent_GetProperties(flashinfer)

hyle_add_module(compute_provider_flashinfer
  SOURCES flashinfer_provider.cc native_attention.cu)
set_target_properties(hyle_compute_provider_flashinfer PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
if(NOT HYLE_SANITIZER STREQUAL "off")
  target_compile_options(hyle_compute_provider_flashinfer PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
  target_link_options(hyle_compute_provider_flashinfer PRIVATE
    "$<$<LINK_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
endif()
target_include_directories(hyle_compute_provider_flashinfer PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda")
target_include_directories(hyle_compute_provider_flashinfer SYSTEM PRIVATE
  "${flashinfer_SOURCE_DIR}/include")
target_compile_definitions(hyle_compute_provider_flashinfer
  PUBLIC HYLE_COMPUTE_FLASHINFER=1
  PRIVATE FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_F16)
target_link_libraries(hyle_compute_provider_flashinfer PRIVATE CUDA::cudart)
hyle_register_package_dependency(
  NAME compute_flashinfer OWNER compute_provider_flashinfer
  FEATURES compute_cuda
  FIND_DEPENDENCIES CUDAToolkit|13.3
  LOCKED_DEPENDENCIES flashinfer
  RUNTIME_FILE_SPECS
    "${flashinfer_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|flashinfer-LICENSE")
