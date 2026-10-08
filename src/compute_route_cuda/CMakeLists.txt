if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(compute_route_cuda SOURCES cuda_context.cc default_policy.cc best_of.cc)
  target_include_directories(hyle_compute_route_cuda PRIVATE "${HYLE_SOURCE_DIR}/src/compute")
  # Evidence is tied to the original static C2 recipe. Normal builds need no
  # user switches; unqualified recipes keep the per-operation mature defaults.
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND
     CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "19.50.35728" AND HYLE_STATIC AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL "13.3.73" AND
     CMAKE_CUDA_ARCHITECTURES STREQUAL "89-real" AND
     HYLE_SANITIZER STREQUAL "off" AND NOT HYLE_TRACE AND
     CMAKE_CXX_FLAGS STREQUAL "/DWIN32 /D_WINDOWS /EHsc" AND
     CMAKE_CXX_FLAGS_RELEASE STREQUAL "/O2 /Ob2 /DNDEBUG" AND
     NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION AND
     NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE AND
     CMAKE_CUDA_FLAGS STREQUAL "-D_WINDOWS -Xcompiler=\" /EHsc\"" AND
     CMAKE_CUDA_FLAGS_RELEASE STREQUAL "-Xcompiler=\"-O2 -Ob2\" -DNDEBUG" AND
     HYLE_GGML_CUDA_PROVIDER_VERSION STREQUAL
       "456172ec+iq1-46cbf0e9-resource-f99368c68335266b")
    get_directory_property(_caller_options COMPILE_OPTIONS)
    get_directory_property(_caller_definitions COMPILE_DEFINITIONS)
    if(NOT _caller_options AND NOT _caller_definitions AND
       (NOT CMAKE_MSVC_RUNTIME_LIBRARY OR
        CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDLL" OR
        CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"))
    target_compile_definitions(hyle_compute_route_cuda PRIVATE
      "$<$<CONFIG:Release>:HYLE_CUDA_DEFAULT_C2_RECIPE=1>")
    endif()
    unset(_caller_options)
    unset(_caller_definitions)
  endif()
  foreach(_provider IN ITEMS torch_cuda flashinfer)
    get_target_property(_provider_features hyle_compute_provider_${_provider} INTERFACE_COMPILE_DEFINITIONS)
    if(_provider_features)
      # Ordinary Direct/Graph consumers see the actual assembled capability.
      # Keep installed values literal; absent targets do not invent features.
      target_compile_definitions(hyle_compute_route_cuda PUBLIC ${_provider_features})
    endif()
  endforeach()
  get_target_property(_torch_absent hyle_compute_provider_torch_cuda HYLE_INTERNAL_ONLY_ABSENT)
  if(HYLE_STATIC AND _torch_absent)
    get_target_property(_cuda_interface hyle_compute_route_cuda INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM _cuda_interface "$<LINK_ONLY:hyle_compute_provider_torch_cuda>")
    list(APPEND _cuda_interface "$<LINK_ONLY:$<BUILD_INTERFACE:hyle_compute_provider_torch_cuda>>")
    set_property(TARGET hyle_compute_route_cuda PROPERTY INTERFACE_LINK_LIBRARIES "${_cuda_interface}")
  endif()
  if(HYLE_STATIC AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # This pin's FlashInfer factor is a build-only absent placeholder here.
    # Static PRIVATE dependencies otherwise become installed link requirements,
    # which would demand an unqualified, intentionally unexported provider.
    get_target_property(_cuda_interface hyle_compute_route_cuda
      INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM _cuda_interface "$<LINK_ONLY:hyle_compute_provider_flashinfer>")
    list(APPEND _cuda_interface
      "$<LINK_ONLY:$<BUILD_INTERFACE:hyle_compute_provider_flashinfer>>")
    set_property(TARGET hyle_compute_route_cuda PROPERTY
      INTERFACE_LINK_LIBRARIES "${_cuda_interface}")
  endif()
else()
  hyle_add_module(compute_route_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_route_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
