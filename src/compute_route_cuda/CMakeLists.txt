if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(compute_route_cuda SOURCES cuda_context.cc default_policy.cc)
  target_include_directories(hyle_compute_route_cuda PRIVATE "${HYLE_SOURCE_DIR}/src/compute")
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
