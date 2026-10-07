if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(compute_route_cuda SOURCES cuda_context.cc)
  get_target_property(_torch_features hyle_compute_provider_torch_cuda INTERFACE_COMPILE_DEFINITIONS)
  if(_torch_features)
    # Ordinary Direct/Graph consumers see the actual assembled capability.
    # Keep the installed value literal; no build-tree target leaks into it.
    target_compile_definitions(hyle_compute_route_cuda PUBLIC ${_torch_features})
  endif()
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
