if(NOT HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_module(compute_provider_ggml_vulkan OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml_vulkan PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()

# Locked vendor source and metadata storage are owned by compute_ggml_runtime.
hyle_add_module(compute_provider_ggml_vulkan
  SOURCES native_device.cc native_graph.cc region.cc)
target_link_libraries(hyle_compute_provider_ggml_vulkan PRIVATE ggml
  "$<BUILD_INTERFACE:Vulkan-Headers>")
