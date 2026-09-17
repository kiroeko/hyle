if(HYLE_COMPUTE_VULKAN_ENABLED)
  configure_file(vulkan_build.h.in vulkan_build.h @ONLY)
  set(_shader_source "${HYLE_SOURCE_DIR}/third_party/ggml_vulkan_shaders/shaders")
  set(_shader_output "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  file(GLOB _shader_inputs CONFIGURE_DEPENDS "${_shader_source}/*")
  add_custom_command(
    OUTPUT "${_shader_output}/vulkan_shaders.cc" "${_shader_output}/vulkan_shaders.h"
    COMMAND "${HYLE_VULKAN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/build_shaders.py"
      --compiler "${HYLE_VULKAN_GLSLANG}" --source "${_shader_source}"
      --output "${_shader_output}"
    DEPENDS build_shaders.py ${_shader_inputs}
    VERBATIM)
  hyle_add_module(compute_route_vulkan SOURCES
    vulkan_device.cc vulkan_run.cc vulkan_transfer.cc
    vulkan_pipeline.cc vulkan_operations.cc vulkan_program.cc vulkan_recorder.cc
    "${_shader_output}/vulkan_shaders.cc")
  target_include_directories(hyle_compute_route_vulkan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute" "${CMAKE_CURRENT_BINARY_DIR}"
    "${_shader_output}")
  target_compile_definitions(hyle_compute_route_vulkan PUBLIC HYLE_COMPUTE_VULKAN=1)
  target_link_libraries(hyle_compute_route_vulkan PUBLIC Vulkan-Headers)
  hyle_register_package_dependency(
    NAME vulkan_headers OWNER compute_route_vulkan FEATURES compute_vulkan
    EXPORT_TARGETS Vulkan-Headers LOCKED_DEPENDENCIES vulkan_headers ggml_vulkan_shaders
    RUNTIME_FILE_SPECS
      "${HYLE_VULKAN_HEADERS_SOURCE_DIR}/LICENSE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|vulkan-headers-LICENSE.md"
      "${HYLE_VULKAN_HEADERS_SOURCE_DIR}/LICENSES/Apache-2.0.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|vulkan-headers-Apache-2.0.txt"
      "${HYLE_VULKAN_HEADERS_SOURCE_DIR}/LICENSES/MIT.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|vulkan-headers-MIT.txt"
      "${HYLE_SOURCE_DIR}/third_party/ggml_vulkan_shaders/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-shaders-LICENSE"
      "${HYLE_SOURCE_DIR}/third_party/ggml_vulkan_shaders/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-shaders-NOTICE.md"
    DEVELOPMENT_DIRECTORY_SPECS
      "${HYLE_VULKAN_HEADERS_SOURCE_DIR}/include/vulkan|${HYLE_INSTALL_INCLUDEDIR}"
      "${HYLE_VULKAN_HEADERS_SOURCE_DIR}/include/vk_video|${HYLE_INSTALL_INCLUDEDIR}")
else()
  hyle_add_module(compute_route_vulkan OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_route_vulkan PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
