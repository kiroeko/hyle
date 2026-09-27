if(HYLE_COMPUTE_APPLE_ENABLED)
  configure_file(apple_build.h.in apple_build.h @ONLY)
  hyle_add_module(compute_route_apple TRACE SOURCES
    apple_context.cc apple_operations.cc apple_tensor.cc
    apple_environment.mm apple_recurrent.cc apple_mature_indexing.cc
    apple_mature_reduction.cc apple_mature_float.cc)
  target_include_directories(hyle_compute_route_apple PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute" "${CMAKE_CURRENT_BINARY_DIR}")
  set_target_properties(hyle_compute_route_apple PROPERTIES
    OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
  target_link_libraries(hyle_compute_route_apple PUBLIC mlx)
  target_link_libraries(hyle_compute_route_apple PRIVATE "-framework Foundation")
  target_compile_definitions(hyle_compute_route_apple PUBLIC HYLE_COMPUTE_APPLE=1)
  include("${HYLE_SOURCE_DIR}/cmake/HyleAppleRuntime.cmake")
  get_target_property(_mlx_binary mlx HYLE_DEP_BINARY_DIR)
  set(_metallib "${_mlx_binary}/mlx/backend/metal/kernels/mlx.metallib")
  set_property(TARGET mlx PROPERTY HYLE_APPLE_METALLIB "${_metallib}")
  if(NOT HYLE_STATIC)
    add_custom_target(hyle_apple_native_resource
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_metallib}" "$<TARGET_FILE_DIR:mlx>/mlx.metallib"
      DEPENDS "${_metallib}"
      VERBATIM)
    add_dependencies(hyle_apple_native_resource mlx)
    add_dependencies(hyle_compute_route_apple hyle_apple_native_resource)
  endif()
  # Match the upstream C++ header installation, rather than shipping vendor
  # implementation/shader sources as part of a headers + libraries SDK.
  file(GLOB_RECURSE _mlx_headers RELATIVE "${HYLE_APPLE_MLX_SOURCE_DIR}"
    "${HYLE_APPLE_MLX_SOURCE_DIR}/mlx/*.h")
  list(REMOVE_ITEM _mlx_headers "mlx/backend/metal/kernels.h")
  set(_header_specs "")
  foreach(_header IN LISTS _mlx_headers)
    get_filename_component(_directory "${_header}" DIRECTORY)
    list(APPEND _header_specs
      "${HYLE_APPLE_MLX_SOURCE_DIR}/${_header}|${HYLE_INSTALL_INCLUDEDIR}/${_directory}")
  endforeach()
  hyle_register_package_dependency(
    NAME apple_mlx OWNER compute_route_apple FEATURES compute_apple
    EXPORT_TARGETS mlx
    LOCKED_DEPENDENCIES mlx mlx_cast_patches metal_cpp json fmt
    GENERATED_RUNTIME_FILE_SPECS
      "mlx|${_metallib}|${CMAKE_INSTALL_LIBDIR}"
    RUNTIME_FILE_SPECS
      "${HYLE_APPLE_MLX_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-LICENSE"
      "${HYLE_SOURCE_DIR}/third_party/mlx_cast/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-cast-LICENSE"
      "${HYLE_SOURCE_DIR}/third_party/mlx_cast/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-cast-NOTICE.md"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/LICENSE.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|metal-cpp-LICENSE"
      "${HYLE_APPLE_JSON_SOURCE_DIR}/LICENSE.MIT|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-json-LICENSE"
      "${HYLE_APPLE_FMT_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-fmt-LICENSE"
    DEVELOPMENT_FILE_SPECS ${_header_specs}
      "${HYLE_SOURCE_DIR}/cmake/HyleAppleRuntime.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/Hyle"
    DEVELOPMENT_DIRECTORY_SPECS
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/Foundation|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/Metal|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/QuartzCore|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp")
else()
  hyle_add_module(compute_route_apple OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_route_apple PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
