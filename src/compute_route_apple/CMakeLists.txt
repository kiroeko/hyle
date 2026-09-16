if(HYLE_COMPUTE_APPLE_ENABLED)
  configure_file(apple_build.h.in apple_build.h @ONLY)
  hyle_add_module(compute_route_apple SOURCES
    apple_context.cc apple_operations.cc apple_environment.mm)
  target_include_directories(hyle_compute_route_apple PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute" "${CMAKE_CURRENT_BINARY_DIR}")
  set_target_properties(hyle_compute_route_apple PROPERTIES
    OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
  target_link_libraries(hyle_compute_route_apple PUBLIC mlx)
  target_link_libraries(hyle_compute_route_apple PRIVATE "-framework Foundation")
  target_compile_definitions(hyle_compute_route_apple PUBLIC HYLE_COMPUTE_APPLE=1)
  hyle_register_package_dependency(
    NAME apple_mlx OWNER compute_route_apple FEATURES compute_apple
    EXPORT_TARGETS mlx
    LOCKED_DEPENDENCIES mlx metal_cpp json fmt
    RUNTIME_FILE_SPECS
      "${HYLE_APPLE_MLX_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-LICENSE"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/LICENSE.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|metal-cpp-LICENSE"
      "${HYLE_APPLE_JSON_SOURCE_DIR}/LICENSE.MIT|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-json-LICENSE"
      "${HYLE_APPLE_FMT_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx-fmt-LICENSE"
    DEVELOPMENT_DIRECTORY_SPECS
      "${HYLE_APPLE_MLX_SOURCE_DIR}/mlx|${HYLE_INSTALL_INCLUDEDIR}"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/Foundation|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/Metal|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp"
      "${HYLE_APPLE_METAL_CPP_SOURCE_DIR}/QuartzCore|${HYLE_INSTALL_INCLUDEDIR}/metal_cpp")
else()
  hyle_add_module(compute_route_apple OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_route_apple PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
