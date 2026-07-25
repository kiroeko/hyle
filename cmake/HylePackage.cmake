include_guard(GLOBAL)

# Install/export is deliberately centralized here. Module CMakeLists describe
# product targets; this file turns the already-resolved target graph into one
# relocatable package without reconstructing dependencies from filenames.
function(hyle_install_package)
  if(NOT HYLE_INSTALL)
    return()
  endif()

  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(_hyle_export_targets hyle_trace hyle_testing)
  set(_hyle_package_components "")
  foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
    if(TARGET hyle_${_component})
      list(APPEND _hyle_export_targets hyle_${_component})
      list(APPEND _hyle_package_components "${_component}")
    endif()
  endforeach()

  # A static archive cannot absorb another static archive. Keep mimalloc as an
  # installed private implementation target so hyle::mem remains self-contained
  # from the consumer's point of view; it is referenced through LINK_ONLY.
  if(HYLE_STATIC AND TARGET mimalloc_static)
    list(APPEND _hyle_export_targets mimalloc_static)
  endif()

  # Tracy is part of the public inline trace contract when enabled, so its one
  # client target and public header tree travel with a trace-enabled package.
  if(TARGET hyle_tracy_client)
    if(NOT EXISTS "${HYLE_TRACY_DIR}/LICENSE")
      message(FATAL_ERROR
        "The selected Tracy checkout has no LICENSE at '${HYLE_TRACY_DIR}/LICENSE'; "
        "refusing to package third-party code without its terms.")
    endif()
    list(APPEND _hyle_export_targets hyle_tracy_client)
    install(DIRECTORY "${HYLE_TRACY_DIR}/public/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hyle/vendor/tracy-public"
      COMPONENT Hyle_Development)
    install(FILES "${HYLE_TRACY_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME tracy-LICENSE
      COMPONENT Hyle_Runtime)
  endif()

  # These fetched implementation libraries do not yet expose a relocatable
  # install contract. A PRIVATE shared dependency is still a runtime dependency;
  # it does not magically become part of hyle_io/hyle_compute. Refuse every
  # package shape that would otherwise install a target with missing artifacts.
  if(HYLE_BUILD_IO AND HYLE_IO_HTTP2)
    message(FATAL_ERROR
      "An install containing io currently requires -DHYLE_IO_HTTP2=OFF; the "
      "pinned nghttp2 target is not yet included in HYLE's relocatable package.")
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_GGML)
    message(FATAL_ERROR
      "An install containing compute currently requires "
      "-DHYLE_COMPUTE_GGML=OFF; the fetched ggml runtime libraries are not yet "
      "included in HYLE's relocatable package.")
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_VULKAN_ENABLED)
    message(FATAL_ERROR
      "An install containing compute currently requires "
      "-DHYLE_COMPUTE_VULKAN=OFF; the fetched volk archive is not yet included "
      "with its license payload in HYLE's relocatable package.")
  endif()

  install(TARGETS ${_hyle_export_targets}
    EXPORT HyleTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Hyle_Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Runtime
            NAMELINK_COMPONENT Hyle_Development
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Development)

  # Headers are small and cross-reference header-only foundations outside the
  # binary component closure (for example util -> core/cache.h). Install the
  # reviewed public tree as one versioned API surface; component availability is
  # governed by imported targets, not by fragile header-copy filtering.
  install(DIRECTORY "${HYLE_SOURCE_DIR}/include/hyle"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Hyle_Development)
  install(FILES "${HYLE_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
    RENAME HYLE-LICENSE
    COMPONENT Hyle_Runtime)
  install(FILES "${HYLE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
    RENAME THIRD_PARTY-NOTICES.md
    COMPONENT Hyle_Runtime)
  # License payload follows the selected implementation graph, just like binary
  # targets. Do not make a migrated package depend on the source checkout for
  # notices belonging to code compiled into its HYLE libraries.
  if(TARGET hyle_mem)
    install(FILES "${HYLE_SOURCE_DIR}/third_party/mimalloc/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME mimalloc-LICENSE
      COMPONENT Hyle_Runtime)
  endif()
  if(TARGET hyle_container)
    install(FILES "${HYLE_SOURCE_DIR}/third_party/xxhash/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME xxhash-LICENSE
      COMPONENT Hyle_Runtime)
    install(FILES "${HYLE_SOURCE_DIR}/third_party/wyhash/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME wyhash-LICENSE
      COMPONENT Hyle_Runtime)
    install(FILES
      "${HYLE_SOURCE_DIR}/third_party/simdutf/LICENSE-MIT"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME simdutf-LICENSE-MIT
      COMPONENT Hyle_Runtime)
    install(FILES
      "${HYLE_SOURCE_DIR}/third_party/simdutf/LICENSE-APACHE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME simdutf-LICENSE-APACHE
      COMPONENT Hyle_Runtime)
  endif()
  if(TARGET hyle_simd)
    # Highway-derived kernels use the Apache-2.0 choice of Highway's dual
    # license. The standard text is already kept in-tree for simdutf as well.
    install(FILES "${HYLE_SOURCE_DIR}/third_party/simdutf/LICENSE-APACHE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME highway-Apache-2.0-LICENSE
      COMPONENT Hyle_Runtime)
    install(FILES
      "${HYLE_SOURCE_DIR}/third_party/ported/LICENSE-JPEG-XL-BSD-3-Clause"
      "${HYLE_SOURCE_DIR}/third_party/ported/LICENSE-XSIMD-BSD-3-Clause"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      COMPONENT Hyle_Runtime)
  endif()
  if(TARGET hyle_sched)
    install(FILES "${HYLE_SOURCE_DIR}/third_party/fcontext/LICENSE_1_0.txt"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME fcontext-LICENSE_1_0.txt
      COMPONENT Hyle_Runtime)
  endif()
  if(TARGET hyle_io AND HYLE_HAS_FCONTEXT)
    install(FILES "${HYLE_SOURCE_DIR}/third_party/picohttpparser/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME picohttpparser-LICENSE
      COMPONENT Hyle_Runtime)
  endif()

  set(_hyle_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Hyle")
  set(HYLE_PACKAGE_COMPONENTS "${_hyle_package_components}")
  set(HYLE_PACKAGE_STATIC "${HYLE_STATIC}")
  set(HYLE_PACKAGE_FEATURES "")
  if(HYLE_TRACE)
    list(APPEND HYLE_PACKAGE_FEATURES trace)
  endif()
  if(HYLE_BUILD_IO AND OpenSSL_FOUND)
    list(APPEND HYLE_PACKAGE_FEATURES io_tls)
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_CUDA)
    list(APPEND HYLE_PACKAGE_FEATURES compute_cuda)
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_VULKAN_ENABLED)
    list(APPEND HYLE_PACKAGE_FEATURES compute_vulkan)
  endif()
  set(HYLE_PACKAGE_NEEDS_OPENSSL OFF)
  set(HYLE_PACKAGE_NEEDS_CUDA OFF)
  if(HYLE_STATIC AND HYLE_BUILD_IO AND OpenSSL_FOUND)
    set(HYLE_PACKAGE_NEEDS_OPENSSL ON)
  endif()
  if(HYLE_STATIC AND HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_CUDA)
    set(HYLE_PACKAGE_NEEDS_CUDA ON)
  endif()

  configure_package_config_file(
    "${HYLE_SOURCE_DIR}/cmake/HyleConfig.cmake.in"
    "${HYLE_BINARY_DIR}/HyleConfig.cmake"
    INSTALL_DESTINATION "${_hyle_package_dir}")
  write_basic_package_version_file(
    "${HYLE_BINARY_DIR}/HyleConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

  install(EXPORT HyleTargets
    FILE HyleTargets.cmake
    NAMESPACE hyle::
    DESTINATION "${_hyle_package_dir}"
    COMPONENT Hyle_Development)
  install(FILES
    "${HYLE_BINARY_DIR}/HyleConfig.cmake"
    "${HYLE_BINARY_DIR}/HyleConfigVersion.cmake"
    DESTINATION "${_hyle_package_dir}"
    COMPONENT Hyle_Development)
endfunction()
