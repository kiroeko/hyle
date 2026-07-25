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
  # nghttp2 is always a private static protocol core. A shared hyle_io absorbs
  # it; only a static hyle_io needs the archive exported through LINK_ONLY.
  if(HYLE_STATIC AND TARGET nghttp2_static)
    list(APPEND _hyle_export_targets nghttp2_static)
  endif()
  # ggml/volk are private PIC archives. Shared hyle_compute absorbs them;
  # static hyle_compute carries their reviewed target closure through LINK_ONLY.
  if(HYLE_STATIC AND TARGET ggml)
    foreach(_hyle_ggml_target IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      if(TARGET ${_hyle_ggml_target})
        list(APPEND _hyle_export_targets ${_hyle_ggml_target})
      endif()
    endforeach()
    unset(_hyle_ggml_target)
  endif()
  if(HYLE_STATIC AND TARGET volk)
    list(APPEND _hyle_export_targets volk)
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

  # An installed shared HYLE module must be able to resolve sibling HYLE
  # modules after the whole SDK prefix is moved.  A consumer executable's
  # RUNPATH is not inherited while loading transitive ELF dependencies, so
  # relying on the consumer's link line works for direct modules but fails for
  # e.g. hyle_simd -> hyle_platform -> hyle_error.  Give each installed HYLE
  # shared object its own loader-relative lookup path.  Keep this target-local:
  # source-embedding parents retain ownership of their build/install RPATH.
  if(NOT HYLE_STATIC AND UNIX)
    if(APPLE)
      set(_hyle_sibling_install_rpath "@loader_path")
    else()
      set(_hyle_sibling_install_rpath "$ORIGIN")
    endif()
    foreach(_hyle_export_target IN LISTS _hyle_export_targets)
      get_target_property(_hyle_export_type
        ${_hyle_export_target} TYPE)
      if(_hyle_export_type STREQUAL "SHARED_LIBRARY" OR
         _hyle_export_type STREQUAL "MODULE_LIBRARY")
        set_target_properties(${_hyle_export_target} PROPERTIES
          INSTALL_RPATH "${_hyle_sibling_install_rpath}"
          INSTALL_RPATH_USE_LINK_PATH OFF)
      endif()
    endforeach()
    unset(_hyle_export_target)
    unset(_hyle_export_type)
    unset(_hyle_sibling_install_rpath)
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
  if(TARGET nghttp2_static)
    if(NOT EXISTS "${nghttp2_SOURCE_DIR}/COPYING")
      message(FATAL_ERROR
        "The selected nghttp2 checkout has no COPYING at "
        "'${nghttp2_SOURCE_DIR}/COPYING'; refusing to package it without terms.")
    endif()
    install(FILES "${nghttp2_SOURCE_DIR}/COPYING"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME nghttp2-LICENSE
      COMPONENT Hyle_Runtime)
    if(HYLE_STATIC)
      # nghttp2_static's imported target retains its reviewed include contract.
      # The static consumer reaches it only through LINK_ONLY, but installing the
      # two public headers keeps the target independently valid and relocatable.
      install(FILES
        "${nghttp2_SOURCE_DIR}/lib/includes/nghttp2/nghttp2.h"
        "${nghttp2_BINARY_DIR}/lib/includes/nghttp2/nghttp2ver.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nghttp2"
        COMPONENT Hyle_Development)
    endif()
  endif()
  if(TARGET ggml)
    if(NOT EXISTS "${ggml_SOURCE_DIR}/LICENSE")
      message(FATAL_ERROR
        "The selected ggml checkout has no LICENSE at '${ggml_SOURCE_DIR}/LICENSE'; "
        "refusing to package it without terms.")
    endif()
    install(FILES "${ggml_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME ggml-LICENSE
      COMPONENT Hyle_Runtime)
    if(HYLE_STATIC)
      install(DIRECTORY "${ggml_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT Hyle_Development)
    endif()
  endif()
  if(TARGET volk)
    foreach(_hyle_license IN ITEMS
        "${volk_SOURCE_DIR}/LICENSE.md"
        "${vulkan_headers_SOURCE_DIR}/LICENSE.md"
        "${vulkan_headers_SOURCE_DIR}/LICENSES/Apache-2.0.txt"
        "${vulkan_headers_SOURCE_DIR}/LICENSES/MIT.txt")
      if(NOT EXISTS "${_hyle_license}")
        message(FATAL_ERROR
          "The selected Vulkan dependency lacks license file '${_hyle_license}'.")
      endif()
    endforeach()
    install(FILES "${volk_SOURCE_DIR}/LICENSE.md"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME volk-LICENSE.md
      COMPONENT Hyle_Runtime)
    install(FILES
      "${vulkan_headers_SOURCE_DIR}/LICENSE.md"
      "${vulkan_headers_SOURCE_DIR}/LICENSES/Apache-2.0.txt"
      "${vulkan_headers_SOURCE_DIR}/LICENSES/MIT.txt"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle/Vulkan-Headers"
      COMPONENT Hyle_Runtime)
    if(HYLE_STATIC)
      install(FILES "${volk_SOURCE_DIR}/volk.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hyle/vendor/volk"
        COMPONENT Hyle_Development)
      install(DIRECTORY
        "${vulkan_headers_SOURCE_DIR}/include/vulkan"
        "${vulkan_headers_SOURCE_DIR}/include/vk_video"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hyle/vendor/volk"
        COMPONENT Hyle_Development)
    endif()
    install(PROGRAMS "${HYLE_SOURCE_DIR}/tools/hyle_shader.py"
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT Hyle_Development)
  endif()

  set(_hyle_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Hyle")
  set(HYLE_PACKAGE_COMPONENTS "${_hyle_package_components}")
  set(HYLE_PACKAGE_STATIC "${HYLE_STATIC}")
  set(HYLE_PACKAGE_FEATURES "")
  if(HYLE_TRACE)
    list(APPEND HYLE_PACKAGE_FEATURES trace)
  endif()
  if(HYLE_BUILD_IO AND HYLE_IO_TLS_ENABLED)
    list(APPEND HYLE_PACKAGE_FEATURES io_tls)
  endif()
  if(TARGET nghttp2_static)
    list(APPEND HYLE_PACKAGE_FEATURES io_http2)
  endif()
  if(TARGET ggml)
    list(APPEND HYLE_PACKAGE_FEATURES compute_ggml)
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_CUDA_ENABLED)
    list(APPEND HYLE_PACKAGE_FEATURES compute_cuda)
  endif()
  if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_VULKAN_ENABLED)
    list(APPEND HYLE_PACKAGE_FEATURES compute_vulkan)
  endif()
  set(HYLE_PACKAGE_NEEDS_OPENSSL OFF)
  set(HYLE_PACKAGE_NEEDS_CUDA OFF)
  set(HYLE_PACKAGE_NEEDS_THREADS OFF)
  set(HYLE_PACKAGE_NEEDS_CXX_LINKER "${HYLE_STATIC}")
  if(HYLE_STATIC AND HYLE_BUILD_IO AND HYLE_IO_TLS_ENABLED)
    set(HYLE_PACKAGE_NEEDS_OPENSSL ON)
  endif()
  if(HYLE_STATIC AND HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_CUDA_ENABLED)
    set(HYLE_PACKAGE_NEEDS_CUDA ON)
  endif()
  if(HYLE_STATIC AND TARGET ggml)
    set(HYLE_PACKAGE_NEEDS_THREADS ON)
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
