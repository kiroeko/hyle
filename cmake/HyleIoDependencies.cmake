include_guard(GLOBAL)

# Protocol dependencies follow the selected factor closure. An io_core-only
# build never enters this file; a selected protocol is a complete capability.

function(hyle_configure_io_dependencies)
  # Protocol transports expose sched types and therefore require the existing
  # coroutine transport factor; unsupported fcontext targets fail closed.
  if(NOT HYLE_FCONTEXT_SUPPORTED)
    message(FATAL_ERROR
      "The io_core factor currently requires sched/fcontext. The standalone "
      "proactor is not a separately deliverable component.")
  endif()

  # Keep the standard SDK/root/toolchain search and fail if a selected TLS
  # factor cannot be supplied, instead of publishing a feature-only shell.
  if(HYLE_BUILD_IO_TLS)
    find_package(OpenSSL 3.0 REQUIRED)
    message(STATUS "HYLE: io TLS -- required OpenSSL ${OPENSSL_VERSION}")
  endif()

  # nghttp2 owns RFC 9113 framing, HPACK, stream state and flow control. HYLE
  # owns the coroutine transport and imports none of nghttp2's runtime policy.
  if(HYLE_BUILD_IO_HTTP2)
    include(FetchContent)
    FetchContent_Declare(nghttp2
      URL      ${HYLE_DEP_NGHTTP2_ARCHIVE_URL}
      URL_HASH SHA256=${HYLE_DEP_NGHTTP2_ARCHIVE_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    # nghttp2 is a pinned PRIVATE protocol kernel, not a second public/runtime
    # ABI. block() prevents its generic options from changing the parent build.
    block()
      set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
      set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
      set(ENABLE_LIB_ONLY ON)
      set(ENABLE_APP OFF)
      set(ENABLE_EXAMPLES OFF)
      set(ENABLE_HPACK_TOOLS OFF)
      set(ENABLE_DOC OFF)
      set(ENABLE_HTTP3 OFF)
      set(ENABLE_FAILMALLOC OFF)
      set(ENABLE_THREADS OFF)
      set(WITH_LIBXML2 OFF)
      set(WITH_JEMALLOC OFF)
      set(WITH_LIBBPF OFF)
      set(WITH_WOLFSSL OFF)
      set(WITH_MRUBY OFF)
      set(WITH_NEVERBLEED OFF)
      set(BUILD_TESTING OFF)
      set(BUILD_SHARED_LIBS OFF)
      set(BUILD_STATIC_LIBS ON)

      # nghttp2 1.70 probes application/HTTP3/documentation dependencies
      # unconditionally before processing ENABLE_LIB_ONLY.  HYLE consumes only
      # libnghttp2, so suppress those irrelevant probes inside this block.  The
      # outer HYLE TLS probe/target is intentionally unaffected by block scope.
      foreach(_hyle_nghttp2_unused_package IN ITEMS
          Python3 OpenSSL Libngtcp2 Libnghttp3 Systemd Jansson Libevent
          LibXml2 Jemalloc)
        set(CMAKE_DISABLE_FIND_PACKAGE_${_hyle_nghttp2_unused_package} TRUE)
      endforeach()
      hyle_fetchcontent_make_available_locked(nghttp2)
    endblock()
    FetchContent_GetProperties(nghttp2)
    set_property(DIRECTORY "${nghttp2_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    set_target_properties(nghttp2_static PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      EXPORT_NAME "_nghttp2_static"
      HYLE_DEP_SOURCE_DIR "${nghttp2_SOURCE_DIR}"
      HYLE_DEP_BINARY_DIR "${nghttp2_BINARY_DIR}")
    hyle_register_package_dependency(
      NAME nghttp2
      OWNER io_http2
      FEATURES io_http2
      STATIC_EXPORT_TARGETS nghttp2_static
      LOCKED_DEPENDENCIES nghttp2
      RUNTIME_FILE_SPECS
        "${nghttp2_SOURCE_DIR}/COPYING|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|nghttp2-LICENSE"
      STATIC_DEVELOPMENT_FILE_SPECS
        "${nghttp2_SOURCE_DIR}/lib/includes/nghttp2/nghttp2.h|${HYLE_INSTALL_INCLUDEDIR}/nghttp2"
        "${nghttp2_BINARY_DIR}/lib/includes/nghttp2/nghttp2ver.h|${HYLE_INSTALL_INCLUDEDIR}/nghttp2")
    message(STATUS
      "HYLE: nghttp2 ${HYLE_DEP_NGHTTP2_VERSION} "
      "(${HYLE_DEP_NGHTTP2_COMMIT}) -- io-11 HTTP/2 enabled "
      "as a private static protocol core")
  endif()
endfunction()
