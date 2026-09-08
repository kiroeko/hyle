include_guard(GLOBAL)

# This file is included only when a protocol factor belongs to the selected
# closure, so protocol controls do not pollute an io_core-only maintainer cache.
if(HYLE_BUILD_IO_TLS)
  hyle_auto_feature_option(HYLE_IO_TLS
    "Build the io TLS capability using OpenSSL" DEFAULT OFF)
else()
  set(HYLE_IO_TLS OFF)
endif()
if(HYLE_BUILD_IO_HTTP2)
  option(HYLE_IO_HTTP2 "Build the io HTTP/2 capability using nghttp2" OFF)
else()
  set(HYLE_IO_HTTP2 OFF)
endif()
set(HYLE_IO_TLS_ENABLED OFF)

function(hyle_configure_io_dependencies)
  # Protocol transports expose sched types and therefore require the existing
  # coroutine transport factor; unsupported fcontext targets fail closed.
  if(NOT HYLE_FCONTEXT_SUPPORTED)
    message(FATAL_ERROR
      "The io_core factor currently requires sched/fcontext. The standalone "
      "proactor is not a separately deliverable component.")
  endif()

  # AUTO preserves convenient source builds; ON makes TLS part of the requested
  # product contract; OFF is deterministic and does not even probe OpenSSL.
  set(_hyle_io_tls_enabled OFF)
  if(HYLE_BUILD_IO_TLS AND NOT HYLE_IO_TLS STREQUAL "OFF")
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND)
      set(_hyle_io_tls_enabled ON)
      message(STATUS
        "HYLE: io TLS ${HYLE_IO_TLS} -- enabled with OpenSSL ${OPENSSL_VERSION}")
    elseif(HYLE_IO_TLS STREQUAL "ON")
      message(FATAL_ERROR
        "HYLE_IO_TLS=ON requires OpenSSL, but it was not found. Set OPENSSL_ROOT_DIR "
        "or CMAKE_TOOLCHAIN_FILE, or select HYLE_IO_TLS=AUTO/OFF.")
    else()
      message(STATUS "HYLE: io TLS AUTO -- OpenSSL not found, capability disabled")
    endif()
  else()
    message(STATUS "HYLE: io TLS OFF -- dependency probe skipped")
  endif()

  # nghttp2 owns RFC 9113 framing, HPACK, stream state and flow control. HYLE
  # owns the coroutine transport and imports none of nghttp2's runtime policy.
  if(HYLE_BUILD_IO_HTTP2 AND HYLE_IO_HTTP2)
    include(FetchContent)
    FetchContent_Declare(nghttp2
      URL      ${HYLE_DEP_NGHTTP2_ARCHIVE_URL}
      URL_HASH SHA256=${HYLE_DEP_NGHTTP2_ARCHIVE_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    # nghttp2 is a pinned PRIVATE protocol kernel, not a second public/runtime
    # ABI. block() prevents its generic options from changing the parent build.
    block()
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

  set(HYLE_IO_TLS_ENABLED "${_hyle_io_tls_enabled}" PARENT_SCOPE)
endfunction()
