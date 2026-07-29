include_guard(GLOBAL)

# Declare the product-shape controls even when io is not selected. This keeps
# command-line presets and embedding parents able to preconfigure a later io
# build without causing any dependency discovery by itself.
hyle_auto_feature_option(HYLE_IO_TLS
  "Build the io TLS capability using OpenSSL")
option(HYLE_IO_HTTP2 "io-11: HTTP/2 via nghttp2 protocol core" ON)
set(HYLE_IO_TLS_ENABLED OFF)

function(hyle_configure_io_dependencies)
  # AUTO preserves convenient source builds; ON makes TLS part of the requested
  # product contract; OFF is deterministic and does not even probe OpenSSL.
  set(_hyle_io_tls_enabled OFF)
  if(NOT HYLE_IO_TLS STREQUAL "OFF")
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
  if(HYLE_IO_HTTP2)
    include(FetchContent)
    set(NGHTTP2_PIN_COMMIT "68cb6900fde14c77f0cd7add0e094a862960eb99") # v1.69.0
    FetchContent_Declare(nghttp2
      GIT_REPOSITORY https://github.com/nghttp2/nghttp2.git
      GIT_TAG        ${NGHTTP2_PIN_COMMIT}
      GIT_SHALLOW    TRUE)
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
      set(BUILD_TESTING OFF)
      set(BUILD_SHARED_LIBS OFF)
      set(BUILD_STATIC_LIBS ON)
      FetchContent_MakeAvailable(nghttp2)
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
      OWNER io
      FEATURES io_http2
      STATIC_EXPORT_TARGETS nghttp2_static
      RUNTIME_FILE_SPECS
        "${nghttp2_SOURCE_DIR}/COPYING|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|nghttp2-LICENSE"
      STATIC_DEVELOPMENT_FILE_SPECS
        "${nghttp2_SOURCE_DIR}/lib/includes/nghttp2/nghttp2.h|${HYLE_INSTALL_INCLUDEDIR}/nghttp2"
        "${nghttp2_BINARY_DIR}/lib/includes/nghttp2/nghttp2ver.h|${HYLE_INSTALL_INCLUDEDIR}/nghttp2")
    message(STATUS
      "HYLE: nghttp2 v1.69.0 (${NGHTTP2_PIN_COMMIT}) -- io-11 HTTP/2 enabled "
      "as a private static protocol core")
  endif()

  set(HYLE_IO_TLS_ENABLED "${_hyle_io_tls_enabled}" PARENT_SCOPE)
endfunction()
