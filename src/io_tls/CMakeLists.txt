set(_io_tls_sources feature.cc)
if(HYLE_IO_TLS_ENABLED)
  list(APPEND _io_tls_sources tls.cc)
endif()

hyle_add_module(io_tls SOURCES ${_io_tls_sources})

if(HYLE_IO_TLS_ENABLED)
  # OpenSSL is a private protocol/crypto mechanism. The public capability macro
  # is carried by the factor target for source, static and shared consumers.
  target_link_libraries(hyle_io_tls PRIVATE OpenSSL::SSL OpenSSL::Crypto)
  target_compile_definitions(hyle_io_tls PUBLIC HYLE_IO_TLS)
  hyle_register_package_dependency(
    NAME openssl
    OWNER io_tls
    FEATURES io_tls
    STATIC_FIND_DEPENDENCIES OpenSSL)
endif()
