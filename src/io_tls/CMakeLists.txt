hyle_add_module(io_tls SOURCES feature.cc tls.cc)

# OpenSSL is a private protocol/crypto mechanism. The public capability macro
# records the selected factor for source, static and shared consumers.
target_link_libraries(hyle_io_tls PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_compile_definitions(hyle_io_tls PUBLIC HYLE_IO_TLS)
hyle_register_package_dependency(
  NAME openssl
  OWNER io_tls
  FEATURES io_tls
  STATIC_FIND_DEPENDENCIES OpenSSL|3.0)
