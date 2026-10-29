hyle_add_module(io_tls SOURCES feature.cc tls.cc)

# OpenSSL is a private protocol/crypto mechanism. The public capability macro
# records the selected factor for source, static and shared consumers.
target_link_libraries(hyle_io_tls PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_compile_definitions(hyle_io_tls PUBLIC HYLE_IO_TLS)
include("${CMAKE_CURRENT_SOURCE_DIR}/openssl_runtime.cmake")
set(_hyle_openssl_install FALSE)
if("io_tls" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
  set(_hyle_openssl_install TRUE)
endif()
hyle_register_openssl_runtime(_hyle_openssl_bundle "${_hyle_openssl_install}")
set(_hyle_openssl_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/HyleOpenSSLRuntime")
file(RELATIVE_PATH HYLE_OPENSSL_PREFIX_FROM_CONFIG "/${_hyle_openssl_package_dir}" "/")
set(_hyle_openssl_config "${CMAKE_CURRENT_BINARY_DIR}/HyleOpenSSLRuntimeConfig.cmake")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/HyleOpenSSLRuntimeConfig.cmake.in"
  "${_hyle_openssl_config}" @ONLY)
set(_hyle_openssl_manifest "${CMAKE_CURRENT_BINARY_DIR}/HyleOpenSSLFiles-$<CONFIG>.cmake")
file(GENERATE OUTPUT "${_hyle_openssl_manifest}"
  CONTENT "set(_hyle_openssl_files [==[${_hyle_openssl_bundle}]==])\n")
if(HYLE_INSTALL AND _hyle_openssl_install)
  install(FILES "${_hyle_openssl_manifest}"
    DESTINATION "${_hyle_openssl_package_dir}"
    RENAME HyleOpenSSLFiles.cmake COMPONENT Hyle_Development)
endif()
hyle_register_package_dependency(
  NAME openssl
  OWNER io_tls
  FEATURES io_tls
  FIND_DEPENDENCIES HyleOpenSSLRuntime
  DEVELOPMENT_FILE_SPECS
    "${_hyle_openssl_config}|${_hyle_openssl_package_dir}"
    "${CMAKE_CURRENT_SOURCE_DIR}/openssl_runtime.cmake|${_hyle_openssl_package_dir}")
