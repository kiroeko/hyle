cmake_minimum_required(VERSION 3.25)
project(hyle_target_contracts LANGUAGES CXX)

# Configure-only checks. The host compiler is detected normally; no libraries
# or SDKs are built. Inspect target usage requirements for other platforms too.
get_filename_component(HYLE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${HYLE_SOURCE_DIR}/cmake/HyleComputeCapabilities.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleSanitizers.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cc" "int probe() { return 1; }\n")

add_library(capability INTERFACE)
hyle_target_compute_capability(capability cuda VISIBILITY INTERFACE)
hyle_target_compute_capability(capability ggml_cpu NO_DEFINE)
get_target_property(_capabilities capability HYLE_COMPUTE_CAPABILITIES)
get_target_property(_defines capability INTERFACE_COMPILE_DEFINITIONS)
if(NOT _capabilities STREQUAL "cuda;ggml_cpu" OR
   NOT _defines STREQUAL "HYLE_COMPUTE_CUDA=1")
  message(FATAL_ERROR "Compute metadata and consumer defines disagree")
endif()

# An uninstrumented application linking a sanitized archive still needs the
# sanitizer runtime. Compile instrumentation must not spread into its sources.
set(MSVC FALSE)
foreach(HYLE_SANITIZER IN ITEMS asan tsan ubsan)
  add_library(static_${HYLE_SANITIZER} STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
  hyle_apply_sanitizer(static_${HYLE_SANITIZER})
  get_target_property(_runtime static_${HYLE_SANITIZER} INTERFACE_LINK_OPTIONS)
  get_target_property(_consumer_compile static_${HYLE_SANITIZER} INTERFACE_COMPILE_OPTIONS)
  if(NOT _runtime MATCHES "^-fsanitize=" OR _consumer_compile)
    message(FATAL_ERROR "Static ${HYLE_SANITIZER} loses runtime or exports instrumentation")
  endif()
  add_library(shared_${HYLE_SANITIZER} SHARED EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
  hyle_apply_sanitizer(shared_${HYLE_SANITIZER})
  get_target_property(_consumer_link shared_${HYLE_SANITIZER} INTERFACE_LINK_OPTIONS)
  if(_consumer_link)
    message(FATAL_ERROR "Shared ${HYLE_SANITIZER} leaked its private runtime flags")
  endif()
endforeach()

# Staging may add host SDK paths for local execution; it must preserve the
# application's independently chosen deployment RPATH on both Unix platforms.
foreach(_system IN ITEMS Linux Darwin)
  set(CMAKE_SYSTEM_NAME "${_system}")
  add_library(runtime_${_system} INTERFACE)
  hyle_register_runtime_files(TARGETS runtime_${_system}
    FILES "${CMAKE_CURRENT_BINARY_DIR}/host-sdk/runtime.so")
  add_executable(app_${_system} EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
  target_link_libraries(app_${_system} PRIVATE runtime_${_system})
  set_property(TARGET app_${_system} PROPERTY INSTALL_RPATH "deployment-owned")
  hyle_stage_runtime(app_${_system})
  hyle_stage_runtime(app_${_system})
  get_target_property(_install_path app_${_system} INSTALL_RPATH)
  get_target_property(_build_path app_${_system} BUILD_RPATH)
  if(NOT _install_path STREQUAL "deployment-owned" OR
     NOT _build_path MATCHES "host-sdk")
    message(FATAL_ERROR "${_system} staging changed deployment or lost local SDK paths")
  endif()
endforeach()
function(_hyle_stage_apple_runtime target assets producers)
  set_property(TARGET ${target} PROPERTY APPLE_STAGING_HELPER_USED TRUE)
endfunction()
add_library(apple_runtime INTERFACE)
set_property(TARGET apple_runtime PROPERTY HYLE_APPLE_METALLIB "sdk-owned.metallib")
add_executable(apple_app EXCLUDE_FROM_ALL "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
target_link_libraries(apple_app PRIVATE apple_runtime)
hyle_stage_runtime(apple_app)
get_target_property(_apple_helper apple_app APPLE_STAGING_HELPER_USED)
if(NOT _apple_helper)
  message(FATAL_ERROR "Common runtime staging replaced the SDK's Apple helper")
endif()
message(STATUS "Target contracts PASS: capabilities, sanitizer linkage, deployment paths")
