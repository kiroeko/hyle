include_guard(GLOBAL)

# HYLE's public layouts and lock-free queues are a 64-bit product contract.
# Reject an otherwise successful 32-bit configure before component selection
# can expose a partially buildable SDK. Unknown 64-bit architectures remain
# eligible for the deliberately generic/scalar paths; tuned ISA capability is
# validated separately by each component.
if(NOT DEFINED CMAKE_SIZEOF_VOID_P)
  message(FATAL_ERROR
    "HYLE target pointer width is unknown; run this check after project() has "
    "identified the C/C++ target ABI")
elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  math(EXPR _hyle_target_pointer_bits "${CMAKE_SIZEOF_VOID_P} * 8")
  message(FATAL_ERROR
    "HYLE requires a 64-bit target ABI; detected "
    "${_hyle_target_pointer_bits}-bit pointers. 32-bit targets are not a "
    "supported source or binary SDK shape.")
  unset(_hyle_target_pointer_bits)
endif()

# Explicit macOS architectures take precedence over CMAKE_SYSTEM_PROCESSOR,
# which can still describe the host (for example arm64 while -arch x86_64
# selects an Intel target). HYLE currently constructs one source/ISA/dependency
# graph per build directory, so a universal binary needs separate builds and
# cannot safely share this graph. Leave an empty architecture selection alone:
# CMake/toolchain defaults remain authoritative.
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  set(_hyle_apple_architectures "${CMAKE_OSX_ARCHITECTURES}")
  list(REMOVE_DUPLICATES _hyle_apple_architectures)
  list(LENGTH _hyle_apple_architectures _hyle_apple_architecture_count)
  if(_hyle_apple_architecture_count GREATER 1)
    message(FATAL_ERROR
      "HYLE requires one target architecture per build directory; "
      "CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' selects multiple "
      "architectures. Configure a separate build directory for each architecture.")
  endif()
  list(GET _hyle_apple_architectures 0 _hyle_target_processor_raw)
  unset(_hyle_apple_architectures)
  unset(_hyle_apple_architecture_count)
# Visual Studio multi-platform generators likewise leave CMAKE_SYSTEM_PROCESSOR
# at the host architecture when -A selects a different compiler target.
elseif(MSVC_CXX_ARCHITECTURE_ID)
  set(_hyle_target_processor_raw "${MSVC_CXX_ARCHITECTURE_ID}")
elseif(CMAKE_GENERATOR_PLATFORM)
  set(_hyle_target_processor_raw "${CMAKE_GENERATOR_PLATFORM}")
else()
  set(_hyle_target_processor_raw "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(_hyle_target_processor_raw MATCHES "^(x86_64|AMD64|amd64|x64|X64)$")
  set(HYLE_TARGET_PROCESSOR "x86_64")
elseif(_hyle_target_processor_raw MATCHES "^(arm64|aarch64|ARM64|AARCH64)$")
  set(HYLE_TARGET_PROCESSOR "aarch64")
elseif(_hyle_target_processor_raw MATCHES "^(riscv64|RISCV64)$")
  set(HYLE_TARGET_PROCESSOR "riscv64")
else()
  set(HYLE_TARGET_PROCESSOR "${_hyle_target_processor_raw}")
endif()
# Source-embedding helpers can be called from the parent directory after
# add_subdirectory(hyle), where this directory-local variable is no longer in
# scope. Publish the same normalized fact globally instead of falling back to
# CMAKE_SYSTEM_PROCESSOR and reintroducing the Visual Studio host/target bug.
set_property(GLOBAL PROPERTY HYLE_TARGET_PROCESSOR
  "${HYLE_TARGET_PROCESSOR}")
unset(_hyle_target_processor_raw)
