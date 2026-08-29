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

# Visual Studio multi-platform generators leave CMAKE_SYSTEM_PROCESSOR at the
# host architecture (for example AMD64 while `-A ARM64` selects an ARM64
# compiler). Derive one project-owned target identity after compiler detection
# and use it for every HYLE source/ISA/component decision.
if(MSVC_CXX_ARCHITECTURE_ID)
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
