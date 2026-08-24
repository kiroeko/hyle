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
