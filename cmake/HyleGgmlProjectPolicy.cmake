# Included by CMAKE_PROJECT_ggml_INCLUDE_BEFORE, inside ggml's project() and
# before its language detection. The pinned ggml project declares generic ASM on MSVC even
# though it has no generic assembler source there. CMake 4.1's CMP0194 NEW
# behavior correctly rejects cl as an ASM compiler; retain the compatibility
# behavior only for this pinned upstream project until its declaration changes.
if(POLICY CMP0194)
  cmake_policy(SET CMP0194 OLD)
endif()

# Do not discard a hook supplied by an embedding parent for the same subproject.
foreach(_hyle_parent_hook IN LISTS HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE)
  include("${_hyle_parent_hook}")
endforeach()
unset(_hyle_parent_hook)
