cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_TEST_ROOT HYLE_GGML_SOURCE
    HYLE_PARENT_GENERATOR HYLE_PARENT_CXX_COMPILER HYLE_PARENT_GLSLC
    HYLE_PARENT_VULKAN_INCLUDE HYLE_PARENT_VULKAN_LIBRARY)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=...")
  endif()
endforeach()
foreach(_path IN ITEMS HYLE_SOURCE_DIR HYLE_TEST_ROOT HYLE_GGML_SOURCE
    HYLE_DEP_SOURCE_CACHE HYLE_PARENT_CXX_COMPILER HYLE_PARENT_GLSLC
    HYLE_PARENT_VULKAN_INCLUDE HYLE_PARENT_VULKAN_LIBRARY)
  file(TO_CMAKE_PATH "${${_path}}" ${_path})
endforeach()

# A real parent owns MASM code and embeds HYLE's actual Vulkan product. The
# provider's original ggml project must not change the parent's source language.
set(_source "${HYLE_TEST_ROOT}/parent")
set(_build "${HYLE_TEST_ROOT}/build")
file(MAKE_DIRECTORY "${_source}")
file(WRITE "${_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_masm_vulkan_parent LANGUAGES CXX ASM_MASM)\n"
"set(_asm\n"
"  \"${HYLE_SOURCE_DIR}/third_party/fcontext/asm/jump_x86_64_ms_pe_masm.asm\"\n"
"  \"${HYLE_SOURCE_DIR}/third_party/fcontext/asm/make_x86_64_ms_pe_masm.asm\"\n"
"  \"${HYLE_SOURCE_DIR}/third_party/fcontext/asm/ontop_x86_64_ms_pe_masm.asm\")\n"
"add_library(hyle_masm_parent STATIC \${_asm})\n"
"set_target_properties(hyle_masm_parent PROPERTIES LINKER_LANGUAGE CXX)\n"
"target_compile_definitions(hyle_masm_parent PRIVATE BOOST_CONTEXT_EXPORT=PUBLIC)\n"
"set(HYLE_COMPONENTS graph)\n"
"set(HYLE_COMPUTE_CPU OFF)\n"
"set(HYLE_COMPUTE_CUDA OFF)\n"
"set(HYLE_COMPUTE_APPLE OFF)\n"
"set(HYLE_COMPUTE_ONEDNN OFF)\n"
"set(HYLE_COMPUTE_VULKAN ON)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"foreach(_file IN LISTS _asm)\n"
"  get_source_file_property(_language \"\${_file}\" LANGUAGE)\n"
"  if(NOT _language STREQUAL \"ASM_MASM\")\n"
"    message(FATAL_ERROR \"Parent MASM source became \${_language}: \${_file}\")\n"
"  endif()\n"
"endforeach()\n")

set(_generator_args -G "${HYLE_PARENT_GENERATOR}")
if(NOT "${HYLE_PARENT_PLATFORM}" STREQUAL "")
  list(APPEND _generator_args -A "${HYLE_PARENT_PLATFORM}")
endif()
if(NOT "${HYLE_PARENT_TOOLSET}" STREQUAL "")
  list(APPEND _generator_args -T "${HYLE_PARENT_TOOLSET}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${_source}" -B "${_build}"
    ${_generator_args} -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_CXX_COMPILER=${HYLE_PARENT_CXX_COMPILER}"
    "-DHYLE_DEP_SOURCE_CACHE=${HYLE_DEP_SOURCE_CACHE}"
    "-DFETCHCONTENT_SOURCE_DIR_GGML_VULKAN_RUNTIME=${HYLE_GGML_SOURCE}"
    "-DVulkan_GLSLC_EXECUTABLE=${HYLE_PARENT_GLSLC}"
    "-DVulkan_INCLUDE_DIR=${HYLE_PARENT_VULKAN_INCLUDE}"
    "-DVulkan_LIBRARY=${HYLE_PARENT_VULKAN_LIBRARY}"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  COMMAND_ECHO STDOUT RESULT_VARIABLE _configure)
if(NOT _configure EQUAL 0)
  message(FATAL_ERROR "Vulkan/MASM parent configure failed: ${_configure}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_build}" --config Release
    --target hyle_masm_parent --parallel 2
  COMMAND_ECHO STDOUT RESULT_VARIABLE _compile)
if(NOT _compile EQUAL 0)
  message(FATAL_ERROR "Vulkan/MASM parent assembly build failed: ${_compile}")
endif()
message(STATUS "Actual Vulkan provider and parent MASM composition passed")
