# A complete original CPU program, isolated from the baseline and GPU owners.
# Read the locked target's composition rather than maintaining a kernel list.
get_target_property(_cpu_sources ggml-cpu SOURCES)
get_target_property(_cpu_source_dir ggml-cpu SOURCE_DIR)
set(_cpu_absolute_sources "")
foreach(_source IN LISTS _cpu_sources)
  if(_source MATCHES "\\$<")
    message(FATAL_ERROR "Locked ggml CPU source composition changed")
  endif()
  cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${_cpu_source_dir}" OUTPUT_VARIABLE _path)
  list(APPEND _cpu_absolute_sources "${_path}")
endforeach()
add_library(hyle_ggml_cpu_avx2 MODULE ${_cpu_absolute_sources} cpu_program_export.cc)
set_target_properties(hyle_ggml_cpu_avx2 PROPERTIES
  EXPORT_NAME "_${_cpu_program_basename}" OUTPUT_NAME "${_cpu_program_basename}"
  C_STANDARD 11 CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(hyle_ggml_cpu_avx2 PRIVATE
  "${HYLE_SOURCE_DIR}/include"
  "$<TARGET_PROPERTY:ggml-cpu,INCLUDE_DIRECTORIES>")
target_compile_definitions(hyle_ggml_cpu_avx2 PRIVATE
  "$<TARGET_PROPERTY:ggml-cpu,COMPILE_DEFINITIONS>"
  GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
target_compile_options(hyle_ggml_cpu_avx2 PRIVATE
  "$<TARGET_PROPERTY:ggml-cpu,COMPILE_OPTIONS>")
if(MSVC)
  target_compile_definitions(hyle_ggml_cpu_avx2 PRIVATE __BMI2__)
endif()
include("${HYLE_SOURCE_DIR}/cmake/HyleIsa.cmake")
hyle_isa_flags(_cpu_isa_flags AVX2)
target_compile_options(hyle_ggml_cpu_avx2 PRIVATE ${_cpu_isa_flags})
target_link_libraries(hyle_ggml_cpu_avx2 PRIVATE
  "$<TARGET_PROPERTY:ggml-cpu,LINK_LIBRARIES>")
hyle_apply_sanitizer(hyle_ggml_cpu_avx2)
add_dependencies(hyle_compute_ggml_runtime hyle_ggml_cpu_avx2)
# dlopen uses the calling shared object's RUNPATH. The executable's runtime
# paths do not supply this private sibling to a build-tree shared adapter.
# Installed shared libraries already use the package's sibling $ORIGIN path.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(TARGET hyle_compute_ggml_runtime APPEND PROPERTY BUILD_RPATH
    "$<TARGET_FILE_DIR:hyle_ggml_cpu_avx2>")
endif()

target_compile_definitions(hyle_compute_ggml_runtime PRIVATE
  HYLE_GGML_CPU_AVX2_PROGRAM=1
  HYLE_GGML_CPU_PROGRAM_LIBRARY="$<TARGET_FILE_NAME:hyle_ggml_cpu_avx2>")
hyle_register_runtime_files(TARGETS hyle_compute_ggml_runtime
  FILES "$<TARGET_FILE:hyle_ggml_cpu_avx2>")

include(CMakePackageConfigHelpers)
if(WIN32)
  set(HYLE_GGML_CPU_PROGRAM_INSTALLED_FILE
    "${CMAKE_INSTALL_LIBDIR}/${_cpu_program_basename}${CMAKE_SHARED_MODULE_SUFFIX}")
else()
  set(HYLE_GGML_CPU_PROGRAM_INSTALLED_FILE
    "${CMAKE_INSTALL_LIBDIR}/lib${_cpu_program_basename}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/HyleGgmlCpuRuntimeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HyleGgmlCpuRuntimeConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HyleGgmlCpuRuntime"
  PATH_VARS HYLE_GGML_CPU_PROGRAM_INSTALLED_FILE)
hyle_register_package_dependency(NAME ggml_cpu_programs OWNER compute_ggml_runtime
  EXPORT_TARGETS hyle_ggml_cpu_avx2
  FEATURES compute_ggml_cpu_program
  FIND_DEPENDENCIES HyleGgmlCpuRuntime
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/HyleGgmlCpuRuntimeConfig.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleGgmlCpuRuntime|HyleGgmlCpuRuntimeConfig.cmake")
