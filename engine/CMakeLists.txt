cmake_minimum_required(VERSION 3.25)

# The same application builds in the source project or against an installed SDK.
# Module-only consumers never enter this directory.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hyle_engine LANGUAGES CXX)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  find_package(Hyle 0.1.0 EXACT REQUIRED
    COMPONENTS compute graph graph_progress compute_stream mem)
  include("${CMAKE_CURRENT_LIST_DIR}/../cmake/HyleWarnings.cmake")
  set(_hyle_engine_standalone TRUE)
  set(_hyle_engine_routes ${Hyle_AVAILABLE_COMPUTE_ROUTES})
else()
  set(_hyle_engine_routes ${HYLE_COMPUTE_ROUTE_LIST})
endif()

if(NOT _hyle_engine_routes)
  message(FATAL_ERROR "Hyle inference requires a CPU, CUDA, Apple or Vulkan Compute route")
endif()

function(hyle_configure_engine_target target)
  target_compile_features(${target} PRIVATE cxx_std_23)
  target_sources(${target} PRIVATE
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/reporting/execution_report.cc")
  target_link_libraries(${target} PRIVATE hyle::graph hyle::compute
    hyle::graph_progress hyle::compute_stream hyle::mem)
  # Source and standalone application builds share the module warning policy;
  # public preprocessing/UTF-8 requirements arrive through linked SDK targets.
  if(COMMAND hyle_apply_warnings)
    hyle_apply_warnings(${target})
  endif()
  set_target_properties(${target} PROPERTIES FOLDER engine)
  hyle_stage_runtime(${target})
endfunction()

add_executable(hyle_engine main.cc)
set_target_properties(hyle_engine PROPERTIES OUTPUT_NAME hyle)
hyle_configure_engine_target(hyle_engine)
add_subdirectory(models/qwen35)

if(_hyle_engine_standalone)
  include(GNUInstallDirs)
  install(TARGETS hyle_engine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Hyle_Runtime)
  include("${CMAKE_CURRENT_LIST_DIR}/../cmake/HylePackage.cmake")
  hyle_install_project_notices("${CMAKE_CURRENT_LIST_DIR}/..")
endif()
