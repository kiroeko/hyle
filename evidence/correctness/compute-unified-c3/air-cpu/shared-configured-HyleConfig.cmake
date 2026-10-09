
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was HyleConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

set(Hyle_AVAILABLE_COMPUTE_ROUTES "cpu")
set(Hyle_COMPUTE_RECIPE_ID "f019d5ee88862c25ecaee8463683381322b474459d51aac4be9d8d6c6b70a1f7")
foreach(_Hyle_legacy IN ITEMS HYLE_COMPUTE_CPU HYLE_COMPUTE_CUDA HYLE_COMPUTE_APPLE
    HYLE_COMPUTE_VULKAN HYLE_COMPUTE_ONEDNN HYLE_COMPUTE_TORCH_CPU)
  if(DEFINED ${_Hyle_legacy})
    set(Hyle_FOUND FALSE)
    set(Hyle_NOT_FOUND_MESSAGE
      "${_Hyle_legacy} is no longer a public build option; request HYLE_COMPUTE_ROUTES or read this SDK's capabilities")
    return()
  endif()
endforeach()
unset(_Hyle_legacy)
# A requested set only asserts capabilities in this immutable SDK. It cannot
# download/recompile Providers or alter the imported target's presence macros.
if(DEFINED HYLE_COMPUTE_ROUTES)
  foreach(_Hyle_requested IN LISTS HYLE_COMPUTE_ROUTES)
    string(STRIP "${_Hyle_requested}" _Hyle_requested)
    string(TOLOWER "${_Hyle_requested}" _Hyle_requested)
    if(NOT _Hyle_requested IN_LIST Hyle_AVAILABLE_COMPUTE_ROUTES)
      set(Hyle_FOUND FALSE)
      set(Hyle_NOT_FOUND_MESSAGE
        "Hyle SDK does not contain requested Compute route '${_Hyle_requested}'; available: ${Hyle_AVAILABLE_COMPUTE_ROUTES}")
      return()
    endif()
  endforeach()
  if(HYLE_COMPUTE_ROUTES STREQUAL "")
    set(Hyle_FOUND FALSE)
    set(Hyle_NOT_FOUND_MESSAGE "HYLE_COMPUTE_ROUTES must be a nonempty route requirement")
    return()
  endif()
  unset(_Hyle_requested)
endif()

set(_Hyle_find_dependency_specs "HyleTorchCpuSdk")
foreach(_Hyle_find_dependency_spec IN LISTS _Hyle_find_dependency_specs)
  string(REPLACE "|" ";" _Hyle_find_dependency_args
    "${_Hyle_find_dependency_spec}")
  find_dependency(${_Hyle_find_dependency_args})
endforeach()
unset(_Hyle_find_dependency_args)
unset(_Hyle_find_dependency_spec)
unset(_Hyle_find_dependency_specs)

include("${CMAKE_CURRENT_LIST_DIR}/HyleTargets.cmake")

# SIMD's expert development target is an opt-in header capability. Load the
# exact same fixed/fat helper implementation used by source embedding only when
# this installation contains the SIMD product.
if(TARGET hyle::simd_expert)
  include("${CMAKE_CURRENT_LIST_DIR}/HyleIsa.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/HyleDispatch.cmake")
endif()

set(Hyle_AVAILABLE_COMPONENTS "core;error;simd;math;compute;graph")
set(Hyle_AVAILABLE_FEATURES "compute_cpu;compute_torch_cpu")
if("compute_apple" IN_LIST Hyle_AVAILABLE_FEATURES)
  include("${CMAKE_CURRENT_LIST_DIR}/HyleAppleRuntime.cmake")
  get_filename_component(_Hyle_metallib
    "${CMAKE_CURRENT_LIST_DIR}/../../mlx.metallib" ABSOLUTE)
  if(NOT EXISTS "${_Hyle_metallib}")
    message(FATAL_ERROR "Hyle Apple SDK is missing native resource '${_Hyle_metallib}'")
  endif()
  set_property(TARGET hyle::_mlx_native PROPERTY HYLE_APPLE_METALLIB "${_Hyle_metallib}")
  unset(_Hyle_metallib)
endif()
set(Hyle_STATIC OFF)
set(_Hyle_missing_components "")
foreach(_Hyle_component IN LISTS Hyle_FIND_COMPONENTS)
  if(_Hyle_component IN_LIST Hyle_AVAILABLE_COMPONENTS AND
     TARGET hyle::${_Hyle_component})
    set(Hyle_${_Hyle_component}_FOUND TRUE)
  else()
    set(Hyle_${_Hyle_component}_FOUND FALSE)
    list(APPEND _Hyle_missing_components "${_Hyle_component}")
    string(CONCAT Hyle_${_Hyle_component}_NOT_FOUND_MESSAGE
      "Hyle product '${_Hyle_component}' is not present in this installation; "
      "available products: ${Hyle_AVAILABLE_COMPONENTS}")
  endif()
endforeach()
if(_Hyle_missing_components)
  list(JOIN _Hyle_missing_components ", " _Hyle_missing_text)
  list(JOIN Hyle_AVAILABLE_COMPONENTS ", " _Hyle_available_text)
  string(CONCAT Hyle_NOT_FOUND_MESSAGE
    "missing requested component(s): ${_Hyle_missing_text}; available in this "
    "installation: ${_Hyle_available_text}")
endif()
check_required_components(Hyle)
unset(_Hyle_missing_components)
unset(_Hyle_missing_text)
unset(_Hyle_available_text)

include("${CMAKE_CURRENT_LIST_DIR}/HyleRuntimeStaging.cmake")
