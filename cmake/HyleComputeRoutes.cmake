include_guard(GLOBAL)

# Public products select one complete route set. Maintenance factors derive
# private availability from their actual DAG rather than offering a lite SKU.
function(_hyle_reject_legacy_compute_options)
  foreach(_name IN ITEMS HYLE_COMPUTE_CPU HYLE_COMPUTE_CUDA HYLE_COMPUTE_APPLE
      HYLE_COMPUTE_VULKAN HYLE_COMPUTE_ONEDNN HYLE_COMPUTE_TORCH_CPU)
    if(DEFINED ${_name})
      message(FATAL_ERROR
        "${_name} is no longer a public build option. Use "
        "HYLE_COMPUTE_ROUTES=cpu, cuda, apple, vulkan, or a semicolon-separated "
        "set such as 'cpu;cuda'. Remove the old option from the parent/preset "
        "and use a fresh build tree; required Providers are automatic.")
    endif()
  endforeach()
endfunction()

function(_hyle_parse_compute_routes value output)
  if(value STREQUAL "")
    message(FATAL_ERROR "HYLE_COMPUTE_ROUTES must be a nonempty route set")
  endif()
  set(_requested "${value}")
  set(_known cpu cuda apple vulkan)
  set(_normalized "")
  foreach(_route IN LISTS _requested)
    string(STRIP "${_route}" _route)
    string(TOLOWER "${_route}" _route)
    if(NOT _route IN_LIST _known)
      message(FATAL_ERROR
        "Unknown HYLE_COMPUTE_ROUTES item '${_route}'; use cpu, cuda, apple, "
        "vulkan or a semicolon-separated set. AUTO and empty items are not routes.")
    endif()
    list(APPEND _normalized "${_route}")
  endforeach()
  # Canonical order is identity only, never provider/device preference.
  set(_canonical "")
  foreach(_route IN LISTS _known)
    if(_route IN_LIST _normalized)
      list(APPEND _canonical "${_route}")
    endif()
  endforeach()
  set(${output} "${_canonical}" PARENT_SCOPE)
endfunction()

function(_hyle_check_compute_route_platform routes)
  foreach(_route IN LISTS routes)
    if(_route STREQUAL "cpu")
      if(NOT (CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$" AND
              HYLE_TARGET_PROCESSOR STREQUAL "x86_64") AND
         NOT (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
              HYLE_TARGET_PROCESSOR STREQUAL "aarch64"))
        message(FATAL_ERROR
          "The complete CPU route requires Windows/Linux x86_64 or macOS ARM64 "
          "for its locked ATen SDK; '${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}' "
          "has no complete qualified CPU recipe. Select another supported route.")
      endif()
    elseif(_route STREQUAL "cuda")
      if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
        message(FATAL_ERROR "The CUDA route requires Windows or Linux")
      endif()
    elseif(_route STREQUAL "apple")
      if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR
         NOT HYLE_TARGET_PROCESSOR STREQUAL "aarch64")
        message(FATAL_ERROR "The Apple route requires macOS Apple Silicon")
      endif()
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux|Darwin)$")
      message(FATAL_ERROR "The Vulkan route requires Windows, Linux, or macOS")
    endif()
  endforeach()
endfunction()

# Used inside the resolver functions; publish to their calling project scope.
macro(_hyle_publish_compute_routes routes)
  set(_active "${routes}")
  set(HYLE_COMPUTE_ROUTE_LIST "${_active}" PARENT_SCOPE)
  foreach(_route IN ITEMS cpu cuda apple vulkan)
    string(TOUPPER "${_route}" _upper)
    if(_route IN_LIST _active)
      set(HYLE_COMPUTE_${_upper} ON PARENT_SCOPE)
    else()
      set(HYLE_COMPUTE_${_upper} OFF PARENT_SCOPE)
    endif()
  endforeach()
  set(_onednn OFF)
  set(_torch OFF)
  if("cpu" IN_LIST _active)
    if(CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux|Darwin)$" AND
       HYLE_TARGET_PROCESSOR STREQUAL "x86_64")
      set(_onednn ON)
    endif()
    if((CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$" AND
        HYLE_TARGET_PROCESSOR STREQUAL "x86_64") OR
       (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
        HYLE_TARGET_PROCESSOR STREQUAL "aarch64"))
      set(_torch ON)
    endif()
  endif()
  set(HYLE_COMPUTE_ONEDNN "${_onednn}" PARENT_SCOPE)
  set(HYLE_COMPUTE_TORCH_CPU "${_torch}" PARENT_SCOPE)
endmacro()

function(hyle_resolve_compute_product_routes products)
  _hyle_reject_legacy_compute_options()
  if(NOT "compute" IN_LIST products AND NOT "graph" IN_LIST products)
    _hyle_publish_compute_routes("")
    return()
  endif()
  if(DEFINED HYLE_COMPUTE_ROUTES)
    set(_input "${HYLE_COMPUTE_ROUTES}")
  else()
    set(_input cpu)
  endif()
  _hyle_parse_compute_routes("${_input}" _routes)
  _hyle_check_compute_route_platform("${_routes}")
  set(HYLE_COMPUTE_ROUTES "${_routes}" CACHE STRING
    "Complete Compute routes: cpu;cuda;apple;vulkan (a nonempty subset)" FORCE)
  set(HYLE_COMPUTE_ROUTES "${_routes}" PARENT_SCOPE)
  _hyle_publish_compute_routes("${_routes}")
endfunction()

function(hyle_resolve_compute_factor_routes)
  _hyle_reject_legacy_compute_options()
  if(DEFINED HYLE_COMPUTE_ROUTES)
    message(FATAL_ERROR
      "HYLE_DEV_FACTORS derives native availability from its exact factor/test "
      "closure. Do not combine it with the product selector HYLE_COMPUTE_ROUTES.")
  endif()
  set(_routes "")
  if(HYLE_BUILD_COMPUTE_BACKEND_CPU OR HYLE_BUILD_COMPUTE_PROVIDER_NATIVE OR
     HYLE_BUILD_COMPUTE_PROVIDER_GGML OR HYLE_BUILD_COMPUTE_PROVIDER_ONEDNN)
    list(APPEND _routes cpu)
  endif()
  if(HYLE_BUILD_COMPUTE_BACKEND_CUDA AND
     CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
    list(APPEND _routes cuda)
  endif()
  if((HYLE_BUILD_COMPUTE_ROUTE_APPLE OR HYLE_BUILD_COMPUTE_PROVIDER_GGML_METAL) AND
     CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
     HYLE_TARGET_PROCESSOR STREQUAL "aarch64")
    list(APPEND _routes apple)
  endif()
  if((HYLE_BUILD_COMPUTE_ROUTE_VULKAN OR HYLE_BUILD_COMPUTE_PROVIDER_GGML_VULKAN OR
      HYLE_BUILD_COMPUTE_PROVIDER_IREE_VULKAN) AND
     CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux|Darwin)$")
    list(APPEND _routes vulkan)
  endif()
  _hyle_publish_compute_routes("${_routes}")
endfunction()
