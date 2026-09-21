include_guard(GLOBAL)

hyle_auto_feature_option(HYLE_COMPUTE_APPLE
  "Build the native Apple MLX Metal execution route" DEFAULT OFF)

function(hyle_configure_apple_dependencies)
  if(HYLE_COMPUTE_APPLE STREQUAL "OFF")
    message(STATUS "HYLE: Apple route OFF -- MLX and Metal probe/download skipped")
    set(HYLE_COMPUTE_APPLE_ENABLED OFF PARENT_SCOPE)
    return()
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR
     NOT HYLE_TARGET_PROCESSOR STREQUAL "aarch64")
    if(HYLE_COMPUTE_APPLE STREQUAL "ON")
      message(FATAL_ERROR "HYLE_COMPUTE_APPLE=ON requires macOS Apple Silicon")
    endif()
    set(HYLE_COMPUTE_APPLE_ENABLED OFF PARENT_SCOPE)
    return()
  endif()

  # JIT mode still builds base kernels into mlx.metallib. Fail at admission
  # with the missing tool, before downloading/configuring the native sources.
  foreach(_tool IN ITEMS metal metallib)
    execute_process(COMMAND xcrun -sdk macosx ${_tool} --version
      RESULT_VARIABLE _result OUTPUT_VARIABLE _version ERROR_VARIABLE _error
      TIMEOUT 10 OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _result EQUAL 0)
      if(HYLE_COMPUTE_APPLE STREQUAL "ON")
        message(FATAL_ERROR "HYLE Apple requires the Xcode Metal toolchain (${_tool}): ${_error}")
      endif()
      message(STATUS "HYLE: Apple AUTO unavailable -- Metal toolchain ${_tool} missing")
      set(HYLE_COMPUTE_APPLE_ENABLED OFF PARENT_SCOPE)
      return()
    endif()
    if(_tool STREQUAL "metal")
      string(REGEX MATCH "^[^\r\n]*" _compiler "${_version}")
      string(REPLACE "\\" "\\\\" _compiler "${_compiler}")
      string(REPLACE "\"" "\\\"" _compiler "${_compiler}")
      set(HYLE_APPLE_METAL_COMPILER "${_compiler}" PARENT_SCOPE)
    endif()
  endforeach()
  if(NOT HYLE_BUILD_COMPUTE_ROUTE_APPLE)
    # The independent ggml Metal factor needs Metal admission, not MLX sources.
    set(HYLE_COMPUTE_APPLE_ENABLED ON PARENT_SCOPE)
    return()
  endif()
  include(FetchContent)
  # Authenticate all nested sources before any vendor configure. Their later
  # FetchContent declarations reuse these names and already populated inputs.
  foreach(_dependency IN ITEMS metal_cpp json fmt)
    string(TOUPPER "${_dependency}" _upper)
    FetchContent_Declare(${_dependency}
      URL "${HYLE_DEP_${_upper}_ARCHIVE_URL}"
      URL_HASH "SHA256=${HYLE_DEP_${_upper}_ARCHIVE_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      EXCLUDE_FROM_ALL)
  endforeach()
  FetchContent_Declare(mlx
    URL "${HYLE_DEP_MLX_ARCHIVE_URL}"
    URL_HASH "SHA256=${HYLE_DEP_MLX_ARCHIVE_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    SOURCE_SUBDIR hyle-verified-source-only)
  hyle_fetchcontent_make_available_locked(mlx)
  FetchContent_GetProperties(mlx)
  hyle_audit_in_tree_dependencies(mlx_cast_patches)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_patch_manifest "${HYLE_SOURCE_DIR}/third_party/mlx_cast/patches.json")
  # Reuse the existing exact-manifest materializer; cache originals stay read-only.
  set(_materializer "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime/prepare_source.py")
  string(SUBSTRING "${HYLE_DEP_MLX_CAST_PATCHES_VENDORED_TREE_SHA256}" 0 16 _patch_id)
  set(_mlx_patched_source "${HYLE_DEP_BUILD_DIR}/mlx-${_patch_id}/source")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_patch_manifest}" "${_materializer}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_materializer}"
      --source "${mlx_SOURCE_DIR}" --output "${_mlx_patched_source}"
      --manifest "${_patch_manifest}"
    RESULT_VARIABLE _patch_result OUTPUT_VARIABLE _patch_output ERROR_VARIABLE _patch_error)
  if(NOT _patch_result EQUAL 0)
    message(FATAL_ERROR "MLX source preparation failed: ${_patch_error}")
  endif()
  block()
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
    set(MLX_METAL_JIT ON CACHE BOOL "" FORCE)
    foreach(_option IN ITEMS MLX_BUILD_CPU MLX_BUILD_CUDA MLX_BUILD_TESTS
        MLX_BUILD_EXAMPLES MLX_BUILD_BENCHMARKS MLX_BUILD_PYTHON_BINDINGS
        MLX_BUILD_PYTHON_STUBS MLX_BUILD_GGUF MLX_BUILD_SAFETENSORS
        MLX_METAL_DEBUG MLX_USE_CCACHE USE_SYSTEM_FMT
        USE_ASAN USE_UBSAN USE_TSAN JSON_BuildTests FMT_TEST FMT_DOC FMT_INSTALL)
      set(${_option} OFF CACHE BOOL "" FORCE)
    endforeach()
    # Native arrays are part of the route-specific C++ API. A shared package
    # must have one shared MLX runtime, never duplicate static allocators/caches
    # in both a HYLE DSO and its consumer.
    if(HYLE_STATIC)
      set(BUILD_SHARED_LIBS OFF)
    else()
      set(BUILD_SHARED_LIBS ON)
    endif()
    hyle_fetchcontent_make_available_locked(metal_cpp)
    hyle_fetchcontent_make_available_locked(json)
    hyle_fetchcontent_make_available_locked(fmt)
    add_subdirectory("${_mlx_patched_source}" "${mlx_BINARY_DIR}" EXCLUDE_FROM_ALL)
  endblock()
  foreach(_dependency IN ITEMS metal_cpp json fmt mlx)
    FetchContent_GetProperties(${_dependency})
    if(_dependency STREQUAL "mlx")
      set(mlx_SOURCE_DIR "${_mlx_patched_source}")
    endif()
    if(EXISTS "${${_dependency}_SOURCE_DIR}/CMakeLists.txt")
      set_property(DIRECTORY "${${_dependency}_SOURCE_DIR}"
        PROPERTY EXCLUDE_FROM_ALL TRUE)
    endif()
    string(TOUPPER "${_dependency}" _upper)
    set(HYLE_APPLE_${_upper}_SOURCE_DIR "${${_dependency}_SOURCE_DIR}" PARENT_SCOPE)
  endforeach()
  set_target_properties(mlx PROPERTIES
    SYSTEM TRUE
    EXPORT_NAME "_mlx_native"
    INSTALL_NAME_DIR "@rpath"
    INSTALL_RPATH "@loader_path"
    HYLE_DEP_SOURCE_DIR "${mlx_SOURCE_DIR}"
    HYLE_DEP_BINARY_DIR "${mlx_BINARY_DIR}")
  # The consumer uses its own Apple SDK. Do not export a producer's absolute
  # CLT/Xcode SDK framework paths into a relocated C++ package.
  get_target_property(_public_links mlx INTERFACE_LINK_LIBRARIES)
  set(_relocatable_links "")
  foreach(_link IN LISTS _public_links)
    if(_link MATCHES "/(Metal|Foundation|QuartzCore)\\.framework$")
      set(_link "-framework ${CMAKE_MATCH_1}")
    endif()
    list(APPEND _relocatable_links "${_link}")
  endforeach()
  set_property(TARGET mlx PROPERTY INTERFACE_LINK_LIBRARIES "${_relocatable_links}")
  set(HYLE_COMPUTE_APPLE_ENABLED ON PARENT_SCOPE)
  message(STATUS "HYLE: MLX ${HYLE_DEP_MLX_VERSION} Metal-only, cold JIT, native region enabled")
endfunction()
