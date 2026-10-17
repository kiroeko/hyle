include_guard(GLOBAL)

# Some native SDKs omit loader-only DLLs from their imported target graph.
# Associate those files with real owners, rather than copying them into every
# executable merely because the SDK was discovered during configuration.
function(hyle_register_runtime_files)
  cmake_parse_arguments(ARG "" "" "TARGETS;FILES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_TARGETS OR NOT ARG_FILES)
    message(FATAL_ERROR "hyle_register_runtime_files requires TARGETS and FILES")
  endif()
  foreach(_owner IN LISTS ARG_TARGETS)
    set(_registered_owner "${_owner}")
    if(TARGET "${_owner}")
      get_target_property(_alias "${_owner}" ALIASED_TARGET)
      if(_alias)
        set(_owner "${_alias}")
      endif()
      get_property(_files TARGET "${_owner}" PROPERTY HYLE_RUNTIME_FILES)
      list(APPEND _files ${ARG_FILES})
      list(REMOVE_DUPLICATES _files)
      set_property(TARGET "${_owner}" PROPERTY HYLE_RUNTIME_FILES "${_files}")
    else()
      # Installed dependency shims run before their Hyle owner is imported.
      # Keep pending registrations in that directory, rather than identifying
      # directory-scoped imported targets by their globally repeated names.
      string(SHA256 _key "${_owner}")
      get_property(_files DIRECTORY PROPERTY "HYLE_RUNTIME_PENDING_${_key}")
      list(APPEND _files ${ARG_FILES})
      list(REMOVE_DUPLICATES _files)
      set_property(DIRECTORY PROPERTY "HYLE_RUNTIME_PENDING_${_key}" "${_files}")
    endif()
    # A parent can inspect directory properties after this scope is processed,
    # even when it cannot inspect the imported target. Record only the fact
    # that this owner needs HYLE's extra resources, including a supplied alias.
    foreach(_registered_name IN ITEMS "${_registered_owner}" "${_owner}")
      string(SHA256 _key "${_registered_name}")
      set_property(DIRECTORY PROPERTY "HYLE_RUNTIME_REGISTERED_${_key}" TRUE)
    endforeach()
  endforeach()
endfunction()

function(_hyle_registered_runtime_files owner output)
  get_property(_files TARGET "${owner}" PROPERTY HYLE_RUNTIME_FILES)
  get_target_property(_directory "${owner}" SOURCE_DIR)
  string(SHA256 _key "${owner}")
  # Source dependency setup may register in an ancestor of the target's
  # creation directory. Imported targets use their find_package directory.
  # A nearer registration replaces a pending name from a more distant scope.
  while(_directory)
    get_property(_registered DIRECTORY "${_directory}"
      PROPERTY "HYLE_RUNTIME_PENDING_${_key}" SET)
    if(_registered)
      get_property(_pending DIRECTORY "${_directory}"
        PROPERTY "HYLE_RUNTIME_PENDING_${_key}")
      list(APPEND _files ${_pending})
      break()
    endif()
    get_property(_directory DIRECTORY "${_directory}" PROPERTY PARENT_DIRECTORY)
  endwhile()
  set(${output} "${_files}" PARENT_SCOPE)
endfunction()

# Imported targets belong to their find_package directory and descendants.
# Ordinary cross-directory wrappers are globally visible, but their private
# imported SDK targets may be invisible where an application requests staging.
# Guard Hyle entry points and explicitly registered extra resources. Ordinary
# imported DLLs still use CMake's TARGET_RUNTIME_DLLS closure; source embedding
# must not require users to rediscover private CUDA/OpenSSL dependencies.
function(_hyle_runtime_target_is_hidden owner candidate output)
  if(candidate MATCHES "^hyle::")
    set(${output} TRUE PARENT_SCOPE)
    return()
  endif()
  get_target_property(_directory "${owner}" SOURCE_DIR)
  string(SHA256 _key "${candidate}")
  while(_directory)
    get_property(_registered DIRECTORY "${_directory}"
      PROPERTY "HYLE_RUNTIME_REGISTERED_${_key}")
    if(_registered)
      set(${output} TRUE PARENT_SCOPE)
      return()
    endif()
    get_property(_directory DIRECTORY "${_directory}" PROPERTY PARENT_DIRECTORY)
  endwhile()
  set(${output} FALSE PARENT_SCOPE)
endfunction()

function(_hyle_collect_runtime_files target output)
  # CMake configure is serialized and this walker does not invoke callbacks.
  # Reset per collection: targets/runtime assets may change before a later call.
  set_property(GLOBAL PROPERTY HYLE_RUNTIME_WALK_SEEN "")
  set_property(GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_ASSETS "")
  set_property(GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_PRODUCERS "")
  set_property(GLOBAL PROPERTY HYLE_RUNTIME_WALK_UNRESOLVED "")
  _hyle_runtime_files_walk("${target}" "1" "" _files)
  list(REMOVE_DUPLICATES _files)
  set(${output} "${_files}" PARENT_SCOPE)
endfunction()

function(_hyle_runtime_files_walk owner condition ancestors output)
  set(_files "")
  if(NOT TARGET "${owner}" OR owner IN_LIST ancestors)
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  string(SHA256 _visit "${owner}|${condition}")
  get_property(_seen GLOBAL PROPERTY HYLE_RUNTIME_WALK_SEEN)
  if(_visit IN_LIST _seen)
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY HYLE_RUNTIME_WALK_SEEN "${_visit}")
  list(APPEND ancestors "${owner}")
  _hyle_registered_runtime_files("${owner}" _extra)
  foreach(_file IN LISTS _extra)
    list(APPEND _files "$<${condition}:${_file}>")
  endforeach()
  get_target_property(_alias "${owner}" ALIASED_TARGET)
  if(_alias)
    _hyle_runtime_files_walk("${_alias}" "${condition}" "${ancestors}" _alias_files)
    list(APPEND _files ${_alias_files})
  else()
    # Route-owned resource, using exactly the same conditional link walk as
    # DLLs. Merely having MLX in a mixed SDK must not stage an unlinked asset.
    get_target_property(_metallib "${owner}" HYLE_APPLE_METALLIB)
    if(_metallib)
      set_property(GLOBAL APPEND PROPERTY HYLE_RUNTIME_WALK_APPLE_ASSETS
        "$<${condition}:${_metallib}>")
      get_target_property(_imported "${owner}" IMPORTED)
      if(NOT _imported)
        set_property(GLOBAL APPEND PROPERTY HYLE_RUNTIME_WALK_APPLE_PRODUCERS
          "$<${condition}:${owner}>")
      endif()
    endif()
  endif()
  set(_link_properties LINK_LIBRARIES INTERFACE_LINK_LIBRARIES
    IMPORTED_LINK_DEPENDENT_LIBRARIES)
  get_target_property(_configurations "${owner}" IMPORTED_CONFIGURATIONS)
  if(NOT _configurations)
    set(_configurations "")
  endif()
  # Handwritten imports may set configuration properties without listing
  # IMPORTED_CONFIGURATIONS. Include ordinary and consumer configurations too.
  list(APPEND _configurations Debug Release RelWithDebInfo MinSizeRel
    ${CMAKE_BUILD_TYPE} ${CMAKE_CONFIGURATION_TYPES})
  list(TRANSFORM _configurations TOUPPER)
  list(REMOVE_DUPLICATES _configurations)
  foreach(_configuration IN LISTS _configurations)
    list(APPEND _link_properties
      "IMPORTED_LINK_DEPENDENT_LIBRARIES_${_configuration}")
  endforeach()
  foreach(_property IN LISTS _link_properties)
    get_target_property(_links "${owner}" "${_property}")
    list(REMOVE_DUPLICATES _links)
    set(_property_condition "${condition}")
    if(_links AND _property MATCHES "^IMPORTED_LINK_DEPENDENT_LIBRARIES_(.+)$")
      set(_configuration "${CMAKE_MATCH_1}")
      get_target_property(_location "${owner}" "IMPORTED_LOCATION_${_configuration}")
      if(NOT _location)
        get_target_property(_location "${owner}" IMPORTED_LOCATION)
      endif()
      # Let CMake select the imported artifact, including explicit mappings and
      # its default fallback. Configurations sharing one binary share the union
      # of its declared loader dependencies; none may be silently omitted.
      set(_property_condition
        "$<AND:${condition},$<CONFIG:${_configuration}>>")
      get_target_property(_owner_type "${owner}" TYPE)
      if(_location AND _owner_type MATCHES "^(STATIC|SHARED|MODULE|UNKNOWN)_LIBRARY$")
        file(TO_CMAKE_PATH "${_location}" _location)
        get_target_property(_owner_directory "${owner}" SOURCE_DIR)
        cmake_path(ABSOLUTE_PATH _location BASE_DIRECTORY "${_owner_directory}"
          NORMALIZE OUTPUT_VARIABLE _location)
        set(_property_condition
          "$<AND:${condition},$<PATH_EQUAL:$<TARGET_FILE:${owner}>,${_location}>>")
      endif()
    endif()
    foreach(_link IN LISTS _links)
      # These single-target wrappers have no conditional runtime branch in the
      # current build. Keep genuine config/boolean expressions below intact.
      set(_literal "${_link}")
      while(_literal MATCHES "^\\$<(LINK_ONLY|BUILD_INTERFACE|1):(.*)>$")
        set(_literal "${CMAKE_MATCH_2}")
      endwhile()
      if(TARGET "${_literal}")
        set(_link "${_literal}")
      endif()
      if(TARGET "${_link}")
        _hyle_runtime_files_walk("${_link}" "${_property_condition}" "${ancestors}" _linked_files)
        list(APPEND _files ${_linked_files})
      else()
        # Discover possible identities, then preserve the actual evaluated
        # edge. An inactive config/boolean branch must not stage its files.
        # LINK_ONLY suppresses usage requirements, not runtime linkage.
        string(REPLACE "$<LINK_ONLY:" "$<1:" _evaluated_link "${_link}")
        string(REGEX MATCHALL "[A-Za-z0-9_.+-]+(::[A-Za-z0-9_.+-]+)*" _names "${_link}")
        list(REMOVE_DUPLICATES _names)
        foreach(_name IN LISTS _names)
          set(_edge "$<IN_LIST:${_name},$<TARGET_GENEX_EVAL:${owner},${_evaluated_link}>>")
          if(TARGET "${_name}")
            _hyle_runtime_files_walk("${_name}" "$<AND:${_property_condition},${_edge}>" "${ancestors}" _linked_files)
            list(APPEND _files ${_linked_files})
          else()
            _hyle_runtime_target_is_hidden("${owner}" "${_name}" _hidden)
            if(_hidden)
              # Preserve the evaluated edge. A private SDK in an inactive
              # branch does not prevent another configuration from building.
              set_property(GLOBAL APPEND PROPERTY HYLE_RUNTIME_WALK_UNRESOLVED
                "$<$<AND:${_property_condition},${_edge}>:${_name} (via ${owner})>")
            endif()
          endif()
        endforeach()
      endif()
    endforeach()
  endforeach()
  set(${output} "${_files}" PARENT_SCOPE)
endfunction()

# Build-time checking keeps conditional links conditional without inventing a
# second directory/target registry. The selected configuration must fail before
# application linkage instead of silently staging an incomplete runtime closure.
function(_hyle_check_runtime_visibility target)
  get_property(_unresolved GLOBAL PROPERTY HYLE_RUNTIME_WALK_UNRESOLVED)
  if(NOT _unresolved)
    return()
  endif()
  set(_stage "hyle_runtime_visibility_${target}")
  set(_script "${CMAKE_CURRENT_BINARY_DIR}/hyle-runtime/${target}-visibility-$<CONFIG>.cmake")
  set(_content [=[
set(_unresolved [==[@_unresolved@]==])
list(FILTER _unresolved EXCLUDE REGEX "^$")
if(_unresolved)
  list(REMOVE_DUPLICATES _unresolved)
  list(JOIN _unresolved ", " _names)
  message(FATAL_ERROR
    "hyle_stage_runtime(@target@) cannot inspect scoped SDK target(s): ${_names}. "
    "Call find_package in the application directory or a common ancestor, or "
    "explicitly use find_package(... GLOBAL). Imported SDK targets must be "
    "visible where hyle_stage_runtime is called; refusing incomplete runtime staging.")
endif()
]=])
  string(CONFIGURE "${_content}" _content @ONLY)
  file(GENERATE OUTPUT "${_script}" CONTENT "${_content}" TARGET ${target})
  add_custom_target(${_stage}
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    VERBATIM)
  set_target_properties(${_stage} PROPERTIES FOLDER "hyle/runtime-staging")
  add_dependencies(${target} ${_stage})
  set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_VISIBILITY_TARGET ${_stage})
endfunction()

# Opt-in Windows DLL staging and Linux SDK loader paths. This file is shared by
# source embedding and installed SDKs; HYLE's own linkage does not describe a
# consumer's complete runtime closure (e.g. a separately compiled kernel DLL).
function(_hyle_stage_runtime_dependencies target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_runtime: target '${target}' does not exist")
  endif()
  get_target_property(_existing ${target} HYLE_RUNTIME_STAGING_TARGET)
  if(_existing)
    return()
  endif()

  if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Darwin)$")
    _hyle_collect_runtime_files("${target}" _runtime_files)
    _hyle_check_runtime_visibility(${target})
    if(_runtime_files)
      set(_runtime_paths "")
      foreach(_file IN LISTS _runtime_files)
        list(APPEND _runtime_paths "$<PATH:GET_PARENT_PATH,${_file}>")
      endforeach()
      set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH ${_runtime_paths})
      # These are build-host SDK locations. Deployment layout belongs to the
      # application: never bake them into its installed executable's RPATH.
      # RUNPATH is not transitive: the HYLE shared adapter, rather than the
      # executable, needs torch_cpu. Opt-in RPATH also covers its SDK-local
      # dependencies, without making a second SDK copy for every application.
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target} PRIVATE
          "$<$<BOOL:$<JOIN:${_runtime_paths},>>:LINKER:--disable-new-dtags>")
      endif()
    endif()
    set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET TRUE)
    return()
  elseif(NOT WIN32)
    return()
  endif()

  set(_stage "hyle_runtime_${target}")
  _hyle_collect_runtime_files("${target}" _extra_runtime_files)
  _hyle_check_runtime_visibility(${target})
  set(_script "${CMAKE_CURRENT_BINARY_DIR}/hyle-runtime/${target}-$<CONFIG>.cmake")
  set(_content [=[
set(_directory [==[$<TARGET_FILE_DIR:@target@>]==])
set(_dlls [==[$<TARGET_RUNTIME_DLLS:@target@>]==])
set(_extra_dlls [==[@_extra_runtime_files@]==])
list(APPEND _dlls ${_extra_dlls})
list(REMOVE_DUPLICATES _dlls)
file(MAKE_DIRECTORY "${_directory}")
# Multiple application targets can share a bin directory. Coordinate only
# their build-time writers, as the Apple resource staging helper does.
file(LOCK "${_directory}/.hyle-runtime-staging.lock" GUARD PROCESS TIMEOUT 60)
foreach(_dll IN LISTS _dlls)
  get_filename_component(_name "${_dll}" NAME)
  if(NOT _dll STREQUAL "${_directory}/${_name}")
    file(COPY_FILE "${_dll}" "${_directory}/${_name}" ONLY_IF_DIFFERENT)
  endif()
endforeach()
]=])
  string(CONFIGURE "${_content}" _content @ONLY)
  file(GENERATE OUTPUT "${_script}" CONTENT "${_content}" TARGET ${target})
  if(CMAKE_GENERATOR MATCHES "^Visual Studio ")
    # MSBuild's project references finish before this event. File dependencies
    # on transitive DLL paths alone do not establish cross-directory project
    # order. PRE_BUILD also runs when the linker keeps an unchanged EXE.
    add_custom_command(TARGET ${target} PRE_BUILD
      COMMAND "${CMAKE_COMMAND}" -P "${_script}"
      VERBATIM)
    set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET TRUE)
    return()
  endif()

  # No dependency on the executable itself: under CMP0112 NEW,
  # TARGET_FILE_DIR is a location, not an order edge. The stage depends on the
  # actual DLLs, so changing a DLL without changing its import library works.
  add_custom_target(${_stage}
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "$<TARGET_RUNTIME_DLLS:${target}>" ${_extra_runtime_files}
    VERBATIM)
  set_target_properties(${_stage} PROPERTIES FOLDER "hyle/runtime-staging")
  get_target_property(_visibility_stage ${target} HYLE_RUNTIME_VISIBILITY_TARGET)
  if(_visibility_stage)
    add_dependencies(${_stage} ${_visibility_stage})
  endif()
  add_dependencies(${target} ${_stage})
  set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET ${_stage})
endfunction()

# Same application API for embedded source and either installed SDK shape.
# The old individual helpers are private so consumers need no platform branch.
function(hyle_stage_runtime target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_runtime: target '${target}' does not exist")
  endif()
  get_target_property(_type ${target} TYPE)
  if(NOT _type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "hyle_stage_runtime requires an executable application target")
  endif()
  get_target_property(_existing ${target} HYLE_UNIFIED_RUNTIME_STAGING)
  if(_existing)
    return()
  endif()
  _hyle_stage_runtime_dependencies(${target})
  get_property(_assets GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_ASSETS)
  if(_assets)
    get_property(_producers GLOBAL PROPERTY HYLE_RUNTIME_WALK_APPLE_PRODUCERS)
    list(REMOVE_DUPLICATES _assets)
    list(REMOVE_DUPLICATES _producers)
    # An installed Apple SDK may already provide this helper in a sibling
    # scope after a CPU-only SDK first defined the common staging API.
    if(NOT COMMAND _hyle_stage_apple_runtime)
      include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HyleAppleRuntime.cmake")
    endif()
    _hyle_stage_apple_runtime(${target} "${_assets}" "${_producers}")
  endif()
  set_property(TARGET ${target} PROPERTY HYLE_UNIFIED_RUNTIME_STAGING TRUE)
endfunction()
