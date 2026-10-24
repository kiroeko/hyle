cmake_minimum_required(VERSION 3.25)
project(hyle_qwen35_text LANGUAGES CXX)

# Standalone consumer: use the installed SDK's CUDA or native Apple/Vulkan route.
# Configuring this directory never configures or rebuilds native dependencies.
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute graph graph_progress compute_stream mem)
set(_qwen35_native FALSE)
if(APPLE AND "apple" IN_LIST Hyle_AVAILABLE_COMPUTE_ROUTES)
  set(_qwen35_native TRUE)
elseif("cuda" IN_LIST Hyle_AVAILABLE_COMPUTE_ROUTES)
  # Matches Backend's preference when the SDK exports both GPU routes.
elseif("vulkan" IN_LIST Hyle_AVAILABLE_COMPUTE_ROUTES)
  set(_qwen35_native TRUE)
else()
  message(FATAL_ERROR "Qwen text requires an Apple, CUDA or Vulkan route in the installed Hyle SDK")
endif()
add_executable(hyle_qwen35_text main.cc)
target_compile_features(hyle_qwen35_text PRIVATE cxx_std_23)
target_link_libraries(hyle_qwen35_text PRIVATE hyle::graph hyle::compute hyle::graph_progress hyle::compute_stream hyle::mem)
target_compile_options(hyle_qwen35_text PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/utf-8;/W4;/WX>)
hyle_stage_runtime(hyle_qwen35_text)

# Prefix checks use actual result owners; retirement instruments the original
# native queue/stream. Both remain explicit real-weight diagnostic targets.
set(_qwen35_probes mtp_session retirement)
if(_qwen35_native)
  # Reject actual completed generated-state export above its finite quota,
  # preserve the old whole state, then retry after releasing the pressure.
  list(APPEND _qwen35_probes retained_budget)
else()
  list(APPEND _qwen35_probes batch)
endif()
foreach(probe IN LISTS _qwen35_probes)
  add_executable(hyle_qwen35_${probe}_probe EXCLUDE_FROM_ALL ${probe}_probe.cc)
  target_compile_features(hyle_qwen35_${probe}_probe PRIVATE cxx_std_23)
  target_link_libraries(hyle_qwen35_${probe}_probe PRIVATE hyle::graph hyle::compute hyle::graph_progress hyle::compute_stream hyle::mem)
  target_compile_options(hyle_qwen35_${probe}_probe PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8;/W4;/WX>)
  if(probe STREQUAL "retirement")
    if(APPLE)
      enable_language(OBJCXX)
      target_sources(hyle_qwen35_retirement_probe PRIVATE native_gate_apple.mm)
      set_target_properties(hyle_qwen35_retirement_probe PROPERTIES
        OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
    else()
      target_sources(hyle_qwen35_retirement_probe PRIVATE native_gate.cc)
    endif()
  endif()
  hyle_stage_runtime(hyle_qwen35_${probe}_probe)
endforeach()
unset(_qwen35_probes)
unset(_qwen35_native)
