# cpu.cc is the CPUID PROBE translation unit and MUST stay LTO-free (conventions
# §10 / llama.cpp's hard-won bug): otherwise the linker may inline an advanced-
# ISA instruction into the detector and SIGILL before it can report an old CPU.
# A private object target gives the probe a real no-IPO boundary without noisy
# `/GL /GL-` or `-flto -fno-lto` option overrides. The rest of hyle_platform
# still inherits a shipping parent's IPO setting.
add_library(hyle_platform_cpu_probe OBJECT cpu.cc)
target_include_directories(hyle_platform_cpu_probe PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(hyle_platform_cpu_probe PRIVATE cxx_std_23)
target_link_libraries(hyle_platform_cpu_probe PRIVATE hyle_build_contract)
target_compile_definitions(hyle_platform_cpu_probe PRIVATE HYLE_PLATFORM_BUILD)
if(HYLE_STATIC)
  target_compile_definitions(hyle_platform_cpu_probe PRIVATE HYLE_STATIC)
endif()
hyle_apply_warnings(hyle_platform_cpu_probe)
hyle_apply_sanitizer(hyle_platform_cpu_probe)
set_target_properties(hyle_platform_cpu_probe PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION OFF
  POSITION_INDEPENDENT_CODE ON
  FOLDER "hyle/modules/private")

hyle_add_module(platform TESTING SOURCES
  wait.cc thread.cc dynlib.cc fs.cc virtual_memory.cc numa.cc mapped_file.cc
  c_platform.cc $<TARGET_OBJECTS:hyle_platform_cpu_probe>)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib (API set).
if(WIN32)
  target_link_libraries(hyle_platform PRIVATE Synchronization)
endif()

# dlopen/dlsym live in libdl on Linux (folded into libc on macOS/BSD).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(hyle_platform PRIVATE ${CMAKE_DL_LIBS})
endif()
