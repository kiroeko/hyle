hyle_add_module(platform SOURCES wait.cc thread.cc dynlib.cc fs.cc cpu.cc virtual_memory.cc numa.cc c_platform.cc PUBLIC_DEPS hyle_error)

# cpu.cc is the CPUID PROBE translation unit and MUST stay LTO-free (conventions §10 /
# llama.cpp's hard-won bug): with LTO on, the linker can inline an advanced-ISA
# instruction INTO the detector, which then SIGILLs on an old CPU before it can report
# that the CPU is old. platform is compiled entirely at the baseline ISA (no -march TU),
# so there is no wide instruction to inline today; this defensively forces IPO/LTO off
# for the whole module so a future global -flto can't break the probe.
set_target_properties(hyle_platform PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib (API set).
if(WIN32)
  target_link_libraries(hyle_platform PRIVATE Synchronization)
endif()

# dlopen/dlsym live in libdl on Linux (folded into libc on macOS/BSD).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(hyle_platform PRIVATE ${CMAKE_DL_LIBS})
endif()
