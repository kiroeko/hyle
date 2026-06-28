# time: monotonic + realtime clock (ns) + sleep. Per-OS in clock.cc (POSIX
# clock_gettime / Windows QPC+FILETIME). Zero HYLE deps -- raw syscalls only.
hyle_add_module(time SOURCES clock.cc)
