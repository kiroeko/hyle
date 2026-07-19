# time: C++ monotonic/realtime reads stay header-inline over std::chrono (one
# system-clock hop); sleep/coarse/TSC paths live in clock.cc. Timer structures
# use the zero-dependency intrusive primitive from core/. No compiled HYLE deps.
hyle_add_module(time SOURCES clock.cc c_time.cc)
