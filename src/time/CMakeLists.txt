# time: C++ monotonic/realtime reads stay header-inline over std::chrono (one
# system-clock hop); sleep/coarse/TSC paths live in clock.cc. Timer structures
# use the lightweight container_intrusive component. Core supplies header-only
# compiler contracts; recoverable boundary failures use the shared error value.
hyle_add_module(time SOURCES clock.cc c_time.cc)
