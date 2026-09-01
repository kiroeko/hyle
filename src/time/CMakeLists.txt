# time: C++ monotonic/realtime reads stay header-inline over std::chrono (one
# system-clock hop); sleep/coarse/TSC paths live in clock.cc. Timer structures
# use the lightweight container_intrusive internal factor. Core supplies the
# only other public source contract; time has no recoverable Status-returning
# edge.
hyle_add_module(time TESTING SOURCES clock.cc)
