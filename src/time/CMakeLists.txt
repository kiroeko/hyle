# time: C++ monotonic/realtime reads stay header-inline over std::chrono (one
# system-clock hop); sleep/coarse/TSC paths live in clock.cc. Timer structures
# use the zero-dependency intrusive primitive from core/. Recoverable C-boundary
# failures use the project-wide error value; time does not invent TLS diagnostics.
hyle_add_module(time SOURCES clock.cc c_time.cc PUBLIC_DEPS hyle_error)
