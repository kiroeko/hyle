# time: host clock/value semantics. C++ monotonic/realtime reads stay
# header-inline over std::chrono (one system-clock hop); sleep/coarse/TSC paths
# live in clock.cc. Caller-owned timer nodes live in the separate header-only
# timer_queue factor, so timestamp-only consumers do not acquire intrusive
# storage. Core is the only HYLE dependency; there is no recoverable Status edge.
hyle_add_module(time TESTING SOURCES clock.cc interop.cc)
