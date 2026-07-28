# io echo bench (io-04): the concurrent-connection perf comparison the io-01/02/03
# notes deferred until there was a real load. Drives kConns x kRoundTrips echoes
# through HYLE's shard-per-core stack at 1/2/4/8 workers (scaling) + a raw epoll
# readiness-loop baseline (Linux). The io product target already closes over sched
# and its fcontext implementation. Needs ws2_32 on Windows (socketpair emulation
# via loopback TCP).
  hyle_add_bench(io SOURCES echo_bench.cc DEPS hyle_sched)
  if(WIN32)
    target_link_libraries(hyle_io_bench PRIVATE ws2_32)
  endif()

  # Optional SOTA parity: echo_bench's HYLE rows vs a hand-rolled raw-liburing
  # echo on the same load. liburing is deliberately a prebuilt dependency:
  # configuring HYLE never runs a foreign build system or writes into a peer's
  # source checkout. Point HYLE_LIBURING_ROOT at an installed prefix, or pass
  # HYLE_LIBURING_INCLUDE_DIR and HYLE_LIBURING_LIBRARY explicitly.
  option(HYLE_BENCH_LIBURING "build echo bench vs raw liburing (io_uring SOTA)" OFF)

  # Optional SOTA parity #2: echo_bench's HYLE rows vs asio + libuv. Asio is
  # header-only, but its io_uring backend consumes the same prebuilt liburing
  # contract as the raw-liburing row.
  option(HYLE_BENCH_ASIO "build echo bench vs asio (io_uring, async-framework SOTA)" OFF)
  if((HYLE_BENCH_LIBURING OR HYLE_BENCH_ASIO) AND UNIX AND NOT APPLE)
    set(HYLE_LIBURING_ROOT "" CACHE PATH
        "installed liburing prefix used only by io benchmarks")
    if(HYLE_LIBURING_ROOT)
      find_path(HYLE_LIBURING_INCLUDE_DIR
        NAMES liburing.h
        HINTS "${HYLE_LIBURING_ROOT}"
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
      find_library(HYLE_LIBURING_LIBRARY
        NAMES uring liburing
        HINTS "${HYLE_LIBURING_ROOT}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)
    else()
      find_path(HYLE_LIBURING_INCLUDE_DIR NAMES liburing.h)
      find_library(HYLE_LIBURING_LIBRARY NAMES uring liburing)
    endif()
    if(NOT EXISTS "${HYLE_LIBURING_INCLUDE_DIR}/liburing.h" OR
       NOT EXISTS "${HYLE_LIBURING_LIBRARY}")
      message(FATAL_ERROR
        "liburing benchmark support requires a prebuilt liburing. Install it and "
        "set HYLE_LIBURING_ROOT, or set HYLE_LIBURING_INCLUDE_DIR and "
        "HYLE_LIBURING_LIBRARY explicitly; HYLE will not build inside the peer "
        "source checkout.")
    endif()
    add_library(hyle_bench_liburing UNKNOWN IMPORTED)
    set_target_properties(hyle_bench_liburing PROPERTIES
      IMPORTED_LOCATION "${HYLE_LIBURING_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${HYLE_LIBURING_INCLUDE_DIR}")
  endif()

  if(HYLE_BENCH_LIBURING AND UNIX AND NOT APPLE)
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_LIBURING)
    target_link_libraries(hyle_io_bench PRIVATE hyle_bench_liburing)
  endif()

  if(HYLE_BENCH_ASIO AND UNIX AND NOT APPLE)
    set(HYLE_ASIO_DIR "${HYLE_SOURCE_DIR}/../asio" CACHE PATH "cloned asio (Kohlhoff)")
    if(NOT EXISTS "${HYLE_ASIO_DIR}/asio/include/asio.hpp")
      message(FATAL_ERROR "HYLE_BENCH_ASIO=ON but no asio at ${HYLE_ASIO_DIR}/asio/include")
    endif()
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_ASIO
      ASIO_HAS_IO_URING ASIO_DISABLE_EPOLL ASIO_STANDALONE)
    target_include_directories(hyle_io_bench SYSTEM PRIVATE "${HYLE_ASIO_DIR}/asio/include")
    target_link_libraries(hyle_io_bench PRIVATE hyle_bench_liburing)
  endif()

  option(HYLE_BENCH_LIBUV "build echo bench vs libuv (epoll, async-runtime SOTA)" OFF)
  if(HYLE_BENCH_LIBUV AND UNIX AND NOT APPLE)
    set(HYLE_LIBUV_DIR "${HYLE_SOURCE_DIR}/../libuv" CACHE PATH "cloned libuv")
    if(NOT EXISTS "${HYLE_LIBUV_DIR}/include/uv.h")
      message(FATAL_ERROR "HYLE_BENCH_LIBUV=ON but no libuv at ${HYLE_LIBUV_DIR}")
    endif()
    # libuv is a CMake project; keep all generated state in HYLE's build tree.
    if(NOT TARGET uv_a)
      set(LIBUV_BUILD_TESTS OFF)
      set(LIBUV_BUILD_BENCH OFF)
      add_subdirectory(
        "${HYLE_LIBUV_DIR}"
        "${HYLE_BINARY_DIR}/bench/peers/libuv"
        EXCLUDE_FROM_ALL)
    endif()
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_LIBUV)
    target_include_directories(hyle_io_bench SYSTEM PRIVATE "${HYLE_LIBUV_DIR}/include")
    target_link_libraries(hyle_io_bench PRIVATE uv_a)
  endif()

  # io scaling bench: the two COMPLEXITY curves echo_bench's fixed 64-conn/64B load
  # can't show -- connection-count scaling (ns/round-trip flat => O(1)/conn, the
  # "proactor beats readiness" argument) + payload-size scaling (bytes/s clean plateau
  # => no hidden copy in the co_await path). Own exe (echo_bench owns main). Same sched
  # dep + ws2_32 on Windows (socketpair emulation via loopback TCP).
  add_executable(hyle_io_scaling_bench io_scaling_bench.cc)
  target_link_libraries(hyle_io_scaling_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_io_scaling_bench)
  if(WIN32)
    target_link_libraries(hyle_io_scaling_bench PRIVATE ws2_32)
  endif()

  # io accept-scaling bench: isolates CONNECTION ESTABLISHMENT (the accept path the
  # echo benches never time -- they pre-build sockets). One TcpListener accepts N
  # conns while N client fibers connect; reports accepts/s swept over N to show the
  # accept path is O(1)/connection. Real TCP over loopback (accept needs a listener),
  # so hyle_io (tcp wrappers) + ws2_32 on Windows.
  add_executable(hyle_accept_bench accept_bench.cc)
  target_link_libraries(hyle_accept_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_accept_bench)
  if(WIN32)
    target_link_libraries(hyle_accept_bench PRIVATE ws2_32)
  endif()

  # io connection-lifecycle bench: the REALISTIC full-connection cost (accept + serve R
  # round-trips + close) that echo_bench (pre-built socketpairs, never accept/close) and
  # accept_bench (accept-then-drop, no service) each factor out. Sweeps rounds/conn =
  # 1/4/16 to walk the ephemeral (setup-dominated) -> persistent (steady-state) spectrum.
  # Real loopback TCP (real handshake + close), so hyle_io (tcp wrappers) + ws2_32 on Win.
  add_executable(hyle_conn_lifecycle_bench conn_lifecycle_bench.cc)
  target_link_libraries(hyle_conn_lifecycle_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_conn_lifecycle_bench)
  if(WIN32)
    target_link_libraries(hyle_conn_lifecycle_bench PRIVATE ws2_32)
  endif()

  # io HTTP bench (io-10): request/response throughput of the FULL HTTP/1.1 path
  # (picohttpparser parse + GrowBuf head/body copies + Request/Response build + status
  # /header formatting) vs a RAW TCP echo over the SAME co_await TcpStream transport
  # moving the IDENTICAL byte volume with no HTTP work. The ratio isolates the HTTP
  # parse/framing layer cost from the io transport (already measured by echo_bench),
  # and the 1/2/4/8-worker sweep shows the HTTP path preserves shard-per-core scaling.
  # Socketpair transport (like echo_bench) -- no accept, no TIME_WAIT. Needs hyle_io
  # (http/tcp) + ws2_32 on Windows.
  add_executable(hyle_http_bench http_bench.cc)
  target_link_libraries(hyle_http_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_http_bench)
  if(WIN32)
    target_link_libraries(hyle_http_bench PRIVATE ws2_32 psapi)
  endif()

  # io multishot-recv bench (io-07b): does the multishot + provided-buffer-ring
  # awaitable beat one-shot RecvOp on a receive-heavy load? Same-worker sender+receiver
  # fibers + DGRAM socketpair (no cross-thread wakeup floor, equal completion counts);
  # the only variable is the receiver's recv method. Isolates the per-recv-submit cost
  # multishot removes. Linux/io_uring only in effect (register_buf_ring returns false
  # elsewhere -> multishot row self-skips); compiles everywhere. Same sched dep.
  add_executable(hyle_multishot_bench multishot_bench.cc)
  target_link_libraries(hyle_multishot_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_multishot_bench)
  if(WIN32)
    target_link_libraries(hyle_multishot_bench PRIVATE ws2_32)
  endif()

  # io multishot ZERO-COPY ECHO bench (io-07c承重): the workload multishot is FOR --
  # echo each received chunk straight back FROM the ring buffer (no copy) vs a one-shot
  # recv-into-buffer + send. Where the recv-drain bench tied, this should show
  # multishot's zero-copy + no-per-recv-submit win. Same sched dep; Linux/io_uring only.
  add_executable(hyle_multishot_echo_bench multishot_echo_bench.cc)
  target_link_libraries(hyle_multishot_echo_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_multishot_echo_bench)
  if(WIN32)
    target_link_libraries(hyle_multishot_echo_bench PRIVATE ws2_32)
  endif()

  # io TLS bench (io-08): TLS handshake/s + encrypted-echo throughput over HYLE's
  # memory-BIO pump vs RAW OpenSSL over a blocking socket (same cert/cipher/payload).
  # The ratio isolates HYLE's co_await pump + fiber overhead from the identical
  # OpenSSL crypto floor both pay. Only when the resolved TLS capability is enabled
  # (same gate as the tls.cc build). Links OpenSSL + ws2_32 on Windows.
  if(HYLE_IO_TLS_ENABLED)
    add_executable(hyle_io_tls_bench tls_bench.cc)
    target_link_libraries(hyle_io_tls_bench PRIVATE hyle_sched hyle_io nanobench
                          OpenSSL::SSL OpenSSL::Crypto)
    hyle_apply_warnings(hyle_io_tls_bench)
    if(WIN32)
      target_link_libraries(hyle_io_tls_bench PRIVATE ws2_32)
    endif()
  endif()

hyle_register_benches(io
  hyle_io_scaling_bench
  hyle_accept_bench
  hyle_conn_lifecycle_bench
  hyle_http_bench
  hyle_multishot_bench
  hyle_multishot_echo_bench)
if(TARGET hyle_io_tls_bench)
  hyle_register_benches(io hyle_io_tls_bench)
endif()
