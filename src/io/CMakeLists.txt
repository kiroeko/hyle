# io proactor (completion-reaping seam) + real IO backends. Per-OS in proactor.cc:
# Windows IOCP (ReadFile/WriteFile/WSARecv/WSASend/AcceptEx/ConnectEx) and Linux
# io_uring (raw syscalls, no liburing link) + epoll reactor fallback. Binds to
# sched via the Waker spine -- proactor.cc only moves completion tokens, sched/
# owns the wake. `hyle/platform.h` contributes compile-time HYLE_OS_* macros but
# no platform library symbol/type; the native async backends therefore do not
# claim a direct binary dependency on the synchronous platform module.
#
# io_context.cc is the io<->sched integration (io-04): it plugs the proactor into
# the sched worker loop, so it depends on hyle_sched. sched only exists where
# fcontext does (HYLE_HAS_FCONTEXT), so io_context.cc is compiled into the module
# only there -- proactor.cc itself never depends on sched, keeping the proactor
# usable on any platform.
set(_io_sources proactor.cc c_io.cc shared_memory.cc)
if(HYLE_HAS_FCONTEXT)
  # The co_await layers (net/tcp/udp/file/signal) and the DNS offload need sched
  # (the scheduler drives the reap; dns uses spawn_blocking). Only buildable where
  # fcontext (hence sched) exists. proactor.cc itself never depends on sched.
  list(APPEND _io_sources io_context.cc tcp.cc udp.cc file.cc signal.cc dns.cc)
  # io-10 HTTP/1.1: integration layer (read loop -> feed parser -> Request/Response
  # -> write) over co_await TcpStream, so it needs sched too. Protocol PARSING is
  # borrowed from vendored picohttpparser (pure state machine, zero IO/threads) --
  # HYLE only writes the integration. See docs scope "HTTP 最终结论".
  list(APPEND _io_sources http.cc "${PROJECT_SOURCE_DIR}/third_party/picohttpparser/picohttpparser.c")
  # io-11 HTTP/2: nghttp2 supplies framing/HPACK/stream+flow-control state only;
  # http2.cc supplies HYLE's TcpStream/TlsStream coroutine transport.
  if(HYLE_IO_HTTP2)
    list(APPEND _io_sources http2.cc)
  else()
    # Keep the public C ABI link-stable when the optional protocol core is
    # disabled.  The stub reports availability=0 and fails every operation;
    # consumers can ship one binary against HTTP/2-enabled and minimal HYLE
    # builds without conditional symbol lookup.
    list(APPEND _io_sources http2_stub.cc)
  endif()
  # io-08 TLS: only when OpenSSL is available (optional capability). TlsStream is a
  # co_await layer over TcpStream, so it also needs sched. Guarded by HYLE_IO_TLS so
  # tls.h is inert without OpenSSL and a TLS-free build carries no OpenSSL dep.
  if(HYLE_IO_TLS_ENABLED)
    list(APPEND _io_sources tls.cc)
  endif()

  hyle_register_package_dependency(
    NAME picohttpparser
    OWNER io
    LOCKED_DEPENDENCIES picohttpparser
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/picohttpparser/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|picohttpparser-LICENSE")
endif()

# Platforms without fcontext do not build the coroutine transport or the real
# HTTP/2 session. They still export the feature-probe/stub C ABI.
if(NOT HYLE_HAS_FCONTEXT)
  list(APPEND _io_sources http2_stub.cc)
endif()
hyle_add_module(io SOURCES ${_io_sources})

# Public headers are installed as one reviewed tree, so capability cannot be
# inferred from header presence.  Export the exact product fact on hyle::io in
# both source-embedded and installed-package use; net.h turns value 0 into an
# actionable diagnostic while proactor/shared_memory and the C stub stay valid.
if(HYLE_HAS_FCONTEXT)
  target_compile_definitions(hyle_io
    PUBLIC HYLE_IO_HAS_COROUTINE_TRANSPORT=1)
else()
  target_compile_definitions(hyle_io
    PUBLIC HYLE_IO_HAS_COROUTINE_TRANSPORT=0)
endif()

if(HYLE_HAS_FCONTEXT)
  # picohttpparser is a vendored single-file C parser. Include its dir as SYSTEM
  # (suppress its warnings under our /W4|-Wall) and silence the .c TU's own warnings,
  # exactly like simdutf.cpp in src/container. It has an optional SSE4.2 fast path
  # gated on __SSE4_2__; the project-global -mssse3 doesn't enable it, so it uses the
  # portable scalar scan (fine -- header parsing isn't the bottleneck).
  target_include_directories(hyle_io SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/picohttpparser")
  if(MSVC)
    set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/picohttpparser/picohttpparser.c"
      PROPERTIES COMPILE_OPTIONS "/w")
  else()
    set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/picohttpparser/picohttpparser.c"
      PROPERTIES COMPILE_OPTIONS "-w")
  endif()
endif()

if(HYLE_HAS_FCONTEXT AND HYLE_IO_HTTP2)
  target_link_libraries(hyle_io PRIVATE nghttp2)
  target_compile_definitions(hyle_io PUBLIC HYLE_IO_HTTP2)
  # MSVC has no POSIX ssize_t. nghttp2's modern `nghttp2_ssize` API remains
  # available; this only hides its deprecated ssize_t overloads.
  if(WIN32)
    target_compile_definitions(hyle_io PRIVATE NGHTTP2_NO_SSIZE_T)
  endif()
endif()

if(HYLE_HAS_FCONTEXT AND HYLE_IO_TLS_ENABLED)
  # Dynamic-link OpenSSL (security lib tracks system CVE updates). PUBLIC define so
  # consumers including tls.h see the real TlsStream; PRIVATE link (impl detail).
  target_link_libraries(hyle_io PRIVATE OpenSSL::SSL OpenSSL::Crypto)
  target_compile_definitions(hyle_io PUBLIC HYLE_IO_TLS)
  hyle_register_package_dependency(
    NAME openssl
    OWNER io
    FEATURES io_tls
    STATIC_FIND_DEPENDENCIES OpenSSL)
endif()

if(WIN32)
  # IOCP (CreateIoCompletionPort / GetQueuedCompletionStatusEx) is in kernel32,
  # linked by default. The socket ops (WSARecv/WSASend + AcceptEx/ConnectEx via
  # WSAIoctl) need Winsock: ws2_32 for the core calls, mswsock for the extension
  # GUIDs. File read/write need neither, but linking both is harmless and keeps
  # the single proactor TU whole.
  target_link_libraries(hyle_io PRIVATE ws2_32 mswsock)
endif()
