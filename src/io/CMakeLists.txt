# io proactor (completion-reaping seam) + real IO backends. Per-OS in proactor.cc:
# Windows IOCP (ReadFile/WriteFile/WSARecv/WSASend/AcceptEx/ConnectEx) and Linux
# io_uring (raw syscalls, no liburing link) + epoll reactor fallback. Binds to
# sched via the Waker spine -- proactor.cc only moves completion tokens, sched/
# owns the wake. Depends on platform (HYLE_OS_* macros).
#
# io_context.cc is the io<->sched integration (io-04): it plugs the proactor into
# the sched worker loop, so it depends on hyle_sched. sched only exists where
# fcontext does (HYLE_HAS_FCONTEXT), so io_context.cc is compiled into the module
# only there -- proactor.cc itself never depends on sched, keeping the proactor
# usable on any platform.
set(_io_sources proactor.cc)
set(_io_deps hyle_platform)
if(HYLE_HAS_FCONTEXT)
  # The co_await layers (net/tcp/udp/file/signal) and the DNS offload need sched
  # (the scheduler drives the reap; dns uses spawn_blocking). Only buildable where
  # fcontext (hence sched) exists. proactor.cc itself never depends on sched.
  list(APPEND _io_sources io_context.cc tcp.cc udp.cc file.cc signal.cc dns.cc)
  list(APPEND _io_deps hyle_sched)
  # io-08 TLS: only when OpenSSL is available (optional capability). TlsStream is a
  # co_await layer over TcpStream, so it also needs sched. Guarded by HYLE_IO_TLS so
  # tls.h is inert without OpenSSL and a TLS-free build carries no OpenSSL dep.
  if(OpenSSL_FOUND)
    list(APPEND _io_sources tls.cc)
  endif()
endif()
hyle_add_module(io SOURCES ${_io_sources} DEPS ${_io_deps})

if(HYLE_HAS_FCONTEXT AND OpenSSL_FOUND)
  # Dynamic-link OpenSSL (security lib tracks system CVE updates). PUBLIC define so
  # consumers including tls.h see the real TlsStream; PRIVATE link (impl detail).
  target_link_libraries(hyle_io PRIVATE OpenSSL::SSL OpenSSL::Crypto)
  target_compile_definitions(hyle_io PUBLIC HYLE_IO_TLS)
endif()

if(WIN32)
  # IOCP (CreateIoCompletionPort / GetQueuedCompletionStatusEx) is in kernel32,
  # linked by default. The socket ops (WSARecv/WSASend + AcceptEx/ConnectEx via
  # WSAIoctl) need Winsock: ws2_32 for the core calls, mswsock for the extension
  # GUIDs. File read/write need neither, but linking both is harmless and keeps
  # the single proactor TU whole.
  target_link_libraries(hyle_io PRIVATE ws2_32 mswsock)
endif()
