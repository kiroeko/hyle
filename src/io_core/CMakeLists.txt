# io proactor (completion-reaping seam) + real IO backends. Per-OS in proactor.cc:
# Windows IOCP (ReadFile/WriteFile/WSARecv/WSASend/AcceptEx/ConnectEx) and Linux
# io_uring (raw syscalls, no liburing link) + epoll reactor fallback. Binds to
# sched via the Waker spine -- proactor.cc only moves completion tokens, sched/
# owns the wake. `hyle/core/environment.h` contributes compile-time HYLE_OS_* macros but
# no platform library symbol/type; the native async backends therefore do not
# claim a direct binary dependency on the synchronous platform module.
#
# io_context.cc is the io<->sched integration (io-04): it plugs the proactor into
# the sched worker loop, so it depends on hyle_sched. sched only exists where
# fcontext does (HYLE_HAS_FCONTEXT), so io_context.cc is compiled into the module
# only there. proactor.cc itself never depends on sched, but the current public
# io component does; a standalone proactor would require a future real component
# split rather than a conditional omission from the dependency graph.
set(_io_sources proactor.cc endpoint.cc operation.cc io_uring.cc)
if(HYLE_HAS_FCONTEXT)
  # The co_await layers (net/tcp/udp/file/signal) and the DNS offload need sched
  # (the scheduler drives the reap; dns uses spawn_blocking). Only buildable where
  # fcontext (hence sched) exists. The root rejects io where it does not.
  list(APPEND _io_sources io_context.cc tcp.cc udp.cc file.cc local.cc signal.cc
       dns.cc)
endif()

set(_io_module_options "")
if(HYLE_HAS_FCONTEXT)
  list(APPEND _io_module_options TRACE)
endif()
hyle_add_module(io_core ${_io_module_options} SOURCES ${_io_sources})
unset(_io_module_options)

# Export the exact coroutine-transport product fact on hyle::io in both
# source-embedded and installed-package use. capabilities.h turns a target/header
# mismatch into an actionable diagnostic; the raw proactor remains an unsupported
# detail seam. Named mapped storage is owned by mem::SharedMapping and reaches the
# same Platform mechanism.
if(HYLE_HAS_FCONTEXT)
  target_compile_definitions(hyle_io_core
    PUBLIC HYLE_IO_HAS_COROUTINE_TRANSPORT=1)
else()
  target_compile_definitions(hyle_io_core
    PUBLIC HYLE_IO_HAS_COROUTINE_TRANSPORT=0)
endif()

if(WIN32)
  # IOCP (CreateIoCompletionPort / GetQueuedCompletionStatusEx) is in kernel32,
  # linked by default. The socket ops (WSARecv/WSASend + AcceptEx/ConnectEx via
  # WSAIoctl) need Winsock: ws2_32 for the core calls, mswsock for the extension
  # GUIDs. File read/write need neither, but linking both is harmless and keeps
  # the single proactor TU whole.
  target_link_libraries(hyle_io_core PRIVATE ws2_32 mswsock)
endif()
