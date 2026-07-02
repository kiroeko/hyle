# io proactor (completion-reaping seam) + real IO backends. Per-OS in proactor.cc:
# Windows IOCP (ReadFile/WriteFile/WSARecv/WSASend/AcceptEx/ConnectEx) and Linux
# io_uring (raw syscalls, no liburing link) + epoll reactor fallback. Binds to
# sched via the Waker spine -- io/ only moves completion tokens, sched/ owns the
# wake. Depends on platform (HYLE_OS_* macros).
hyle_add_module(io SOURCES proactor.cc DEPS hyle_platform)

if(WIN32)
  # IOCP (CreateIoCompletionPort / GetQueuedCompletionStatusEx) is in kernel32,
  # linked by default. The socket ops (WSARecv/WSASend + AcceptEx/ConnectEx via
  # WSAIoctl) need Winsock: ws2_32 for the core calls, mswsock for the extension
  # GUIDs. File read/write need neither, but linking both is harmless and keeps
  # the single proactor TU whole.
  target_link_libraries(hyle_io PRIVATE ws2_32 mswsock)
endif()
