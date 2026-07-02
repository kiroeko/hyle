# io-01/02: proactor (completion-reaping seam) + backends. Per-OS in proactor.cc
# (Windows IOCP built+tested now; Linux io_uring next, borrowing liburing's
# syscall shim). Binds to sched via the Waker spine -- io/ only moves completion
# tokens, sched/ owns the wake. Depends on platform (HYLE_OS_* macros).
hyle_add_module(io SOURCES proactor.cc DEPS hyle_platform)

if(WIN32)
  # IOCP (CreateIoCompletionPort / GetQueuedCompletionStatusEx) is in kernel32,
  # already linked by default -- no extra lib needed for the completion seam.
  # (Winsock ws2_32 comes in with net/ sockets at io-04, not here.)
endif()
