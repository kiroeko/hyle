#pragma once

#include "hyle/io/capabilities.h"

#if !HYLE_IO_HAS_COROUTINE_TRANSPORT
#  error "This HYLE io product has no coroutine transport (sched/fcontext). Use hyle/io/proactor.h or hyle/io/shared_memory.h; net/tcp/udp/file/dns/signal/http C++ APIs are unavailable. The pre-release Native FFI HTTP/2 candidate, when built, only reports unavailable."
#endif
