#pragma once

#include "hyle/io/capabilities.h"

#if !HYLE_IO_HAS_COROUTINE_TRANSPORT
#  error "This HYLE io product has no coroutine transport (sched/fcontext). Use hyle/io/proactor.h, hyle/io/shared_memory.h, or the stable C HTTP/2 availability/stub surface; net/tcp/udp/file/dns/signal/http C++ APIs are unavailable."
#endif

