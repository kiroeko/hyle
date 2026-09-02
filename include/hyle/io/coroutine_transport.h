#pragma once

#include "hyle/io/capabilities.h"

#if !HYLE_IO_HAS_COROUTINE_TRANSPORT
#  error "This HYLE io product has no coroutine transport (sched/fcontext). Use hyle/io/proactor.h for raw completion IO or hyle/mem/mapping.h for named mapped storage; net/tcp/udp/file/dns/signal/http C++ APIs are unavailable."
#endif
