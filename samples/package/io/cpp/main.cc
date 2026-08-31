#include <hyle/io/capabilities.h>
#include <hyle/io/http.h>
#include <hyle/io/net.h>
#include <hyle/io/shared_memory.h>

#if HYLE_IO_HAS_COROUTINE_TRANSPORT != 1
#  error "supported installed io package lost its coroutine-transport capability"
#endif

int main() {
  // Public io headers expose Status and allocator-backed protocol values.  The
  // consumer deliberately links only hyle::io.
  auto status = hyle::io::SharedMemory::unlink(nullptr);
  if (!status.IsInvalidArgument()) return 1;

  if (!hyle::mem::system_allocator().valid()) return 2;
  hyle::io::StringView body{"installed io payload"};
  if (body.find("payload") == hyle::io::StringView::npos) return 3;

  // complete() is header-inline and calls Waker::wake().  The harmless public
  // scheduler query proves the same hyle::io usage requirement is propagated
  // to an installed consumer and staged at runtime.
  if (hyle::sched::current_worker_index() != -1) return 4;
  using CompleteFn = void (*)(const hyle::io::Completion&) noexcept;
  CompleteFn volatile complete = &hyle::io::complete;
  return complete == nullptr ? 5 : 0;
}
