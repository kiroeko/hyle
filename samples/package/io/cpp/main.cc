#include <hyle/io/capabilities.h>
#include <hyle/io/endpoint.h>
#include <hyle/io/http.h>
#include <hyle/mem/mapping.h>

#if HYLE_IO_HAS_COROUTINE_TRANSPORT != 1
#  error "supported installed io package lost its coroutine-transport capability"
#endif

int main() {
  // Public io headers expose Status and allocator-backed protocol values.  The
  // consumer deliberately links only hyle::io.
  auto status = hyle::mem::SharedMapping::unlink(nullptr);
  if (!status.IsInvalidArgument()) return 1;

  if (!hyle::mem::system_allocator().valid()) return 2;
  hyle::io::StringView body{"installed io payload"};
  if (body.find("payload") == hyle::io::StringView::npos) return 3;

  // A typed endpoint parse crosses the io_core binary boundary without exposing
  // native sockaddr storage, completion tokens, or the backend proactor.
  if (hyle::sched::current_worker_index() != -1) return 4;
  auto endpoint = hyle::io::Endpoint::parse("127.0.0.1", 2048);
  if (!endpoint.has_value() || !endpoint->is_loopback()) return 5;
  hyle::io::IoCapabilities capabilities;
  return capabilities.backend_name() == "none" ? 0 : 6;
}
