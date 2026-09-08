#include <hyle/io/capabilities.h>
#include <hyle/io/endpoint.h>
#include <hyle/io/http.h>
#include <hyle/io/transfer.h>
#if defined(HYLE_IO_TLS)
#include <hyle/io/tls.h>
#endif
#include <hyle/mem/mapping.h>

#if HYLE_IO_HAS_COROUTINE_TRANSPORT != 1
#error "supported installed io package lost its coroutine-transport capability"
#endif

int main() {
  // Public io headers expose Status and allocator-backed protocol values.  The
  // consumer deliberately links only hyle::io.
  auto status = hyle::mem::SharedMapping::unlink(nullptr);
  if (!status.IsInvalidArgument())
    return 1;

  if (!hyle::mem::system_allocator().valid())
    return 2;
  hyle::io::StringView body{"installed io payload"};
  if (body.find("payload") == hyle::io::StringView::npos)
    return 3;

  // A typed endpoint parse crosses the io_core binary boundary without exposing
  // native sockaddr storage, completion tokens, or the backend proactor.
  if (hyle::sched::current_worker_index() != -1)
    return 4;
  auto endpoint = hyle::io::Endpoint::parse("127.0.0.1", 2048);
  if (!endpoint.has_value() || !endpoint->is_loopback())
    return 5;
  hyle::io::IoCapabilities capabilities;
  if (capabilities.backend_name() != "none")
    return 6;

  // The public product must carry the HTTP factor through every delivery
  // shape. Exercise a typed factory across the binary boundary without
  // depending on a live peer: an empty transport is rejected before any
  // connection state is allocated.
  hyle::io::HttpLimits http_limits;
  http_limits.max_head_bytes = 32 * 1024;
  auto http = hyle::io::HttpClientConnection::create(
      hyle::mem::system_allocator(), hyle::io::HttpTransport{}, http_limits);
  if (http.has_value() || !http.error().IsInvalidArgument())
    return 7;

  // Reliable transfer is part of the same public IO product in source,
  // static, and shared delivery. The plan owns both paths immediately and
  // performs no filesystem work until transfer() is awaited on an IO worker.
  auto transfer = hyle::io::TransferPlan::local(
      hyle::mem::system_allocator(), {"model.source"}, {"model.ready"});
  if (!transfer.has_value() || transfer->source() != "model.source" ||
      transfer->destination() != "model.ready")
    return 8;

#if defined(HYLE_IO_TLS)
  if (!hyle::io::tls_available())
    return 9;
  hyle::io::TlsClientOptions tls_options;
  tls_options.verify = hyle::io::TlsVerifyMode::kNone;
  auto tls_context = hyle::io::TlsContext::make_client(tls_options);
  if (!tls_context.has_value() || !tls_context->valid())
    return 10;
#endif
  return 0;
}
