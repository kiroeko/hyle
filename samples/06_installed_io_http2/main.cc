#include "hyle/io/http2.h"
#include "hyle/mem/allocator.h"

#if !defined(HYLE_IO_HTTP2)
#error "installed package advertises io_http2 but did not expose HYLE_IO_HTTP2"
#endif

int main() {
  const auto id = hyle::io::Http2StreamId::from_value(1);
  hyle::io::Http2RequestHead request{.method = "GET",
                                     .scheme = "https",
                                     .authority = "localhost",
                                     .path = "/health"};
  return hyle::io::http2_available() && id.has_value() &&
                 request.method == "GET"
             ? 0
             : 1;
}
