#include "hyle/io/http2.h"
#include "hyle/mem/allocator.h"

#if !defined(HYLE_IO_HTTP2)
#  error "installed package advertises io_http2 but did not expose HYLE_IO_HTTP2"
#endif

int main() {
  hyle::io::Http2ClientConn client(hyle::mem::system_allocator(),
                                   hyle::io::TcpStream{});
  return client.valid() ? 1 : 0;
}
