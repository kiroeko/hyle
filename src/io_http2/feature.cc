#include "hyle/io/http2.h"

namespace hyle::io {

bool http2_available() noexcept {
#if defined(HYLE_IO_HTTP2)
  return true;
#else
  return false;
#endif
}

}  // namespace hyle::io
