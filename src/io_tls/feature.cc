#include "hyle/io/tls.h"

namespace hyle::io {

bool tls_available() noexcept {
#if defined(HYLE_IO_TLS)
  return true;
#else
  return false;
#endif
}

}  // namespace hyle::io
