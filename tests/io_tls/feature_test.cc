#include <gtest/gtest.h>

#include "hyle/io/tls.h"

TEST(IoTlsFeature, ReportsTheCompiledCapability) {
#if defined(HYLE_IO_TLS)
  EXPECT_TRUE(hyle::io::tls_available());
#else
  EXPECT_FALSE(hyle::io::tls_available());
#endif
}
