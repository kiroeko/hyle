#include <gtest/gtest.h>

#include "hyle/io/http2.h"

TEST(IoHttp2Feature, ReportsTheCompiledCapability) {
#if defined(HYLE_IO_HTTP2)
  EXPECT_TRUE(hyle::io::http2_available());
#else
  EXPECT_FALSE(hyle::io::http2_available());
#endif
}
