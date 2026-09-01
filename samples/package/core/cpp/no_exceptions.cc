#include <hyle/core/assert.h>
#include <hyle/core/cache.h>
#include <hyle/core/compiler.h>
#include <hyle/core/environment.h>
#include <hyle/version.h>

#if HYLE_HAS_EXCEPTIONS
#  error "Core-only no-exception consumer unexpectedly has exceptions enabled"
#endif

int main() {
  int value = 42;
  HYLE_FATAL_ASSERT(value == 42);
  HYLE_DEBUG_ASSERT(value > 0);
  hyle::core::prefetch_read(&value);
  return hyle::kVersionPacked != 0 && HYLE_LIKELY(value == 42) ? 0 : 1;
}
