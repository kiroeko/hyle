/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "hyle/core/assert.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"

namespace {

HYLE_FORCE_INLINE int force_inline_body(int value) noexcept {
  return value * 3 + 1;
}

}  // namespace

// Stable C linkage makes this maintenance-only object straightforward to
// inspect on all three supported object formats. None of these symbols ship.
extern "C" {

HYLE_NOINLINE int hyle_core_probe_noinline_body(int value) noexcept {
  return value * 5 + 2;
}

int hyle_core_probe_noinline_caller(int value) noexcept {
  return hyle_core_probe_noinline_body(value);
}

int hyle_core_probe_force_inline_caller(int value) noexcept {
  return force_inline_body(value);
}

int hyle_core_probe_identity(int value) noexcept { return value; }

int hyle_core_probe_assume(int value) noexcept {
  HYLE_ASSUME(value > 0);
  return value > 0 ? value : 0;
}

void hyle_core_probe_prefetch_read(const void* address) noexcept {
  hyle::core::prefetch_read(address);
}

void hyle_core_probe_prefetch_read_non_temporal(const void* address) noexcept {
  hyle::core::prefetch_read_non_temporal(address);
}

void hyle_core_probe_prefetch_write(const void* address) noexcept {
  hyle::core::prefetch_write(address);
}

int hyle_core_probe_fatal_hot(const int* value) noexcept {
  HYLE_FATAL_ASSERT(value != nullptr);
  return *value;
}

}  // extern "C"
