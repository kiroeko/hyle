/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "hyle/core/assert.h"
#include "hyle/visibility.h"

extern "C" HYLE_EXPORT int hyle_core_visibility_probe(
    const int* value) noexcept {
  HYLE_FATAL_ASSERT(value != nullptr);
  return *value;
}
