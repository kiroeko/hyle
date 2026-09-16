#pragma once

#include "hyle/error/status.h"

namespace hyle::compute::apple::detail {
// Called only at the native adapter's exception boundary, inside a catch.
Status native_exception() noexcept;
} // namespace hyle::compute::apple::detail
