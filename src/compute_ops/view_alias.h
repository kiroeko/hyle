#pragma once

#include <cstddef>

#include "hyle/compute/operation.h"

namespace hyle::compute::detail {

// Same-build metadata seam shared by Graph and direct providers. The caller
// must first infer this View successfully and pass that exact output. This
// checks the remaining storage offset/span contract, without touching payload.
HYLE_COMPUTE_OPS_API Result<std::size_t>
view_alias_offset(const View &view, const TensorDesc &input,
                  const TensorDesc &output) noexcept;

} // namespace hyle::compute::detail
