#pragma once

#include <cstdint>

#include "hyle/compute/operation.h"

#if defined(_WIN32)
#  if defined(HYLE_COMPUTE_OPS_FIXTURE_BUILD)
#    define HYLE_COMPUTE_OPS_FIXTURE_API __declspec(dllexport)
#  else
#    define HYLE_COMPUTE_OPS_FIXTURE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_COMPUTE_OPS_FIXTURE_API __attribute__((visibility("default")))
#else
#  define HYLE_COMPUTE_OPS_FIXTURE_API
#endif

namespace hyle::compute::test {

// C++-only test seam: the producer's descriptor type stays private so the
// consumer must recover it from schema identity rather than a shared vtable.
HYLE_COMPUTE_OPS_FIXTURE_API Operation
make_shared_custom_operation(std::uint32_t tag) noexcept;

} // namespace hyle::compute::test

#undef HYLE_COMPUTE_OPS_FIXTURE_API

