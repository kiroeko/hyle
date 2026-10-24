#pragma once

#include <cmath>
#include <stdexcept>

#include "hyle/compute/model_ops.h"

// Immutable graph-construction arithmetic, independent of source identity and
// provider choice. The existing official default remains byte-for-byte Divide.
namespace glm_fp8_arithmetic {
struct Arithmetic final {
  hyle::compute::model_ops::Float8ScalePolicy policy =
      hyle::compute::model_ops::Float8ScalePolicy::kDivide;
  float absmax_floor = 1.0e-10F;
  void require() const {
    using Policy = hyle::compute::model_ops::Float8ScalePolicy;
    if ((policy != Policy::kDivide && policy != Policy::kMultiplyReciprocal &&
         policy != Policy::kPowerOfTwo) || !std::isfinite(absmax_floor) ||
        !(absmax_floor > 0) || !std::isnormal(absmax_floor / 448.0F))
      throw std::runtime_error("GLM needs its explicit finite FP8 arithmetic");
  }
};
} // namespace glm_fp8_arithmetic
