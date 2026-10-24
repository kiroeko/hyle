#pragma once
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include "../compute_models/dspark_original_roles.h"

namespace dspark_head_checks {
namespace original = dspark_original_roles;
inline bool projection_equal(std::span<const float> actual,std::span<const float> expected,
                      std::string_view name) {
  if (actual.size()!=expected.size()) {ADD_FAILURE()<<name<<" extent";return false;}
  for (std::size_t i=0;i<actual.size();++i) {
    // Existing F32 projection gate, cublas_provider_test.cc:151. The BF16
    // multi-stage allowance is used ONLY above this head's explicit boundary.
    const double tolerance=1.0e-6*std::abs(double(expected[i]))+1.0e-6;
    if (!std::isfinite(actual[i]) || std::abs(double(actual[i])-expected[i])>tolerance) {
      ADD_FAILURE()<<name<<"["<<i<<"] actual="<<actual[i]<<" reference="<<expected[i]<<" tolerance="<<tolerance;
      return false;
    }
  }
  return true;
}

inline bool adjusted_logits_equal(std::span<const float> actual,
    std::span<const float> base_logits, const original::HeadReference &expected) {
  if (actual.size() != expected.exact_logits.size()) {
    ADD_FAILURE() << "adjusted logit extent";
    return false;
  }
  // Each trained Markov product is exact in F32 (two BF16 operands), and
  // this generated LM has one exact BF16 product per logit. There are 257
  // exact terms and at most256 F32 additions, including the final Add.
  // A single-projection relative gate on their possibly cancelling result
  // does not propagate reduction error. Keep the ORIGINAL Graph unchanged.
  constexpr double u32 = 0x1p-24, u64 = 0x1p-53;
  constexpr double gamma32 = 256 * u32 / (1 - 256 * u32);
  constexpr double gamma64 = 256 * u64 / (1 - 256 * u64);
  std::size_t former_gate_exceedances = 0;
  double maximum_error = 0, maximum_ratio = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(double(actual[i]) - expected.exact_logits[i]);
    const double terms_upper = expected.logit_absolute_terms[i] / (1 - gamma64);
    const double bound = std::nextafter((gamma32 + gamma64) * terms_upper,
                                       std::numeric_limits<double>::infinity());
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, bound ? error / bound : 0);
    if (!std::isfinite(actual[i]) || error > bound) {
      ADD_FAILURE() << "adjusted logit[" << i << "] actual=" << actual[i]
                    << " exact=" << expected.exact_logits[i]
                    << " error=" << error << " F32 reduction bound=" << bound;
      return false;
    }
    const double old_gate = 1.0e-6 * std::abs(double(expected.logits[i])) + 1.0e-6;
    if (std::abs(double(actual[i]) - expected.logits[i]) > old_gate) {
      if (former_gate_exceedances++ == 0)
        std::printf("DSpark composed logit index=%zu L=%.17g B=%.17g S=%.17g actual=%.17g exact=%.17g bound=%.17g\n",
            i, double(base_logits[i]), expected.exact_bias[i],
            expected.bias_absolute_products[i], double(actual[i]), expected.exact_logits[i], bound);
    }
  }
  std::printf("DSpark composed F32 logits: prior_single_projection_gate_exceedances=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
              former_gate_exceedances, maximum_error, maximum_ratio);
  return true;
}


} // namespace dspark_head_checks
