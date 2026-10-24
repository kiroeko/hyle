#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "dspark_original_roles.h"

// Independent first-episode reference for the full original BF16 LM bank.
// Only small result/error vectors are retained; coefficients stay in their
// actual BF16 source and never expand into another dense F32 weight bank.
namespace dspark_original_lm_reference {
namespace original=dspark_original_roles;
inline constexpr double u32=0x1p-24,u64=0x1p-53;
inline constexpr double gamma(std::size_t n,double u) {return n*u/(1-n*u);}
struct Reference final {
  std::vector<double> exact,absolute_products,f32_bound;
};
inline Reference evaluate(std::span<const std::uint16_t> head,
                           std::span<const float> normalized) {
  if(head.size()!=original::vocabulary*original::hidden || normalized.empty() ||
      normalized.size()%original::hidden)
    throw std::runtime_error("original full-vocabulary LM reference dimensions differ");
  for(const auto value:normalized)
    if(!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value)&0xffffU))
      throw std::runtime_error("original LM reference needs actual widened BF16 RMS values");
  const auto rows=normalized.size()/original::hidden;
  Reference result;
  result.exact.resize(rows*original::vocabulary);
  result.absolute_products.resize(result.exact.size());
  result.f32_bound.resize(result.exact.size());
  constexpr double g32=gamma(original::hidden,u32),g64=gamma(original::hidden,u64);
  for(std::size_t row=0;row<rows;++row) {
    const auto input=normalized.subspan(row*original::hidden,original::hidden);
    for(std::size_t token=0;token<original::vocabulary;++token) {
      const auto value=original::dot(input,head.subspan(token*original::hidden,original::hidden));
      if(!value.exact_f32_products_with_normal_quantum || !std::isfinite(value.value) ||
          value.sum_absolute_products>double(std::numeric_limits<float>::max())/4)
        throw std::runtime_error("original LM dot needs a separate underflow/overflow analysis");
      const auto i=row*original::vocabulary+token;
      result.exact[i]=value.value;
      result.absolute_products[i]=value.sum_absolute_products;
      // All BF16 products are exact F32. The F32 accumulation has at most K
      // rounding steps; include the independent F64 oracle's error as well.
      const double upper=value.sum_absolute_products/(1-g64);
      result.f32_bound[i]=std::nextafter((g32+g64)*upper,
          std::numeric_limits<double>::infinity());
    }
  }
  return result;
}
inline bool logits_equal(std::span<const float> actual,const Reference &reference) {
  if(actual.size()!=reference.exact.size()) {ADD_FAILURE()<<"original LM output extent";return false;}
  double max_error=0,max_ratio=0;
  for(std::size_t i=0;i<actual.size();++i) {
    const double error=std::abs(double(actual[i])-reference.exact[i]);
    const double bound=reference.f32_bound[i];
    if(!std::isfinite(actual[i]) || error>bound) {
      ADD_FAILURE()<<"original LM["<<i<<"] actual="<<actual[i]<<" F64="<<reference.exact[i]
          <<" error="<<error<<" F32 dot bound="<<bound;
      return false;
    }
    max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,bound?error/bound:0);
  }
  std::printf("DSpark original full-V LM: values=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      actual.size(),max_error,max_ratio);
  return true;
}
inline bool composed_logits_equal(std::span<const float> actual,const Reference &lm,
                                    const original::HeadReference &markov) {
  if(actual.size()!=lm.exact.size() || actual.size()!=markov.exact_bias.size() ||
      actual.size()!=markov.bias_absolute_products.size()) {
    ADD_FAILURE()<<"original LM plus Markov extent";return false;
  }
  constexpr double g32=gamma(original::rank,u32),g64=gamma(original::rank,u64);
  double max_error=0,max_ratio=0;
  for(std::size_t i=0;i<actual.size();++i) {
    const double bias_bound=(g32+g64)*markov.bias_absolute_products[i]/(1-g64);
    const double operands=std::abs(lm.exact[i])+std::abs(markov.exact_bias[i]);
    // Propagate each independently checked dot through the final F32 Add.
    // Cancellation must use absolute term bounds, never a relative-only gate
    // on the potentially tiny combined logit. The old Markov gate is intact.
    const double bound=std::nextafter(lm.f32_bound[i]+bias_bound+
        u32*(operands+lm.f32_bound[i]+bias_bound)+u64*operands,
        std::numeric_limits<double>::infinity());
    const double expected=lm.exact[i]+markov.exact_bias[i];
    const double error=std::abs(double(actual[i])-expected);
    if(!std::isfinite(actual[i]) || error>bound) {
      ADD_FAILURE()<<"original LM plus Markov["<<i<<"] actual="<<actual[i]<<" F64="<<expected
          <<" error="<<error<<" composed bound="<<bound;
      return false;
    }
    max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,bound?error/bound:0);
  }
  std::printf("DSpark original composed full-V logits: maxAbs=%.17g maxBoundRatio=%.17g\n",max_error,max_ratio);
  return true;
}

// Confidence is a separate F32 dot on raw collapsed hidden and the actual
// preceding-token Markov embedding. It never consumes the normalized LM input
// or any reference upstream activation. All operands retain their BF16 values.
inline bool confidence_equal(std::span<const float> actual,
    std::span<const float> hidden,std::span<const std::uint16_t> markov,
    std::span<const std::uint16_t> coefficients) {
  constexpr auto k=original::hidden+original::rank;
  if(actual.empty() || hidden.size()!=actual.size()*original::hidden ||
      markov.size()!=actual.size()*original::rank || coefficients.size()!=k) {
    ADD_FAILURE()<<"original confidence input extent";return false;
  }
  for(const auto value:hidden)
    if(!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value)&0xffffU)) {
      ADD_FAILURE()<<"original confidence needs actual widened BF16 raw hidden";return false;
    }
  constexpr double g32=gamma(k,u32),g64=gamma(k,u64);
  std::array<float,k> input{};
  std::size_t former_gate_exceedances=0;
  double maximum_error=0,maximum_sumabs=0,maximum_ratio=0;
  for(std::size_t row=0;row<actual.size();++row) {
    std::copy_n(hidden.begin()+row*original::hidden,original::hidden,input.begin());
    for(std::size_t column=0;column<original::rank;++column)
      input[original::hidden+column]=original_role_input::widen(markov[row*original::rank+column]);
    const auto reference=original::dot(input,coefficients);
    if(!reference.exact_f32_products_with_normal_quantum || !std::isfinite(reference.value) ||
        reference.sum_absolute_products>double(std::numeric_limits<float>::max())/4) {
      ADD_FAILURE()<<"original confidence needs a separate underflow/overflow analysis";return false;
    }
    // K exact F32 products, at most K F32 accumulation roundings. The same
    // sufficient normal-quantum condition used by the Markov oracle rules out
    // FTZ-sensitive cancellation. Include the compensated F64 oracle error
    // and the error in its positive absolute-product sum; no upstream error
    // or observed discrepancy is used to choose this analytic bound.
    const double upper=reference.sum_absolute_products/(1-g64);
    const double bound=std::nextafter((g32+g64)*upper,std::numeric_limits<double>::infinity());
    const double error=std::abs(double(actual[row])-reference.value);
    const double ratio=bound?error/bound:0;
    if(!std::isfinite(actual[row]) || error>bound) {
      ADD_FAILURE()<<"original confidence["<<row<<"] actual="<<actual[row]<<" F64="<<reference.value
          <<" error="<<error<<" sumabs="<<reference.sum_absolute_products<<" F32 dot bound="<<bound;
      return false;
    }
    maximum_error=std::max(maximum_error,error);
    maximum_sumabs=std::max(maximum_sumabs,reference.sum_absolute_products);
    maximum_ratio=std::max(maximum_ratio,ratio);
    const float old_reference=float(reference.value);
    const double old_gate=1.0e-6*std::abs(double(old_reference))+1.0e-6;
    if(std::abs(double(actual[row])-old_reference)>old_gate) {
      ++former_gate_exceedances;
      std::printf("DSpark original confidence row=%zu actual=%.17g F64=%.17g error=%.17g sumabs=%.17g bound=%.17g ratio=%.17g oldGate=%.17g\n",
          row,double(actual[row]),reference.value,error,reference.sum_absolute_products,bound,ratio,old_gate);
    }
  }
  std::printf("DSpark original confidence F32 dot: K=%zu prior_single_projection_gate_exceedances=%zu maxAbs=%.17g maxSumAbs=%.17g maxBoundRatio=%.17g\n",
      k,former_gate_exceedances,maximum_error,maximum_sumabs,maximum_ratio);
  return true;
}
} // namespace dspark_original_lm_reference
