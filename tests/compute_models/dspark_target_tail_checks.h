#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "dspark_original_lm_reference.h"
#include "row_cache_fixture.h"

// Local checks for the actual target-owned capture/LM boundary. The target
// layers' independent FP8/FP4/compressor observers live with their pure math.
namespace dspark_target_tail_checks {
namespace original=dspark_original_roles;
namespace lm=dspark_original_lm_reference;

inline void capture_mean(std::span<const float> capture,std::span<const float> raw,
    std::span<const std::uint16_t> stored) {
  if(capture.size()!=4*original::hidden||raw.size()!=original::hidden||stored.size()!=raw.size())
    throw std::runtime_error("actual four-stream capture mean extent differs");
  for(const auto value:capture)if(!std::isfinite(value)||(std::bit_cast<std::uint32_t>(value)&0xffffU))
    throw std::runtime_error("target capture must be an exact widening of actual BF16 four-stream output");
  for(std::size_t i=0;i<raw.size();++i) {
    double expected=0;for(std::size_t stream=0;stream<4;++stream)expected+=capture[stream*original::hidden+i];
    expected/=4;
    EXPECT_NEAR(raw[i],float(expected),8.0e-5F+4.0e-5F*std::abs(float(expected)))<<i;
    EXPECT_EQ(stored[i],row_cache_test::bfloat(raw[i]))<<i;
  }
}

inline void head(std::span<const std::uint16_t> coefficients,std::span<const float> normalized,
    std::span<const float> scores,std::span<const std::int32_t> ids,
    std::span<const float> top_values,bool full) {
  if(normalized.empty()||normalized.size()%original::hidden||
      scores.size()!=normalized.size()/original::hidden*original::vocabulary||
      ids.size()!=normalized.size()/original::hidden||top_values.size()!=ids.size())
    throw std::runtime_error("actual full-vocabulary target verifier extent differs");
  for(std::size_t row=0;row<ids.size();++row) {
    const auto logits=scores.subspan(row*original::vocabulary,original::vocabulary);
    std::vector<std::int32_t> best{0,1,2,3};
    const auto better=[&](std::int32_t a,std::int32_t b) {
      return logits[std::size_t(a)]>logits[std::size_t(b)]||
          (logits[std::size_t(a)]==logits[std::size_t(b)]&&a<b);
    };
    std::sort(best.begin(),best.end(),better);
    for(std::size_t token=0;token<logits.size();++token) {
      if(!std::isfinite(logits[token]))throw std::runtime_error("actual target verifier contains nonfinite scores");
      if(token>=4&&better(std::int32_t(token),best.back())) {
        best.back()=std::int32_t(token);std::sort(best.begin(),best.end(),better);
      }
    }
    EXPECT_EQ(ids[row],best.front());EXPECT_EQ(top_values[row],logits[std::size_t(best.front())]);
    if(!full) {
      const auto actual=normalized.subspan(row*original::hidden,original::hidden);
      for(const auto value:actual)if(!std::isfinite(value)||(std::bit_cast<std::uint32_t>(value)&0xffffU))
        throw std::runtime_error("target LM reference requires actual widened BF16 normalized values");
      constexpr auto g32=lm::gamma(original::hidden,lm::u32),g64=lm::gamma(original::hidden,lm::u64);
      for(const auto token:best) {
        const auto dot=original::dot(actual,coefficients.subspan(std::size_t(token)*original::hidden,original::hidden));
        if(!dot.exact_f32_products_with_normal_quantum||!std::isfinite(dot.value)||
            dot.sum_absolute_products>double(std::numeric_limits<float>::max())/4)
          throw std::runtime_error("target selected LM dot needs separate underflow/overflow analysis");
        const auto bound=std::nextafter((g32+g64)*dot.sum_absolute_products/(1-g64),std::numeric_limits<double>::infinity());
        EXPECT_LE(std::abs(double(logits[std::size_t(token)])-dot.value),bound)<<"token="<<token;
      }
    }
  }
  if(full)EXPECT_TRUE(lm::logits_equal(scores,lm::evaluate(coefficients,normalized)));
}
} // namespace dspark_target_tail_checks
