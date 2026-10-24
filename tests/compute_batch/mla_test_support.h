#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "hyle/compute/batch_layout.h"
#include "hyle/simd/dtype.h"

namespace mla_test {
namespace c = hyle::compute;
inline hyle::Status check(bool value, const char *message) {
  return value ? hyle::Status::Ok() : hyle::Status::Internal(message);
}
inline std::uint16_t encode(float x, bool bf) {
  return bf ? hyle::simd::f32_to_bf16(x).bits : hyle::simd::f32_to_f16(x).bits;
}
inline float decode(std::uint16_t x, bool bf) {
  return bf ? hyle::simd::bf16_to_f32({x}) : hyle::simd::f16_to_f32({x});
}
// Full independent FP64 softmax/value reference. V is the first 512 values
// of the same CKV row used by K; no expanded or duplicate V fixture exists.
inline hyle::Status oracle(c::BatchCapacity cap, c::BatchLayoutVersion layout,
    const std::vector<std::uint16_t> &q, const std::vector<std::uint16_t> &cache,
    const c::ReadLease &out, bool bf, std::int32_t window_left=-1) {
  const auto host = layout.host();
  std::vector<std::uint16_t> actual(std::size_t(cap.max_query_rows)*cap.query_heads*512);
  HYLE_ASSIGN_OR_RETURN(auto view, out.view());
  HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(actual))));
  double maximum = 0;
  for (unsigned r=0; r<layout.facts().requests; ++r) {
    const auto rows = host.query_indptr[r+1]-host.query_indptr[r];
    const auto refs = host.kv_indptr[r+1]-host.kv_indptr[r];
    const bool paged = cap.storage == c::BatchStorage::kPaged;
    const auto keys = paged ? (refs-1)*int(cap.page_size)+host.last_page_lengths[r] : refs;
    const auto base = [&](int k) {
      const auto row = paged ? host.page_indices[host.kv_indptr[r]+k/cap.page_size]*cap.page_size+k%cap.page_size
                             : host.kv_indptr[r]+k;
      return std::size_t(row)*cap.qk_head_dim;
    };
    for (int row=0; row<rows; ++row)
      for (unsigned head=0; head<cap.query_heads; ++head) {
        const auto qb = (std::size_t(host.query_indptr[r]+row)*cap.query_heads+head)*cap.qk_head_dim;
        const auto count = keys-rows+row+1;
        const auto first = window_left<0 ? 0 : std::max(0,count-1-window_left);
        std::vector<double> probabilities(count-first);
        double high=-1e100, sum=0;
        for (int k=first; k<count; ++k) {
          double dot=0;
          for (unsigned d=0; d<cap.qk_head_dim; ++d)
            dot += double(decode(q[qb+d],bf))*decode(cache[base(k)+d],bf);
          probabilities[k-first]=dot/16;
          high=std::max(high,probabilities[k-first]);
        }
        for (auto &x:probabilities) { x=std::exp(x-high); sum+=x; }
        const auto ob=(std::size_t(host.query_indptr[r]+row)*cap.query_heads+head)*512;
        for (unsigned d=0; d<512; ++d) {
          double expected=0;
          for (int k=first; k<count; ++k)
            expected += probabilities[k-first]/sum*decode(cache[base(k)+d],bf);
          const auto error=std::abs(decode(actual[ob+d],bf)-expected);
          maximum=std::max(maximum,error);
          if (!std::isfinite(error) || error>=.007)
            std::fprintf(stderr,"MLA mismatch req=%u row=%d head=%u d=%u actual=%.9g expected=%.12g error=%.9g\n",
                r,row,head,d,decode(actual[ob+d],bf),expected,error);
          HYLE_RETURN_IF_ERROR(check(std::isfinite(error) && error<.007,"MLA independent FP64 oracle differs"));
        }
      }
  }
  std::printf("MLA H%u D%u %s full FP64 maxAbs=%.9g PASS\n",cap.query_heads,cap.qk_head_dim,bf?"BF16":"F16",maximum);
  return hyle::Status::Ok();
}
} // namespace mla_test
