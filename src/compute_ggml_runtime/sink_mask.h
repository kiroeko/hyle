#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "ggml.h"

namespace hyle::compute::ggml_runtime {

// Public SinkAttention inference has checked the complete score descriptor.
// A shared rank2 mask does not contain the batch extent of those scores.
inline std::uint64_t sink_score_elements(std::span<const std::int64_t> query,
                                         std::span<const std::int64_t> key) {
  return std::uint64_t(query[0]) * std::uint64_t(query[1]) *
         std::uint64_t(query[2]) * std::uint64_t(key[1]);
}

// Dense public [B,Q,S] is initially [S,Q,B,1] in ggml. Softmax broadcasts
// over heads at ne2 and selects batches at ne3: [S,Q,1,B]. Shared [Q,S]
// and already normalized masks need no additional tensor or storage.
inline ggml_tensor *sink_mask(ggml_context *context, ggml_tensor *mask) {
  return mask->ne[2] == 1 ? mask
      : ggml_reshape_4d(context, mask, mask->ne[0], mask->ne[1], 1, mask->ne[2]);
}

inline ggml_tensor *sink_mask_rows(ggml_context *context, ggml_tensor *mask,
                                    std::int64_t begin, std::int64_t rows) {
  mask = sink_mask(context, mask);
  auto *part = ggml_view_4d(context, mask, mask->ne[0], rows, mask->ne[2],
      mask->ne[3], mask->nb[1], mask->nb[2], mask->nb[3],
      std::size_t(begin) * mask->nb[1]);
  // Batch tiles keep the original full-query pitch. Native softmax requires
  // a contiguous mask, so pack only this tile when that pitch leaves gaps.
  return ggml_is_contiguous(part) ? part : ggml_cont(context, part);
}

} // namespace hyle::compute::ggml_runtime
