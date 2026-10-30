#pragma once

#include <cstddef>
#include <cstdint>

#include "../../inference/sampling_cursor.h"

namespace qwen35 {

// Value facts of the complete model state, independent of request bookkeeping.
struct GenerationBoundary final {
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t next_id = -1, catchup_token = -1;
  hyle::inference::SamplingCursor sampling{};
  friend bool operator==(const GenerationBoundary &, const GenerationBoundary &) = default;
};

} // namespace qwen35
