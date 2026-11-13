#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hyle::inference {

// Immutable, request-owned conditioning for an exact prompt interval. The
// provider interprets the format, shape and payload; common admission/transport
// never interprets patch order, projector equations or positional mathematics.
// This is input data, not an execution plan, native pointer or cache identity.
// A zero-row interval is a valid anchor for model-owned cross-attention or
// other conditioning that does not replace language embeddings.
struct PromptAttachment final {
  std::string format;
  std::size_t begin = 0, rows = 0;
  std::vector<std::size_t> shape;
  std::vector<std::byte> payload;
};

} // namespace hyle::inference
