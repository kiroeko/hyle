#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace hyle::inference {

// Immutable output configuration, not model/KV state or a sampling cursor.
// Extra IDs are copied/normalized once at cold request construction. Copies
// into an original ledger are cold metadata; no request address is borrowed.
class RequestStopPolicy final {
public:
  RequestStopPolicy() = default;
  explicit RequestStopPolicy(std::span<const std::int32_t> extra,
                             bool ignore_model_eos = false)
      : extra_(extra.begin(), extra.end()), ignore_model_eos_(ignore_model_eos) {
    if (std::ranges::any_of(extra_, [](auto id) { return id < 0; }))
      throw std::invalid_argument("request stop token must be nonnegative");
    std::ranges::sort(extra_);
    extra_.erase(std::unique(extra_.begin(), extra_.end()), extra_.end());
  }
  bool is_default() const noexcept { return extra_.empty() && !ignore_model_eos_; }
  bool ignore_model_eos() const noexcept { return ignore_model_eos_; }
  std::span<const std::int32_t> extra_tokens() const noexcept { return extra_; }
  void validate(std::int64_t vocabulary) const {
    // The sorted cold maximum makes every actual admission check O(1).
    if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)() ||
        (!extra_.empty() && extra_.back() >= vocabulary))
      throw std::out_of_range("request stop token is outside the model vocabulary");
  }
  bool stopped(std::int32_t token, bool model_eos) const noexcept {
    return (!ignore_model_eos_ && model_eos) ||
        std::binary_search(extra_.begin(), extra_.end(), token);
  }
  friend bool operator==(const RequestStopPolicy &, const RequestStopPolicy &) = default;
private:
  std::vector<std::int32_t> extra_;
  bool ignore_model_eos_ = false;
};

inline const RequestStopPolicy default_request_stop_policy;
inline const RequestStopPolicy &request_stop_policy(const RequestStopPolicy *value) noexcept {
  return value ? *value : default_request_stop_policy;
}

} // namespace hyle::inference
