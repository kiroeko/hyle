#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// Host policy for one retired, finite verifier. This owns no model state,
// sampler or cache. The final emitted token remains pending; only consumed
// target rows may select a committed conv/GDN/KV prefix.
namespace hyle::inference::mtp {
// Request limits are supplied by the consumer, independent of model geometry.
// A one-row tail uses target-only sampling; the final emitted ID stays pending.
struct SamplingLimits final {
  std::size_t quota = std::numeric_limits<std::size_t>::max();
  std::size_t remaining_context = std::numeric_limits<std::size_t>::max();
  std::span<const std::int32_t> eos{};
  void validate(std::int64_t vocabulary) const {
    if (!quota || !remaining_context || vocabulary <= 0 ||
        vocabulary > std::numeric_limits<std::int32_t>::max() ||
        std::ranges::any_of(eos, [vocabulary](auto id) { return id < 0 || id >= vocabulary; }))
      throw std::runtime_error("MTP sampling needs positive output/context limits and valid EOS IDs");
  }
  bool single_row() const noexcept { return quota == 1 || remaining_context == 1; }
};
struct AcceptedPrefix final {
  // Nonterminal accepted proposals that entered model state, not acceptance
  // probability/count of emitted proposals. An accepted EOS is still pending.
  std::size_t consumed = 0, consumed_drafts = 0;
  std::int32_t next_id = -1;
  bool first_agrees = false, stopped = false;
};
namespace detail {
inline void token_ids(std::span<const std::int32_t> ids, std::int32_t vocabulary) {
  if (vocabulary <= 0 || std::ranges::any_of(ids,
      [vocabulary](auto id) { return id < 0 || id >= vocabulary; }))
    throw std::runtime_error("MTP prefix contains an invalid token ID or vocabulary");
}
inline void prepare_emitted(std::vector<std::int32_t> &emitted, std::size_t count) {
  emitted.clear(); emitted.reserve(count);
}
inline void prepare_emitted(std::span<std::int32_t> emitted, std::size_t count) {
  if (emitted.size() < count) throw std::runtime_error("MTP emitted output span is smaller than the bounded prefix");
}
inline void emit(std::vector<std::int32_t> &emitted, std::size_t, std::int32_t token) {
  emitted.push_back(token);
}
inline void emit(std::span<std::int32_t> emitted, std::size_t row, std::int32_t token) {
  emitted[row] = token;
}

template<class Output, class Accept, class Final>
AcceptedPrefix choose_prefix(std::span<const std::int32_t> inputs,
    std::size_t quota, std::span<const std::int32_t> eos,
    Output &emitted, Accept accept, Final final_id) {
  if (inputs.empty() || !quota) throw std::runtime_error("MTP selection needs a pending input and positive output quota");
  const auto ended = [&](auto id) { return std::ranges::find(eos, id) != eos.end(); };
  if (ended(inputs.front())) throw std::runtime_error("MTP cannot consume a terminal pending token");
  prepare_emitted(emitted, (std::min)(quota, inputs.size()));
  AcceptedPrefix result;
  const auto proposal_count = inputs.size() - 1;
  for (std::size_t row = 0; row <= proposal_count; ++row) {
    const bool agrees = row < proposal_count && accept(row);
    if (!row) result.first_agrees = agrees;
    const auto token = agrees ? inputs[row + 1] : final_id(row);
    emit(emitted, row, token);
    // Even a matching/accepted EOS is an emitted pending token. Quota has the
    // same boundary: never consume the last output merely because it agrees.
    if (!agrees || ended(token) || row + 1 == quota) {
      result.consumed = row + 1; result.next_id = token; result.stopped = ended(token);
      return result;
    }
    ++result.consumed_drafts;
  }
  throw std::runtime_error("MTP prefix selection did not reach a final output");
}
} // namespace detail

inline AcceptedPrefix select_greedy_prefix(std::span<const std::int32_t> inputs,
    std::span<const std::int32_t> target_ids, std::size_t quota,
    std::span<const std::int32_t> eos, std::int32_t vocabulary,
    std::vector<std::int32_t> &emitted) {
  if (inputs.size() != target_ids.size()) throw std::runtime_error("MTP greedy target rows differ from verifier inputs");
  detail::token_ids(inputs, vocabulary); detail::token_ids(target_ids, vocabulary);
  return detail::choose_prefix(inputs, quota, eos, emitted,
      [&](auto row) { return inputs[row + 1] == target_ids[row]; },
      [&](auto row) { return target_ids[row]; });
}

// Allocation-free output for finite callers. Provide at least
// min(quota, inputs.size()) elements, disjoint from inputs and target_ids.
// Only the first result.consumed elements are written; the tail is unchanged.
// A short span is rejected before any output is written.
inline AcceptedPrefix select_greedy_prefix(std::span<const std::int32_t> inputs,
    std::span<const std::int32_t> target_ids, std::size_t quota,
    std::span<const std::int32_t> eos, std::int32_t vocabulary,
    std::span<std::int32_t> emitted) {
  if (inputs.size() != target_ids.size()) throw std::runtime_error("MTP greedy target rows differ from verifier inputs");
  detail::token_ids(inputs, vocabulary); detail::token_ids(target_ids, vocabulary);
  return detail::choose_prefix(inputs, quota, eos, emitted,
      [&](auto row) { return inputs[row + 1] == target_ids[row]; },
      [&](auto row) { return target_ids[row]; });
}

// Each final_ids[i] is the public sampling::decision result for p_i/q_i:
// rejection uses its residual; acceptance uses p_{i+1}. Only the first rejected
// row or the last accepted row's bonus is selected, with one final RNG address.
inline AcceptedPrefix select_sampled_prefix(std::span<const std::int32_t> inputs,
    std::span<const std::uint8_t> accepts, std::span<const std::int32_t> final_ids,
    std::size_t quota, std::span<const std::int32_t> eos, std::int32_t vocabulary,
    std::vector<std::int32_t> &emitted) {
  if (inputs.size() == 1 && accepts.empty() && final_ids.size() == 1) {
    detail::token_ids(inputs, vocabulary); detail::token_ids(final_ids, vocabulary);
    return detail::choose_prefix(inputs, quota, eos, emitted,
        [](auto) { return false; }, [&](auto) { return final_ids[0]; });
  }
  if (inputs.size() < 2 || accepts.size() != inputs.size() - 1 || final_ids.size() != accepts.size() ||
      std::ranges::any_of(accepts, [](auto value) { return value > 1; }))
    throw std::runtime_error("MTP sampled prefix needs one valid decision per proposal");
  detail::token_ids(inputs, vocabulary); detail::token_ids(final_ids, vocabulary);
  return detail::choose_prefix(inputs, quota, eos, emitted,
      [&](auto row) { return accepts[row] != 0; },
      [&](auto row) { return final_ids[(std::min)(row, final_ids.size() - 1)]; });
}
} // namespace hyle::inference::mtp
