#pragma once

#include "request_progress.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace hyle::inference {

// One request's index of observed two-token continuations. Only the prompts
// through the active turn and its committed output can enter the index.
class PrefixDraftLookup final {
public:
  using Pair = std::array<std::int32_t, 2>;

  explicit PrefixDraftLookup(std::size_t capacity) noexcept : capacity_(capacity) {}

  std::optional<Pair> propose(const RequestProgress &progress) {
    if (!progress.prefilled || progress.generated.empty() || progress.turn_index >= progress.turns().size() ||
        progress.turn_output_begin > progress.generated.size()) return std::nullopt;

    const auto turn = progress.turn_index;
    const bool new_turn = turn_ != turn;
    if (!turn_ || (new_turn && (turn != *turn_ + 1 || indexed_outputs_ > progress.turn_output_begin)) ||
        indexed_outputs_ > progress.generated.size()) {
      // A skipped turn has no recorded output boundary here. Start with the
      // current turn so a lookup can never cross an unobserved boundary.
      for (auto &table : tables_) table.clear();
      history_.clear();
      turn_.reset();
      indexed_outputs_ = progress.turn_output_begin;
    }
    const auto generated = std::span(progress.generated).subspan(indexed_outputs_);
    const bool append_prompt = !turn_ || new_turn;
    const auto prompt = append_prompt ? progress.turns()[turn].prompt : std::span<const std::int32_t>{};
    const auto remaining = capacity_ - history_.size();
    if (prompt.size() > remaining || generated.size() > remaining - prompt.size()) return std::nullopt;

    if (append_prompt) {
      const auto previous_tail = progress.turn_output_begin - indexed_outputs_;
      append(generated.first(previous_tail));
      append(prompt);
      append(generated.subspan(previous_tail));
    } else append(generated);
    turn_ = turn;
    indexed_outputs_ = progress.generated.size();

    for (std::size_t table = 0; table < widths_.size(); ++table) {
      const auto width = widths_[table];
      if (history_.size() < width) continue;
      const auto found = tables_[table].find(pattern(width, 0));
      if (found != tables_[table].end()) return found->second;
    }
    return std::nullopt;
  }

private:
  using Pattern = std::array<std::int32_t, 8>;
  struct PatternHash {
    std::size_t operator()(const Pattern &tokens) const noexcept {
      std::size_t hash = 0;
      for (const auto token : tokens) hash = hash * 0x9e3779b1u + static_cast<std::uint32_t>(token);
      return hash;
    }
  };
  static constexpr std::array<std::size_t, 4> widths_{8, 6, 4, 3};

  Pattern pattern(std::size_t width, std::size_t tail) const noexcept {
    Pattern key{};
    std::copy_n(history_.data() + (history_.size() - width - tail), width, key.begin());
    return key;
  }

  void append(std::span<const std::int32_t> tokens) {
    for (const auto token : tokens) {
      history_.push_back(token);
      const auto size = history_.size();
      if (size < widths_.back() + 2) continue;
      const Pair next{history_[size - 2], token};
      for (std::size_t table = 0; table < widths_.size(); ++table) {
        const auto width = widths_[table];
        if (size >= width + next.size())
          tables_[table].insert_or_assign(pattern(width, next.size()), next);
      }
    }
  }

  std::size_t capacity_;
  std::optional<std::size_t> turn_;
  std::size_t indexed_outputs_ = 0;
  std::vector<std::int32_t> history_;
  std::array<std::unordered_map<Pattern, Pair, PatternHash>, widths_.size()> tables_;
};

} // namespace hyle::inference
