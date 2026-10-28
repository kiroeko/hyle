#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "prefill_plan.h"
#include "sampling_cursor.h"

namespace hyle::inference {

struct GenerationTurn final {
  std::vector<std::int32_t> prompt;
  std::size_t max_outputs = 0;
};

struct GenerationSessionSchedule final {
  std::string id;
  std::vector<GenerationTurn> turns;
};

struct RequestInputCommit final {
  std::size_t turn = 0, begin = 0;
  std::span<const std::int32_t> ids{};
};

struct RequestInputPart final {
  std::size_t begin = 0;
  std::span<const std::int32_t> ids{};
  bool complete = false;
};

// Host progress only. The schedule owner outlives this borrowed turn span.
// For open input it reserves stable turn storage before binding, only appends
// complete turns, and never changes any already exposed turn/quota/prompt.
// Mutations are serialized by that owner; this is not a concurrent ingress API.
// The model's committed head/ledger remains authoritative for its next cursor;
// initial_sampling identifies a new request, not a second mutable RNG state.
struct RequestProgress final {
  std::vector<std::int32_t> generated;
  std::size_t turn_index = 0, turn_output_begin = 0;
  SamplingCursor initial_sampling{};
  bool prefilled = false, turn_done = false;

  explicit RequestProgress(std::size_t maximum_outputs) : maximum_outputs_(maximum_outputs) {
    if (!maximum_outputs) throw std::invalid_argument("request output limit must be positive");
    generated.reserve(maximum_outputs);
  }
  RequestProgress(const RequestProgress &) = delete;
  RequestProgress &operator=(const RequestProgress &) = delete;
  RequestProgress(RequestProgress &&) noexcept = default;
  RequestProgress &operator=(RequestProgress &&) noexcept = default;

  // The schedule is borrowed once, before admission. Empty user suffixes are
  // valid: a later turn can consist solely of its historical pending token.
  void bind_turns(std::span<const GenerationTurn> turns) {
    if (turns_bound_ || started_ || observed_commits_ || !generated.empty() || turns.empty())
      throw std::logic_error("request turns must be bound once before progress starts");
    std::size_t outputs = 0;
    for (const auto &turn : turns) {
      if (!turn.max_outputs || turn.max_outputs > maximum_outputs_ - outputs)
        throw std::out_of_range("request turn quotas exceed the prepared output limit");
      outputs += turn.max_outputs;
    }
    turns_ = turns; turns_bound_ = true;
  }

  // Cold bounds for one stable borrowed schedule, not a second turn store.
  // The prompt budget counts user suffix tokens only, not pending tokens or
  // model KV/context capacity. The model owner still admits actual input rows.
  void bind_open_turns(std::span<const GenerationTurn> turns, std::size_t max_turns,
                       std::size_t max_total_prompt_tokens) {
    if (turns_bound_ || started_ || observed_commits_ || !generated.empty() || turns.empty())
      throw std::logic_error("open request turns must be bound once before progress starts");
    if (!max_turns || turns.size() > max_turns)
      throw std::out_of_range("request turns exceed the prepared turn limit");
    std::size_t outputs = 0, prompts = 0;
    validate_turn_tail(turns, max_total_prompt_tokens, outputs, prompts);
    turns_ = turns; turns_bound_ = true;
    maximum_turns_ = max_turns; maximum_prompt_tokens_ = max_total_prompt_tokens;
    scheduled_outputs_ = outputs; scheduled_prompt_tokens_ = prompts;
    input_open_ = true;
  }

  // Only new suffix metadata is read. Old span contents and prompts are an
  // immutable owner contract, not copied or rehashed here. All rejection
  // precedes publication of the expanded span and cumulative reservations.
  void extend_turns(std::span<const GenerationTurn> expanded) {
    if (!input_open_) throw std::logic_error("request input is closed");
    if (expanded.data() != turns_.data() || expanded.size() < turns_.size())
      throw std::invalid_argument("request extension requires the same stable turn prefix");
    if (expanded.size() > maximum_turns_)
      throw std::out_of_range("request turns exceed the prepared turn limit");
    auto outputs = scheduled_outputs_, prompts = scheduled_prompt_tokens_;
    validate_turn_tail(expanded.subspan(turns_.size()), maximum_prompt_tokens_, outputs, prompts);
    turns_ = expanded; scheduled_outputs_ = outputs; scheduled_prompt_tokens_ = prompts;
  }

  // Idempotent; closing input does not stop or change an active known turn.
  void close_input() noexcept { input_open_ = false; }
  bool input_open() const noexcept { return input_open_; }
  bool waiting_for_input() const noexcept {
    return input_open_ && started_ && turn_done && !more_turns();
  }
  bool request_done() const noexcept {
    return !input_open_ && started_ && turn_done && !more_turns();
  }

  std::span<const GenerationTurn> turns() const noexcept { return turns_; }
  std::size_t maximum_outputs() const noexcept { return maximum_outputs_; }
  bool turn_started() const noexcept { return started_; }

  // The model admits context and token IDs before binding this original
  // owner-backed span, whose storage survives this turn. Rebinding the same
  // input never rewinds its commits; no token owner is retained here.
  bool input_admitted() const noexcept {
    return input_turn_ != (std::numeric_limits<std::size_t>::max)() && input_turn_ == turn_index;
  }
  std::size_t input_turn() const noexcept { return input_turn_; }
  std::size_t input_consumed() const noexcept { return input_admitted() ? input_consumed_ : 0; }
  std::span<const std::int32_t> input() const noexcept {
    return input_admitted() ? input_ : std::span<const std::int32_t>{};
  }
  void bind_input(std::span<const std::int32_t> actual) {
    if (!started_) throw std::logic_error("request input requires an admitted turn");
    if (input_admitted()) {
      if (actual.data() != input_.data() || actual.size() != input_.size())
        throw std::invalid_argument("request turn input cannot change after admission");
      return;
    }
    if (prefilled || turn_done)
      throw std::logic_error("request input must be bound before observing this turn's outputs");
    input_ = actual; input_turn_ = turn_index; input_consumed_ = 0;
  }
  std::span<const std::int32_t> remaining_input() const {
    if (!input_admitted() || input_consumed_ > input_.size())
      throw std::logic_error("request has no current admitted input extent");
    return input_.subspan(input_consumed_);
  }
  RequestInputPart next_input(const PrefillPolicy &policy) const {
    (void)remaining_input();
    const auto part = policy.next(input_.size(), input_consumed_);
    return {part.begin, input_.subspan(part.begin, part.rows), part.complete};
  }

  // This borrows caller-owned, cold-prepared storage; it never creates token
  // ownership. Validate the full extent/suffix/pending before writing slot 0.
  // A no-schedule request is one turn with its constructor's actual limit.
  std::span<const std::int32_t> begin_turn(std::span<std::int32_t> prepared_input = {},
                                         std::optional<std::int32_t> actual_pending = {}) {
    if (started_ || turn_index >= (turns_.empty() ? 1 : turns_.size()))
      throw std::logic_error("request turn has already begun or is outside its schedule");
    const auto pending_slots = turn_index ? std::size_t{1} : 0;
    if (pending_slots && (!actual_pending || generated.empty() || *actual_pending != generated.back()))
      throw std::logic_error("later request turn requires its actual last emitted pending token");
    if (!pending_slots && actual_pending)
      throw std::logic_error("initial request turn cannot prepend a pending token");
    if (generated.size() > maximum_outputs_)
      throw std::out_of_range("request outputs exceed its prepared output limit");
    if (!turns_.empty()) {
      const auto &turn = turns_[turn_index];
      if (turn.prompt.size() > (std::numeric_limits<std::size_t>::max)() - pending_slots ||
          prepared_input.size() != turn.prompt.size() + pending_slots ||
          !std::equal(turn.prompt.begin(), turn.prompt.end(), prepared_input.begin() + pending_slots))
        throw std::invalid_argument("prepared turn input differs from its user suffix and pending extent");
      if (turn.max_outputs > maximum_outputs_ - generated.size())
        throw std::out_of_range("request turn exceeds its prepared output limit");
    }
    if (pending_slots) prepared_input[0] = *actual_pending;
    prefilled = false;
    turn_done = false;
    turn_output_begin = generated.size();
    started_ = true;
    return prepared_input;
  }

  std::size_t output_limit() const {
    if (turns_.empty()) return maximum_outputs_;
    if (turn_index >= turns_.size() || turn_output_begin > maximum_outputs_ ||
        turns_[turn_index].max_outputs > maximum_outputs_ - turn_output_begin)
      throw std::out_of_range("request output boundary is outside its prepared limit");
    return turn_output_begin + turns_[turn_index].max_outputs;
  }

  std::size_t remaining_outputs() const {
    const auto limit = output_limit();
    if (generated.size() > limit) throw std::out_of_range("request outputs exceed this turn's quota");
    return turn_done ? 0 : limit - generated.size();
  }

  bool more_turns() const noexcept { return !turns_.empty() && turn_index + 1 < turns_.size(); }
  void advance_turn() {
    if (!started_ || !turn_done || !more_turns())
      throw std::logic_error("request can advance only an observed complete turn with a successor");
    // Keep the last admitted turn identity until the new input is admitted.
    ++turn_index; started_ = false; input_ = {};
  }

  std::size_t observed_commits() const noexcept { return observed_commits_; }

  // Adopt only initial, already committed input history. Its old outputs do
  // not belong to the new request; a full hit applies its new pending below.
  // The imported ordinal count and consumed token prefix are independent;
  // neither adoption alone establishes completion of this request's prefill.
  void preflight_adopt_committed_history(std::size_t consumed_input = 0) const {
    if (history_adopted_ || observed_commits_ || !generated.empty() || turn_index || prefilled || turn_done)
      throw std::logic_error("request can adopt committed history only before its first output");
    if ((consumed_input && !input_admitted()) ||
        (input_admitted() && (input_consumed_ || consumed_input > input_.size())))
      throw std::out_of_range("imported request input exceeds its original admitted prefix");
  }
  void adopt_committed_history(std::size_t count, std::size_t consumed_input = 0) {
    preflight_adopt_committed_history(consumed_input);
    observed_commits_ = count; history_adopted_ = true;
    if (input_admitted()) input_consumed_ = consumed_input;
  }

  void preflight_outputs(std::span<const std::int32_t> emitted, bool user_output,
                         bool prefill_complete, bool stopped) const {
    (void)prefill_complete; (void)stopped;
    if (!started_) throw std::logic_error("request output observation requires an admitted turn");
    const auto count = user_output ? emitted.size() : 0;
    const auto limit = output_limit();
    if (generated.size() > maximum_outputs_ || generated.size() > generated.capacity() ||
        count > maximum_outputs_ - generated.size() || count > generated.capacity() - generated.size() ||
        generated.size() > limit || count > limit - generated.size())
      throw std::out_of_range("observed request outputs exceed the reserved storage or turn quota");
    if (user_output && (turn_done || emitted.empty()))
      throw std::logic_error("request cannot emit an empty result or append after a stopped turn");
  }

  // Both full-hit/non-ledger output and committed-ledger output use these
  // exact host rules. stopped is the model's actual selector/EOS fact.
  void apply_outputs(std::span<const std::int32_t> emitted, bool user_output,
                     bool prefill_complete, bool stopped) {
    preflight_outputs(emitted, user_output, prefill_complete, stopped);
    if (user_output) generated.insert(generated.end(), emitted.begin(), emitted.end());
    prefilled = prefilled || prefill_complete;
    if (user_output) {
      const auto limit = output_limit();
      turn_done = generated.size() == limit || stopped;
      // Only a newly finished, actually observed turn releases its unused
      // open-input quota. Duplicate commits return before this point; a second
      // user output after turn_done is rejected by preflight_outputs.
      if (turn_done && input_open_) scheduled_outputs_ -= limit - generated.size();
    }
  }

  // A duplicate is a no-op, including its payload. Gaps and bounds fail before
  // any host mutation. The model's authoritative commit/RNG is never advanced.
  bool preflight_commit(std::size_t ordinal, std::span<const std::int32_t> emitted,
                        bool user_output, bool prefill_complete, bool stopped,
                        std::optional<RequestInputCommit> consumed_input = {}) const {
    if (ordinal < observed_commits_) return false;
    if (ordinal != observed_commits_ || observed_commits_ == (std::numeric_limits<std::size_t>::max)())
      throw std::out_of_range("request commit observations must be contiguous and bounded");
    preflight_outputs(emitted, user_output, prefill_complete, stopped);
    if (consumed_input) preflight_input_commit(*consumed_input, prefill_complete);
    return true;
  }

  bool observe_commit(std::size_t ordinal, std::span<const std::int32_t> emitted,
                      bool user_output, bool prefill_complete, bool stopped,
                      std::optional<RequestInputCommit> consumed_input = {}) {
    if (!preflight_commit(ordinal, emitted, user_output, prefill_complete, stopped, consumed_input)) return false;
    apply_outputs(emitted, user_output, prefill_complete, stopped);
    if (consumed_input) input_consumed_ += consumed_input->ids.size();
    ++observed_commits_;
    return true;
  }

private:
  void preflight_input_commit(const RequestInputCommit &consumed, bool prefill_complete) const {
    if (consumed.ids.empty())
      throw std::invalid_argument("committed request input must contain actual teacher rows");
    if (!input_admitted() || consumed.turn != turn_index)
      throw std::logic_error("committed input does not belong to the current admitted turn");
    if (input_consumed_ > input_.size() || consumed.begin != input_consumed_ ||
        consumed.ids.size() > input_.size() - input_consumed_)
      throw std::out_of_range("committed input is not the next admitted turn slice");
    const auto *expected = input_.data();
    if (input_consumed_) expected += input_consumed_;
    if (consumed.ids.data() != expected ||
        prefill_complete != (consumed.ids.size() == input_.size() - input_consumed_))
      throw std::invalid_argument("committed input span or completion differs from the actual turn slice");
  }

  void validate_turn_tail(std::span<const GenerationTurn> tail, std::size_t maximum_prompt_tokens,
                          std::size_t &outputs, std::size_t &prompts) const {
    for (const auto &turn : tail) {
      if (!turn.max_outputs || turn.max_outputs > maximum_outputs_ - outputs)
        throw std::out_of_range("request turn quotas exceed the prepared output limit");
      if (turn.prompt.size() > maximum_prompt_tokens - prompts)
        throw std::out_of_range("request user suffixes exceed the prepared prompt token limit");
      outputs += turn.max_outputs; prompts += turn.prompt.size();
    }
  }
  std::span<const GenerationTurn> turns_{};
  std::span<const std::int32_t> input_{};
  std::size_t input_turn_ = (std::numeric_limits<std::size_t>::max)(), input_consumed_ = 0;
  std::size_t maximum_outputs_ = 0, observed_commits_ = 0;
  std::size_t maximum_turns_ = 0, maximum_prompt_tokens_ = 0;
  // Outputs are delivered tokens plus remaining known-turn quota; a real early
  // stop releases only that turn's unused reservation. User prompt budget is
  // cumulative across all exposed turns and is never refunded as KV capacity.
  std::size_t scheduled_outputs_ = 0, scheduled_prompt_tokens_ = 0;
  bool started_ = false, turns_bound_ = false, history_adopted_ = false;
  bool input_open_ = false;
};

} // namespace hyle::inference
