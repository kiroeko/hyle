#pragma once

#include <hyle/compute/detail/dependent_completion.h>
#include <hyle/compute/detail/generated_result.h>
#include <hyle/core/assert.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

// Direct source fixture: original F2 controls, original accepted completions,
// and original result slot. Public pending frames remain a separate gate.
namespace hyle_g03_outcome_probe {
template<class Update> struct Sources final {
  using Completion = hyle::compute::Completion;
  using Lifetime = hyle::compute::detail::CompletionLifetime;
  struct Node final {
    Sources *owner = nullptr;
    hyle::compute::detail::CompletionRetention retention;
  };
  explicit Sources(Update value) : update(std::move(value)) {}
  Update update;
  std::array<Completion *, 64> completions{};
  std::array<Node, 64> nodes{};
  std::size_t count = 0;
  std::atomic<std::size_t> remaining{0};
  std::mutex mutex;
  hyle::Status terminal;
  Lifetime retirement;
  std::shared_ptr<Sources> accepted;

  void record(const hyle::Status &status) noexcept {
    std::lock_guard lock(mutex);
    terminal.Update(status);
  }
  hyle::compute::detail::OutcomeSources protocol() noexcept {
    return {this,
      [](void *opaque) noexcept -> hyle::Result<hyle::compute::CompletionPoll> {
        auto &state = *static_cast<Sources *>(opaque);
        for (std::size_t i = 0; i < state.count; ++i) {
          auto result = state.completions[i]->poll();
          if (!result) state.record(result.error());
        }
        if (state.remaining.load(std::memory_order_acquire))
          return hyle::compute::CompletionPoll::kPending;
        state.record(state.update.observe());
        std::lock_guard lock(state.mutex);
        if (!state.terminal.ok()) return std::unexpected(state.terminal);
        return hyle::compute::CompletionPoll::kComplete;
      },
      [](void *opaque) noexcept {
        auto &state = *static_cast<Sources *>(opaque);
        for (std::size_t i = 0; i < state.count; ++i)
          state.record(state.completions[i]->wait());
        state.record(state.update.wait());
        std::lock_guard lock(state.mutex);
        return state.terminal;
      },
      [](void *opaque) noexcept {
        return static_cast<Sources *>(opaque)->update.request_cancel();
      },
      [](void *opaque) noexcept {
        static_cast<Sources *>(opaque)->update = {};
      },
      [](void *opaque, Lifetime retirement) noexcept {
        auto &state = *static_cast<Sources *>(opaque);
        state.retirement = retirement;
        state.remaining.store(state.count, std::memory_order_release);
        for (std::size_t i = 0; i < state.count; ++i) {
          auto &node = state.nodes[i];
          node.owner = &state;
          node.retention.own = {&node,
            [](void *pointer, const hyle::Status &status) noexcept {
              static_cast<Node *>(pointer)->owner->record(status);
            },
            [](void *pointer) noexcept {
              auto &source = *static_cast<Node *>(pointer)->owner;
              if (source.remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
                return;
              hyle::Status terminal;
              { std::lock_guard lock(source.mutex); terminal = source.terminal; }
              const auto lifetime = source.retirement;
              lifetime.terminal(lifetime.state, terminal);
              lifetime.release(lifetime.state); // may release the source owner
            }};
          HYLE_FATAL_ASSERT(hyle::compute::detail::CompletionAccess::attach_retention(
              *state.completions[i], node.retention).ok());
        }
      },
      [](void *opaque) noexcept {
        auto keep = std::move(static_cast<Sources *>(opaque)->accepted);
      }};
  }
};

template<class T, class Update> class Outcome final {
  using State = Sources<Update>;
public:
  Outcome(hyle::compute::OwnedResult<T> result, std::shared_ptr<State> state)
      : state_(std::move(state)), result_(std::move(result)) {}
  Outcome(const Outcome &) = delete;
  Outcome &operator=(const Outcome &) = delete;
  Outcome(Outcome &&) = default;
  Outcome &operator=(Outcome &&) = default;

  template<class Access>
  static hyle::Result<Outcome> take(hyle::compute::OwnedResult<T> &&result,
                                   Update &&update) {
    if (!result.valid() || !update.valid())
      return std::unexpected(hyle::Status::InvalidArgument("empty outcome/source"));
    auto state = std::make_shared<State>(std::move(update));
    HYLE_ASSIGN_OR_RETURN(state->count, Access::retirement_sources(state->update, {}));
    if (!state->count || state->count > state->completions.size())
      return std::unexpected(hyle::Status::ResourceExhausted("fixture source count"));
    HYLE_ASSIGN_OR_RETURN(auto filled, Access::retirement_sources(
        state->update, std::span(state->completions).first(state->count)));
    HYLE_FATAL_ASSERT(filled == state->count);
    if (state->count > 1) {
      std::array<typename State::Completion *, 1> short_buffer{state->completions[0]};
      auto rejected = Access::retirement_sources(state->update, short_buffer);
      HYLE_FATAL_ASSERT(!rejected && short_buffer[0] == state->completions[0]);
    }
    HYLE_ASSIGN_OR_RETURN(auto slot, hyle::compute::detail::DependentCompletion::create(
        hyle::mem::system_allocator()));
    state->accepted = state;
    auto &original = hyle::compute::detail::ResultAccess::completion(result);
    auto replaced = hyle::compute::detail::ResultAccess::replace_completion(
        result, slot.publish(std::move(original), state->protocol()));
    HYLE_FATAL_ASSERT(!replaced.valid());
    return Outcome(std::move(result), std::move(state));
  }
  const hyle::compute::Completion &completion() const noexcept {
    return result_.completion();
  }
  hyle::Result<hyle::compute::ResultLease<T>> result() { return result_.result(); }
  hyle::Result<hyle::compute::ResultLease<T>> wait() {
    const auto terminal = completion().wait(); // Actual Graph consumption shape.
    if (!terminal.ok()) return std::unexpected(terminal);
    return result_.wait();
  }
  hyle::Status cancel_source() { return state_->update.request_cancel(); }
  std::weak_ptr<State> watch() const { return state_; }
private:
  std::shared_ptr<State> state_;
  hyle::compute::OwnedResult<T> result_;
};
} // namespace hyle_g03_outcome_probe
