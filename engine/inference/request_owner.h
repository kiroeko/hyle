#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "generation_input.h"
#include "request_admission.h"
#include "request_loop.h"
#include "token_delivery.h"

namespace hyle::inference {

// The transport decides whether it can preserve requested turn boundaries.
// Model math, native work and request-state preparation are not involved.
template<class Consumer>
void validate_turn_delivery(bool requested, Consumer &consumer) {
  if (!requested) return;
  if constexpr (std::is_null_pointer_v<std::remove_cvref_t<Consumer>>) {
    throw RequestRejected("turn boundaries require an output consumer");
  } else if constexpr (requires { { consumer.supports_turn_ends() } noexcept -> std::same_as<bool>; }) {
    if (!consumer.supports_turn_ends())
      throw RequestRejected("consumer does not support requested turn boundaries");
  } else if constexpr (!requires(RequestKey key, std::size_t turn) {
      { consumer.try_turn_end(key, turn) } noexcept -> std::same_as<bool>;
    }) {
    throw RequestRejected("consumer does not support requested turn boundaries");
  }
}

// Delivery storage for one invocation of the original request owner. Size it
// before binding any request, and replace a slot only after its original key
// retires and its consumer is notified. Deque storage keeps bound windows
// stable. This class owns no model state, identity allocator or native work.
class RequestOutputSlots final {
public:
  explicit RequestOutputSlots(std::size_t count = 0) : slots_(count) {}
  RequestOutputSlots(const RequestOutputSlots &) = delete;
  RequestOutputSlots &operator=(const RequestOutputSlots &) = delete;
  void resize(std::size_t count) { slots_.resize(count); }
  void prepare(std::size_t slot, std::size_t maximum_pending, std::size_t maximum_turns = 0) {
    slots_[slot].emplace(maximum_pending, maximum_turns);
  }
  template<TokenConsumer Consumer>
  void start(RequestLoop &loop, std::size_t slot, RequestKey &key, Consumer &consumer) {
    auto &delivery = *slots_[slot];
    key = loop.enqueue(slot, &delivery.window());
    delivery.bind(loop, key);
    if constexpr (requires { { consumer.start(key) } noexcept -> std::same_as<bool>; })
      if (!consumer.start(key))
        throw std::runtime_error("output consumer refused its original request identity");
  }
  template<class Adapter, TokenConsumer Consumer>
  bool service(RequestLoop &loop, Adapter &adapter, Consumer &consumer) {
    bool advanced = false;
    for (auto &delivery : slots_) if (delivery)
      advanced = delivery->service(loop, adapter, consumer) || advanced;
    return advanced;
  }
  bool cancelled(std::size_t slot) const noexcept { return slots_[slot] && slots_[slot]->cancelled(); }
  std::size_t cancelled_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(),
        [](const auto &delivery) { return delivery && delivery->cancelled(); }));
  }
private:
  std::deque<std::optional<TokenDelivery>> slots_;
};

// The original loop/input source decide which retired slot can be reused.
// validate is side-effect-free; only its admission errors become request-local
// rejections. prepare installs model-owned payload and returns its stable key
// field. Preparation/start failures propagate; they never fabricate retirement.
// Both stages bind statically, outside graph/kernel execution.
template<TokenConsumer Consumer, class Validate, class Prepare>
std::size_t admit_waiting_requests(RequestLoop &loop, GenerationInputRef &input,
    Consumer &consumer, RequestOutputSlots &output, std::size_t live_turn_bound,
    Validate &&validate, Prepare &&prepare) {
  std::size_t admitted = 0;
  while (const auto slot = loop.free_slot([&](std::size_t index) { return input.can_admit(index); })) {
    auto next = input.poll_admission(*slot);
    if (!next) break;
    try {
      validate(*next);
      validate_turn_delivery(next->schedule.reports_turn_ends(), consumer);
    } catch (const std::invalid_argument &error) {
      input.reject_admission(*slot, std::make_exception_ptr(RequestRejected(error.what())));
      continue;
    } catch (const std::out_of_range &error) {
      input.reject_admission(*slot, std::make_exception_ptr(RequestRejected(error.what())));
      continue;
    }
    const auto maximum_turns = next->schedule.reports_turn_ends()
        ? (next->schedule.accepts_live_input() ? live_turn_bound : next->schedule.turns.size()) : 0;
    auto &key = prepare(*slot, std::move(*next));
    output.prepare(*slot, consumer.max_pending(), maximum_turns);
    output.start(loop, *slot, key, consumer);
    ++admitted;
  }
  return admitted;
}

struct RequestOwnerInputProgress final {
  bool arrived = false;
  bool waiting = false;
};

// Common host orchestration only. Callers retain model state, native work,
// actual retirement, input and output owners. Template binding keeps the
// existing direct calls and does not introduce a runtime execution interface.
template<class ServiceOutput, class Pump, class Observe, class AwaitRetirement,
         class PollInput, class WaitIdle>
void drive_request_owner(std::size_t &active, ServiceOutput &&service_output, Pump &&pump,
                         Observe &&observe, AwaitRetirement &&await_retirement,
                         PollInput &&poll_input, WaitIdle &&wait_idle) {
  while (active) {
    bool output_progress = service_output();
    const auto pumped = pump();
    observe(pumped);
    output_progress = service_output() || output_progress;
    await_retirement(pumped);
    const auto input = poll_input(pumped);
    if (!pumped.executed && !pumped.waiting_retirement && !input.arrived && !output_progress && active) {
      if (wait_idle(pumped, input)) continue;
      throw std::runtime_error("generation admission has no runnable resident request");
    }
  }
}

} // namespace hyle::inference
