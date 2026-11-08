#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "evaluation_failure.h"
#include "output_delivery.h"
#include "request_loop.h"
#include "token_consumer.h"

namespace hyle::inference {

// Serialized delivery of one original request's committed token record. The
// request loop remains the identity, output-credit and retirement owner; this
// object owns only its delivery cursor and the consumer notification state.
// Its address and consumer must remain stable until the final service after
// real retirement. Bind each object once to its original request identity.
class TokenDelivery final {
public:
  explicit TokenDelivery(std::size_t maximum_pending, std::size_t maximum_turns = 0)
      : window_(maximum_pending, maximum_turns) {}
  TokenDelivery(const TokenDelivery &) = delete;
  TokenDelivery &operator=(const TokenDelivery &) = delete;
  TokenDelivery(TokenDelivery &&) = delete;
  TokenDelivery &operator=(TokenDelivery &&) = delete;
  OutputDelivery &window() noexcept { return window_; }
  bool cancelled() const noexcept { return cancelled_; }
  void bind(RequestLoop &loop, RequestKey key) {
    if (owner_ || !loop.matches(key))
      throw std::logic_error("token delivery requires one active original request identity");
    key_ = key; owner_ = &loop;
  }

  template<class Adapter, TokenConsumer Consumer>
  bool service(RequestLoop &loop, Adapter &adapter, Consumer &consumer) {
    if (owner_ != &loop) throw std::logic_error("token delivery requires its original request owner");
    const auto key = key_;
    if (!loop.matches(key)) {
      if (notified_) return false;
      consumer.finished(key, cancelled_);
      notified_ = true;
      return true;
    }
    bool advanced = false;
    auto &progress = adapter.progress(key);
    const auto cancel = [&] {
      if (cancelled_ || !consumer.cancel_requested(key)) return;
      auto status = loop.cancel(key, adapter);
      if (!status.ok() && !(status.IsCancelled() && !status.has_native_error()))
        throw EvaluationFailure(std::move(status));
      cancelled_ = true; advanced = true;
    };
    cancel();
    for (;;) {
      const auto pending = window_.pending(progress.generated);
      const auto boundary = window_.next_turn_end();
      if (boundary && boundary->through < window_.acknowledged())
        throw std::logic_error("turn boundary fell behind its delivery cursor");
      const auto limit = boundary ? (std::min)(pending.size(), boundary->through - window_.acknowledged()) : pending.size();
      std::size_t accepted = 0;
      if (cancelled_) accepted = pending.size();
      else while (accepted < limit && !consumer.cancel_requested(key) &&
                  consumer.try_write(key, pending[accepted])) ++accepted;
      if (accepted) {
        if (!loop.acknowledge(key, window_.acknowledged() + accepted, adapter))
          throw std::logic_error("output acknowledgement lost its original request");
        advanced = true;
      }
      if (cancelled_ || consumer.cancel_requested(key)) break;
      if (!boundary || window_.acknowledged() != boundary->through) break;
      if constexpr (requires { { consumer.try_turn_end(key, boundary->turn) } noexcept -> std::same_as<bool>; }) {
        if (!consumer.try_turn_end(key, boundary->turn)) break;
        window_.accept_turn_end(); advanced = true;
      } else {
        throw std::logic_error("consumer does not support requested turn boundaries");
      }
    }
    // Cancellation may arrive during the last write. Abandon only this
    // consumer's undelivered prefix and retain native retirement ownership.
    cancel();
    if (cancelled_) window_.discard_turn_ends();
    if (cancelled_ && window_.acknowledged() != progress.generated.size()) {
      if (!loop.acknowledge(key, progress.generated.size(), adapter))
        throw std::logic_error("cancelled output lost its original request");
      advanced = true;
    }
    return advanced;
  }
private:
  OutputDelivery window_;
  RequestKey key_{};
  RequestLoop *owner_ = nullptr;
  bool cancelled_ = false, notified_ = false;
};

} // namespace hyle::inference
