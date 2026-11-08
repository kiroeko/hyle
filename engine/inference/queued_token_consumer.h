#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "token_consumer.h"

namespace hyle::inference {

// A bounded transport handoff, not a model-output or session owner. The model
// owner alone calls start/try_write/finished/wait; transport threads may read,
// request cancellation, release retired slots and shut down. Each accepted
// token is copied once into this queue; the existing output record and native
// completion/retirement remain owned by the request loop and model.
class QueuedTokenConsumer final {
public:
  enum class ReadState { kToken, kCompleted, kCancelled, kClosed, kStale };
  struct ReadResult final { ReadState state; std::int32_t token = 0; };

  QueuedTokenConsumer(std::size_t slots, std::size_t pending) : slots_(slots), pending_(pending) {
    if (!slots || !pending) throw std::invalid_argument("token queue requires positive slot and pending capacities");
    for (auto &slot : slots_) slot.tokens.resize(pending);
  }
  QueuedTokenConsumer(const QueuedTokenConsumer &) = delete;
  QueuedTokenConsumer &operator=(const QueuedTokenConsumer &) = delete;

  std::size_t max_pending() const noexcept { return pending_; }

  // Called once after the request owner enqueues its actual key. A slot cannot
  // be reused until its former terminal has been observed, drained and released.
  bool start(RequestKey key) noexcept {
    std::lock_guard lock(mutex_);
    if (closed_ || key.slot >= slots_.size() || !key.generation || slots_[key.slot].active) return false;
    auto &slot = slots_[key.slot];
    slot.key = key; slot.head = slot.size = 0; slot.active = true; slot.done = slot.cancel = slot.cancelled = false;
    ready_.notify_all();
    return true;
  }
  bool try_write(RequestKey key, std::int32_t token) noexcept {
    std::lock_guard lock(mutex_);
    auto *slot = find(key);
    if (!slot || closed_ || slot->cancel || slot->done || slot->size == pending_) return false;
    slot->tokens[(slot->head + slot->size) % pending_] = token;
    ++slot->size; ready_.notify_all(); return true;
  }
  bool cancel_requested(RequestKey key) noexcept {
    std::lock_guard lock(mutex_);
    const auto *slot = find(key);
    return closed_ || !slot || slot->cancel;
  }
  void finished(RequestKey key, bool cancelled) noexcept {
    std::lock_guard lock(mutex_);
    if (auto *slot = find(key); slot && !slot->done) {
      slot->done = true; slot->cancelled = cancelled; ready_.notify_all();
    }
  }
  // The predicate is set while holding the same mutex as reader dequeue and
  // cancellation. A wake between a refused write and this call is not lost.
  void wait() noexcept {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [&] { return closed_ || producer_wake_; });
    producer_wake_ = false;
  }

  std::optional<RequestKey> wait_for_start(std::size_t index) {
    std::unique_lock lock(mutex_);
    if (index >= slots_.size()) throw std::out_of_range("token queue slot is out of range");
    ready_.wait(lock, [&] { return closed_ || slots_[index].active; });
    if (!slots_[index].active) return std::nullopt;
    return slots_[index].key;
  }
  ReadResult read(RequestKey key) {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [&] {
      const auto *slot = find(key);
      return !slot || slot->size || slot->done || closed_;
    });
    auto *slot = find(key);
    if (!slot) return {ReadState::kStale};
    if (slot->size) {
      const auto token = slot->tokens[slot->head];
      slot->head = (slot->head + 1) % pending_; --slot->size;
      wake_producer();
      return {ReadState::kToken, token};
    }
    if (slot->done) return {slot->cancelled ? ReadState::kCancelled : ReadState::kCompleted};
    return {ReadState::kClosed};
  }
  bool request_cancel(RequestKey key) noexcept {
    std::lock_guard lock(mutex_);
    auto *slot = find(key);
    if (!slot || slot->done) return false;
    slot->cancel = true; wake_producer(); return true;
  }
  bool release(RequestKey key) noexcept {
    std::lock_guard lock(mutex_);
    auto *slot = find(key);
    if (!slot || !slot->done || slot->size) return false;
    slot->active = false; ready_.notify_all(); return true;
  }
  // Shutdown wakes readers and the serialized owner. It requests cancellation;
  // it never fabricates native completion or marks an active request finished.
  void shutdown() noexcept {
    std::lock_guard lock(mutex_);
    closed_ = true; wake_producer();
  }
private:
  struct Slot final {
    RequestKey key{};
    std::vector<std::int32_t> tokens;
    std::size_t head = 0, size = 0;
    bool active = false, done = false, cancel = false, cancelled = false;
  };
  Slot *find(RequestKey key) noexcept {
    if (key.slot >= slots_.size()) return nullptr;
    auto &slot = slots_[key.slot];
    return slot.active && slot.key == key ? &slot : nullptr;
  }
  void wake_producer() noexcept { producer_wake_ = true; ready_.notify_all(); }
  std::vector<Slot> slots_;
  const std::size_t pending_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool closed_ = false, producer_wake_ = false;
};

static_assert(TokenConsumer<QueuedTokenConsumer>);

} // namespace hyle::inference
