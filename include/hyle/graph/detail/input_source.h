#pragma once

#include <span>
#include <utility>

#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

enum class InputDependencyRoute : unsigned char { kDeviceWait, kHostWait };

// Same-build typed adapter for an immutable external tensor owner. DeviceWait
// is legal only when enqueue transfers the source's complete terminal contract
// into the original admitted Queue sequence. Other producers use HostWait.
// Callbacks may run on different consumer lanes and must serialize their own
// producer observations. They never create another Graph execution runtime.
struct InputSourceVTable final {
  void (*retain)(void *) noexcept;
  void (*release)(void *) noexcept;
  Result<InputDependencyRoute> (*route)(
      void *, const compute::detail::QueueRef *) noexcept;
  Result<compute::CompletionPoll> (*poll)(void *) noexcept;
  Status (*wait)(void *) noexcept;
  compute::detail::EnqueueResult (*enqueue)(
      void *, compute::detail::QueueSequence &) noexcept;
  // Called once after this consumer's successfully imported native suffix has
  // actually retired. It preserves diagnostics not carried by native events;
  // it must not wait on work this consumer did not successfully import.
  Status (*consumer_retired)(void *) noexcept;
};

class InputSource final {
public:
  InputSource() noexcept = default;
  // The stable identity names the actual immutable producer, independently
  // of an adapter's intrusive/shared ownership representation. All states
  // with the same (vtable, identity) must describe that complete producer.
  // The retained state keeps the identity alive; it is never a payload address.
  HYLE_GRAPH_API static Result<InputSource>
  acquire(void *state, const InputSourceVTable *vtable,
          const void *identity) noexcept;
  InputSource(const InputSource &other) noexcept
      : state_(other.state_), vtable_(other.vtable_), identity_(other.identity_) {
    if (state_) vtable_->retain(state_);
  }
  InputSource &operator=(const InputSource &other) noexcept {
    if (this != &other) {
      InputSource copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  InputSource(InputSource &&other) noexcept
      : state_(std::exchange(other.state_, nullptr)),
        vtable_(std::exchange(other.vtable_, nullptr)),
        identity_(std::exchange(other.identity_, nullptr)) {}
  InputSource &operator=(InputSource &&other) noexcept {
    if (this != &other) {
      reset();
      state_ = std::exchange(other.state_, nullptr);
      vtable_ = std::exchange(other.vtable_, nullptr);
      identity_ = std::exchange(other.identity_, nullptr);
    }
    return *this;
  }
  ~InputSource() { reset(); }
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  void *state() const noexcept { return state_; }
  const InputSourceVTable *vtable() const noexcept { return vtable_; }
  const void *identity() const noexcept { return identity_; }
  bool same_owner(const InputSource &other) const noexcept {
    return identity_ == other.identity_ && vtable_ == other.vtable_;
  }
  void reset() noexcept {
    auto *state = std::exchange(state_, nullptr);
    const auto *vtable = std::exchange(vtable_, nullptr);
    identity_ = nullptr;
    if (state) vtable->release(state);
  }

private:
  InputSource(void *state, const InputSourceVTable *vtable,
              const void *identity) noexcept
      : state_(state), vtable_(vtable), identity_(identity) {}
  void *state_ = nullptr;
  const InputSourceVTable *vtable_ = nullptr;
  const void *identity_ = nullptr;
};

struct InputSourceBinding final {
  ValueId value{};
  compute::ConstTensorView tensor{};
  InputSource source{};
};

struct InputAccess final {
  // Cold binding retains the actual immutable owner without taking the
  // Invocation's typed resource payload slot. The tensor keeps its real
  // MemoryResource identity, and may still await its producer's completion.
  HYLE_GRAPH_API static Status bind(Invocation &, ValueId,
      compute::ConstTensorView, InputSource) noexcept;
  // Validate all incoming bindings against their final ranges and immutable
  // owners before changing any binding. Failure preserves the old bindings
  // and preparation; successful commit cannot allocate or fail.
  HYLE_GRAPH_API static Status bind(Invocation &,
      std::span<const InputSourceBinding>) noexcept;
};

} // namespace hyle::graph::detail
