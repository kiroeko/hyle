#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "request_progress.h"
#include "request_stop_policy.h"
#include "token_consumer.h"

namespace hyle::inference {

// A borrowed, non-allocating type erasure at the model/request boundary. It
// forwards transport events only; no Graph, kernel, cache or session operation
// passes through it. The underlying consumer outlives the synchronous call.
class TokenConsumerRef final {
public:
  template<TokenConsumer T> requires (!std::same_as<std::remove_cvref_t<T>, TokenConsumerRef>)
  explicit TokenConsumerRef(T &consumer) noexcept
      : object_(&consumer), pending_([](void *p) noexcept { return static_cast<T *>(p)->max_pending(); }),
        start_([](void *p, RequestKey key) noexcept {
          if constexpr (requires(T &c) { { c.start(key) } noexcept -> std::same_as<bool>; })
            return static_cast<T *>(p)->start(key);
          else return true;
        }),
        write_([](void *p, RequestKey key, std::int32_t token) noexcept {
          return static_cast<T *>(p)->try_write(key, token);
        }),
        cancel_([](void *p, RequestKey key) noexcept { return static_cast<T *>(p)->cancel_requested(key); }),
        finished_([](void *p, RequestKey key, bool cancelled) noexcept {
          static_cast<T *>(p)->finished(key, cancelled);
        }), wait_([](void *p) noexcept { static_cast<T *>(p)->wait(); }) {}
  std::size_t max_pending() const noexcept { return pending_(object_); }
  bool start(RequestKey key) noexcept { return start_(object_, key); }
  bool try_write(RequestKey key, std::int32_t token) noexcept { return write_(object_, key, token); }
  bool cancel_requested(RequestKey key) noexcept { return cancel_(object_, key); }
  void finished(RequestKey key, bool cancelled) noexcept { finished_(object_, key, cancelled); }
  void wait() noexcept { wait_(object_); }
private:
  void *object_;
  std::size_t (*pending_)(void *) noexcept;
  bool (*start_)(void *, RequestKey) noexcept;
  bool (*write_)(void *, RequestKey, std::int32_t) noexcept;
  bool (*cancel_)(void *, RequestKey) noexcept;
  void (*finished_)(void *, RequestKey, bool) noexcept;
  void (*wait_)(void *) noexcept;
};
static_assert(TokenConsumer<TokenConsumerRef>);

// One already-admitted finite cohort. Its owner keeps all borrowed request and
// policy data stable until execute returns. This lower-level contract is not a
// complete service scheduler: it neither prevents other ingress designs nor
// claims persistence of a session across separate execute calls.
struct GenerationBatch final {
  std::span<const GenerationSessionSchedule> sessions;
  std::span<const std::size_t> sampling_indices{};
  std::span<const RequestStopPolicy> stop_policies{};
};

// An admission rejection is known to precede native submission and state
// mutation. Other exceptions do not imply that the model is reusable.
class RequestRejected final : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

// Immutable metadata from the selected model adapter, available before
// admission. Stop IDs are model policy, not guessed by transport or tokenizer.
struct ModelDescription final {
  std::size_t vocabulary = 0, context = 0;
  std::vector<std::int32_t> stop_ids;
};

// One prepared model execution owner. Implementations report their actual
// supported configuration at preparation/admission; this common boundary does
// not name a model, platform, quantization or decoding algorithm. execute is
// serialized; normal return requires each original request to have really retired.
// An exception is not evidence of native retirement or permission to reuse.
class ModelExecutor {
public:
  virtual ~ModelExecutor() = default;
  virtual ModelDescription describe() const = 0;
  // Serialized, side-effect-free admission; reject a request with
  // RequestRejected before any model work, not with a fabricated retirement.
  virtual void validate(GenerationBatch batch) = 0;
  virtual void execute(GenerationBatch batch, TokenConsumerRef &consumer) = 0;
};

} // namespace hyle::inference
