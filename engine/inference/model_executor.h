#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "request_progress.h"
#include "request_admission.h"
#include "generation_input.h"
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
          else { (void)p; (void)key; return true; }
        }),
        write_([](void *p, RequestKey key, std::int32_t token) noexcept {
          return static_cast<T *>(p)->try_write(key, token);
        }),
        turn_end_([](void *p, RequestKey key, std::size_t turn) noexcept {
          if constexpr (requires(T &c) { { c.try_turn_end(key, turn) } noexcept -> std::same_as<bool>; })
            return static_cast<T *>(p)->try_turn_end(key, turn);
          else { (void)p; (void)key; (void)turn; return false; }
        }),
        supports_turn_ends_(requires(T &c, RequestKey key, std::size_t turn) {
          { c.try_turn_end(key, turn) } noexcept -> std::same_as<bool>;
        }),
        cancel_([](void *p, RequestKey key) noexcept { return static_cast<T *>(p)->cancel_requested(key); }),
        finished_([](void *p, RequestKey key, bool cancelled) noexcept {
          static_cast<T *>(p)->finished(key, cancelled);
        }), fail_([](void *p, RequestKey key, std::exception_ptr error) noexcept {
          if constexpr (requires(T &c) { { c.fail(key, error) } noexcept -> std::same_as<bool>; })
            return static_cast<T *>(p)->fail(key, std::move(error));
          else { (void)p; (void)key; (void)error; return false; }
        }), wait_([](void *p) noexcept { static_cast<T *>(p)->wait(); }) {}
  std::size_t max_pending() const noexcept { return pending_(object_); }
  bool start(RequestKey key) noexcept { return start_(object_, key); }
  bool try_write(RequestKey key, std::int32_t token) noexcept { return write_(object_, key, token); }
  bool supports_turn_ends() const noexcept { return supports_turn_ends_; }
  bool try_turn_end(RequestKey key, std::size_t turn) noexcept { return turn_end_(object_, key, turn); }
  bool cancel_requested(RequestKey key) noexcept { return cancel_(object_, key); }
  void finished(RequestKey key, bool cancelled) noexcept { finished_(object_, key, cancelled); }
  // A request-local rejection before its next turn submits model work.
  // The original request owner still cancels and observes real retirement.
  bool fail(RequestKey key, std::exception_ptr error) noexcept { return fail_(object_, key, std::move(error)); }
  void wait() noexcept { wait_(object_); }
private:
  void *object_;
  std::size_t (*pending_)(void *) noexcept;
  bool (*start_)(void *, RequestKey) noexcept;
  bool (*write_)(void *, RequestKey, std::int32_t) noexcept;
  bool (*turn_end_)(void *, RequestKey, std::size_t) noexcept;
  bool supports_turn_ends_;
  bool (*cancel_)(void *, RequestKey) noexcept;
  void (*finished_)(void *, RequestKey, bool) noexcept;
  bool (*fail_)(void *, RequestKey, std::exception_ptr) noexcept;
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
  GenerationInputRef *input = nullptr;
};

// Immutable metadata from the selected model adapter, available before
// admission. Stop IDs are model policy, not guessed by transport or tokenizer.
struct ExecutionProfileDescription final {
  std::string name = "default";
  std::string algorithm;
  std::size_t max_draft_tokens = 0;
  // Self is implicit. Additional compatible identities must be symmetric;
  // this is residency compatibility, not permission to combine device rows.
  std::vector<std::size_t> co_resident_with;
  // Actual accepted input kinds of this prepared program. These are public
  // semantic capabilities, not asset role names or model-specific layouts.
  std::vector<std::string> input_modalities{"text"};
};

struct ModelDescription final {
  std::size_t vocabulary = 0, context = 0;
  std::vector<std::int32_t> stop_ids;
  std::vector<ExecutionProfileDescription> execution_profiles{ExecutionProfileDescription{}};
  // Opaque immutable source identity from the provider, not a model family name.
  std::string source_identity;
};

inline bool execution_profiles_co_resident(const ModelDescription &model, std::size_t a, std::size_t b) noexcept {
  if (a >= model.execution_profiles.size() || b >= model.execution_profiles.size()) return false;
  if (a == b) return true;
  const auto &allowed = model.execution_profiles[a].co_resident_with;
  return std::find(allowed.begin(), allowed.end(), b) != allowed.end();
}

inline void validate_model_description(const ModelDescription &model) {
  if (model.execution_profiles.empty()) throw std::invalid_argument("model has no prepared execution profile");
  for (std::size_t a = 0; a < model.execution_profiles.size(); ++a) {
    const auto &profile = model.execution_profiles[a];
    if (profile.name.empty()) throw std::invalid_argument("model execution profile has no name");
    if (profile.input_modalities.empty()) throw std::invalid_argument("model execution profile has no input modality");
    for (std::size_t i = 0; i < profile.input_modalities.size(); ++i) {
      if (profile.input_modalities[i].empty()) throw std::invalid_argument("empty execution-profile input modality");
      for (std::size_t j = 0; j < i; ++j)
        if (profile.input_modalities[i] == profile.input_modalities[j])
          throw std::invalid_argument("duplicate execution-profile input modality");
    }
    for (std::size_t b = 0; b < a; ++b)
      if (profile.name == model.execution_profiles[b].name) throw std::invalid_argument("duplicate model execution profile name");
    for (auto b : profile.co_resident_with)
      if (b >= model.execution_profiles.size() || !execution_profiles_co_resident(model, b, a))
        throw std::invalid_argument("invalid or asymmetric execution-profile compatibility");
  }
}

// One prepared model execution owner. Implementations report their actual
// supported configuration at preparation/admission; this common boundary does
// not name a model, platform, quantization or decoding algorithm. execute is
// serialized; normal return requires each original request to have really retired.
// An exception is not evidence of native retirement or permission to reuse.
class ModelExecutor {
public:
  virtual ~ModelExecutor() = default;
  virtual ModelDescription describe() const = 0;
  // Evaluated at construction, before execution. A supporting owner polls
  // the same input source for new requests in genuinely reusable slots.
  virtual bool supports_dynamic_admission() const noexcept { return false; }
  // Whether the same serialized owner can keep requests with different
  // prepared sampling indices active. Actual device batches remain selected
  // by the implementation's existing compatibility checks.
  virtual bool supports_concurrent_sampling_profiles() const noexcept { return false; }
  // Serialized, side-effect-free admission; reject a request with
  // RequestRejected before any model work, not with a fabricated retirement.
  virtual void validate(GenerationBatch batch) = 0;
  virtual void execute(GenerationBatch batch, TokenConsumerRef &consumer) = 0;
};

} // namespace hyle::inference
