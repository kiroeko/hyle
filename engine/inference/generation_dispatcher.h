#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "model_executor.h"
#include "queued_token_consumer.h"

namespace hyle::inference {

namespace detail {
enum class IngressState { kPending, kAssigned, kStarted, kCompleted, kCancelled, kFailed, kClosed };
struct IngressRequest final {
  GenerationSessionSchedule schedule;
  RequestStopPolicy stop;
  std::size_t sampling_index = 0;
  std::mutex mutex;
  std::condition_variable changed;
  IngressState state = IngressState::kPending;
  bool cancel_intent = false;
  std::shared_ptr<QueuedTokenConsumer> output;
  std::optional<RequestKey> key;
  std::exception_ptr failure;
  bool open_input = false, input_closed = false;
  std::size_t context = 0, vocabulary = 0, reserved_context = 0;
  std::size_t emitted = 0, turn_begin = 0, completed_turns = 0;
  std::vector<std::size_t> turn_quotas;
  std::deque<GenerationTurn> pending_turns;
};
} // namespace detail

// Client-side handle for one request. A single reader consumes tokens in order;
// cancel may be called concurrently. Destroying an unfinished handle requests
// cancellation, so an abandoned transport cannot permanently hold credit.
class GenerationRequestHandle final {
public:
  enum class InputResult { kAccepted, kClosed, kInvalid, kCapacity };
  enum class EventKind { kToken, kTurnEnd, kCompleted, kCancelled, kFailed, kClosed };
  struct Event final { EventKind kind; std::int32_t token = 0; std::exception_ptr failure{}; std::size_t turn = 0; };
  GenerationRequestHandle() = default;
  GenerationRequestHandle(const GenerationRequestHandle &) = delete;
  GenerationRequestHandle &operator=(const GenerationRequestHandle &) = delete;
  GenerationRequestHandle(GenerationRequestHandle &&other) noexcept : request_(std::move(other.request_)) {}
  GenerationRequestHandle &operator=(GenerationRequestHandle &&other) noexcept {
    if (this != &other) { cancel(); request_ = std::move(other.request_); }
    return *this;
  }
  ~GenerationRequestHandle() { cancel(); }

  // Add input only to this live request; the model owner later admits it at
  // its original idle turn boundary. Context reservations are conservative
  // until an early-stopped turn actually publishes its unused quota.
  InputResult append(GenerationTurn turn) {
    if (!request_) return InputResult::kClosed;
    std::shared_ptr<QueuedTokenConsumer> output;
    {
      std::lock_guard lock(request_->mutex);
      auto &r = *request_;
      if (!r.open_input || r.input_closed || r.cancel_intent ||
          (r.state != detail::IngressState::kPending && r.state != detail::IngressState::kAssigned &&
           r.state != detail::IngressState::kStarted)) return InputResult::kClosed;
      if (!turn.max_outputs || turn.prompt.size() >= r.context ||
          std::ranges::any_of(turn.prompt, [&](auto id) { return id < 0 || std::size_t(id) >= r.vocabulary; }))
        return InputResult::kInvalid;
      if (r.turn_quotas.size() >= r.context || turn.prompt.size() > r.context - r.reserved_context ||
          turn.max_outputs > r.context - r.reserved_context - turn.prompt.size()) return InputResult::kCapacity;
      const auto quota = turn.max_outputs, cost = turn.prompt.size() + quota;
      r.pending_turns.push_back(std::move(turn));
      try { r.turn_quotas.push_back(quota); }
      catch (...) { r.pending_turns.pop_back(); throw; }
      r.reserved_context += cost; output = r.output;
    }
    if (output) output->notify_input();
    return InputResult::kAccepted;
  }
  bool close_input() noexcept {
    if (!request_) return false;
    std::shared_ptr<QueuedTokenConsumer> output;
    {
      std::lock_guard lock(request_->mutex);
      auto &r = *request_;
      if (!r.open_input || r.input_closed || r.cancel_intent ||
          (r.state != detail::IngressState::kPending && r.state != detail::IngressState::kAssigned &&
           r.state != detail::IngressState::kStarted)) return false;
      r.input_closed = true; output = r.output;
    }
    if (output) output->notify_input();
    return true;
  }

  bool cancel() noexcept {
    if (!request_) return false;
    std::shared_ptr<QueuedTokenConsumer> output;
    std::optional<RequestKey> key;
    {
      std::lock_guard lock(request_->mutex);
      switch (request_->state) {
        case detail::IngressState::kPending:
          request_->state = detail::IngressState::kCancelled;
          request_->changed.notify_all(); return true;
        case detail::IngressState::kAssigned:
          request_->cancel_intent = true; return true;
        case detail::IngressState::kStarted:
          request_->cancel_intent = true;
          output = request_->output; key = request_->key; break;
        default: return false;
      }
    }
    return output && key && output->request_cancel(*key);
  }

  Event read() {
    if (!request_) throw std::logic_error("generation request handle is empty");
    std::shared_ptr<QueuedTokenConsumer> output;
    RequestKey key;
    {
      std::unique_lock lock(request_->mutex);
      request_->changed.wait(lock, [&] {
        return request_->state != detail::IngressState::kPending &&
            request_->state != detail::IngressState::kAssigned;
      });
      switch (request_->state) {
        case detail::IngressState::kCompleted: return {EventKind::kCompleted};
        case detail::IngressState::kCancelled: return {EventKind::kCancelled};
        case detail::IngressState::kFailed: return {EventKind::kFailed, 0, request_->failure};
        case detail::IngressState::kClosed: return {EventKind::kClosed};
        case detail::IngressState::kStarted:
          output = request_->output; key = *request_->key; break;
        default: throw std::logic_error("invalid ingress request state");
      }
    }
    const auto item = output->read(key);
    if (item.state == QueuedTokenConsumer::ReadState::kToken) return {EventKind::kToken, item.token};
    if (item.state == QueuedTokenConsumer::ReadState::kTurnEnd) return {EventKind::kTurnEnd, 0, {}, item.turn};
    std::lock_guard lock(request_->mutex);
    if (item.state == QueuedTokenConsumer::ReadState::kCompleted) {
      request_->state = detail::IngressState::kCompleted;
      (void)output->release(key);
      return {EventKind::kCompleted};
    }
    if (item.state == QueuedTokenConsumer::ReadState::kCancelled) {
      request_->state = detail::IngressState::kCancelled;
      (void)output->release(key);
      return {EventKind::kCancelled};
    }
    if (request_->failure) {
      request_->state = detail::IngressState::kFailed;
      return {EventKind::kFailed, 0, request_->failure};
    }
    request_->state = detail::IngressState::kClosed;
    return {EventKind::kClosed};
  }
private:
  explicit GenerationRequestHandle(std::shared_ptr<detail::IngressRequest> request) : request_(std::move(request)) {}
  std::shared_ptr<detail::IngressRequest> request_;
  friend class GenerationDispatcher;
};

// Thread-safe ingress for the existing serialized model owner. Its RequestLoop
// remains the only active-request/retirement owner. A supporting executor may
// admit waiting requests into actually retired and transport-released slots;
// otherwise arrivals wait for the next execute call. Compatible arrivals keep
// FIFO order without letting an incompatible sampling choice block free slots.
class GenerationDispatcher final {
public:
  GenerationDispatcher(std::unique_ptr<ModelExecutor> executor, std::size_t batch_limit,
                       std::size_t waiting_limit, std::size_t output_capacity)
      : executor_(std::move(executor)), batch_limit_(batch_limit), waiting_limit_(waiting_limit),
        output_capacity_(output_capacity) {
    if (!executor_ || !batch_limit || !waiting_limit || !output_capacity)
      throw std::invalid_argument("generation ingress requires a model and positive capacities");
    description_ = executor_->describe();
    dynamic_admission_ = executor_->supports_dynamic_admission();
    worker_ = std::thread([this] { work(); });
  }
  GenerationDispatcher(const GenerationDispatcher &) = delete;
  GenerationDispatcher &operator=(const GenerationDispatcher &) = delete;
  ~GenerationDispatcher() { shutdown(); }

  GenerationRequestHandle submit(GenerationSessionSchedule schedule, RequestStopPolicy stop = {},
                                 std::size_t sampling_index = 0) {
    if (schedule.turns.empty() || schedule.turns.front().prompt.empty())
      throw std::invalid_argument("generation request requires an initial prompt");
    for (const auto &turn : schedule.turns) if (!turn.max_outputs)
      throw std::invalid_argument("generation request requires positive output limits");
    auto request = std::make_shared<detail::IngressRequest>();
    if (schedule.accepts_live_input()) {
      if (schedule.turns.size() != 1 || !description_.context || !description_.vocabulary)
        throw std::invalid_argument("live generation needs one initial turn, model limits and turn boundaries");
      const auto &first = schedule.turns.front();
      if (first.prompt.size() > description_.context || first.max_outputs - 1 > description_.context - first.prompt.size() ||
          std::ranges::any_of(first.prompt, [&](auto id) { return id < 0 || std::size_t(id) >= description_.vocabulary; }))
        throw std::invalid_argument("initial live turn exceeds model limits");
      request->open_input = true; request->context = description_.context; request->vocabulary = description_.vocabulary;
      request->reserved_context = first.prompt.size() + first.max_outputs - 1;
      request->turn_quotas.reserve(description_.context); request->turn_quotas.push_back(first.max_outputs);
    }
    request->schedule = std::move(schedule); request->stop = std::move(stop);
    request->sampling_index = sampling_index;
    std::shared_ptr<QueuedTokenConsumer> wake_active;
    {
      std::lock_guard lock(mutex_);
      if (closed_) throw std::runtime_error("generation ingress is closed");
      std::erase_if(waiting_, [](const auto &queued) {
        std::lock_guard queued_lock(queued->mutex);
        return queued->state != detail::IngressState::kPending;
      });
      if (waiting_.size() >= waiting_limit_) throw std::runtime_error("generation ingress queue is full");
      waiting_.push_back(request);
      if (dynamic_admission_) wake_active = active_;
    }
    if (wake_active) wake_active->notify_input();
    ready_.notify_one();
    return GenerationRequestHandle(std::move(request));
  }

  // Shutdown abandons queued work and asks active consumers to cancel. It
  // joins the original model owner; it never reports a pending native call as
  // retired merely because shutdown or a transport disconnect was requested.
  void shutdown() noexcept {
    std::call_once(shutdown_once_, [&] {
      std::shared_ptr<QueuedTokenConsumer> active;
      {
        std::lock_guard lock(mutex_);
        closed_ = true; active = active_;
        for (const auto &request : waiting_) {
          std::lock_guard request_lock(request->mutex);
          if (request->state == detail::IngressState::kPending) {
            request->state = detail::IngressState::kClosed; request->changed.notify_all();
          }
        }
        waiting_.clear();
      }
      if (active) active->shutdown();
      ready_.notify_all();
      if (worker_.joinable()) worker_.join();
    });
  }
private:
  using Job = std::shared_ptr<detail::IngressRequest>;
  class CohortConsumer final {
  public:
    CohortConsumer(GenerationDispatcher &owner, std::shared_ptr<QueuedTokenConsumer> output,
                   std::vector<Job> &jobs, std::size_t sampling)
        : owner_(owner), output_(std::move(output)), jobs_(jobs), started_(jobs.size()),
          finished_(jobs.size()), keys_(jobs.size()), sampling_(sampling) {}
    std::size_t max_pending() const noexcept { return output_->max_pending(); }
    bool start(RequestKey key) noexcept {
      if (key.slot >= jobs_.size() || started_[key.slot] || !output_->start(key)) return false;
      auto &job = *jobs_[key.slot]; bool cancel = false;
      {
        std::lock_guard lock(job.mutex);
        if (job.state != detail::IngressState::kAssigned) return false;
        job.key = key; job.state = detail::IngressState::kStarted; cancel = job.cancel_intent;
        job.changed.notify_all();
      }
      keys_[key.slot] = key; started_[key.slot] = true;
      if (cancel) (void)output_->request_cancel(key);
      return true;
    }
    bool try_write(RequestKey key, std::int32_t token) noexcept {
      if (!output_->try_write(key, token)) return false;
      if (key.slot < jobs_.size() && jobs_[key.slot]->open_input) {
        std::lock_guard lock(jobs_[key.slot]->mutex); ++jobs_[key.slot]->emitted;
      }
      return true;
    }
    bool try_turn_end(RequestKey key, std::size_t turn) noexcept {
      if (key.slot >= jobs_.size()) return false;
      auto &job = *jobs_[key.slot];
      std::lock_guard lock(job.mutex);
      if (!output_->try_turn_end(key, turn)) return false;
      if (job.open_input) {
        if (turn != job.completed_turns || turn >= job.turn_quotas.size() || job.emitted < job.turn_begin ||
            job.emitted - job.turn_begin > job.turn_quotas[turn]) { contract_error_ = true; return true; }
        job.reserved_context -= job.turn_quotas[turn] - (job.emitted - job.turn_begin);
        job.turn_begin = job.emitted; ++job.completed_turns;
      }
      return true;
    }
    TurnInput poll_turn(RequestKey key) noexcept {
      if (key.slot >= jobs_.size() || !started_[key.slot] || keys_[key.slot] != key)
        return {TurnInputState::kClosed, {}};
      auto &job = *jobs_[key.slot]; std::lock_guard lock(job.mutex);
      if (!job.pending_turns.empty()) {
        auto turn = std::move(job.pending_turns.front()); job.pending_turns.pop_front();
        return {TurnInputState::kTurn, std::move(turn)};
      }
      return {job.input_closed ? TurnInputState::kClosed : TurnInputState::kWaiting, {}};
    }
    bool can_admit(std::size_t slot) noexcept {
      return owner_.dynamic_admission_ && slot < jobs_.size() &&
          (!jobs_[slot] || finished_[slot]) && output_->can_start(slot);
    }
    std::optional<GenerationAdmission> poll_admission(std::size_t slot) noexcept {
      if (!can_admit(slot)) return std::nullopt;
      std::lock_guard lock(owner_.mutex_);
      if (owner_.closed_) return std::nullopt;
      auto queued = owner_.waiting_.begin();
      while (queued != owner_.waiting_.end()) {
        auto job = *queued;
        std::lock_guard job_lock(job->mutex);
        if (job->state != detail::IngressState::kPending) { queued = owner_.waiting_.erase(queued); continue; }
        if (job->sampling_index != sampling_) { ++queued; continue; }
        owner_.waiting_.erase(queued); job->state = detail::IngressState::kAssigned; job->output = output_;
        jobs_[slot] = job; started_[slot] = finished_[slot] = false; keys_[slot] = {};
        return GenerationAdmission{std::move(job->schedule), std::move(job->stop), job->sampling_index};
      }
      return std::nullopt;
    }
    void reject_admission(std::size_t slot, std::exception_ptr error) noexcept {
      if (slot >= jobs_.size() || !jobs_[slot] || started_[slot]) { contract_error_ = true; return; }
      auto job = jobs_[slot];
      {
        std::lock_guard lock(job->mutex);
        if (job->state != detail::IngressState::kAssigned) { contract_error_ = true; return; }
        if (job->cancel_intent) job->state = detail::IngressState::kCancelled;
        else { job->failure = std::move(error); job->state = detail::IngressState::kFailed; }
        job->changed.notify_all();
      }
      jobs_[slot].reset();
    }
    bool cancel_requested(RequestKey key) noexcept { return output_->cancel_requested(key); }
    void finished(RequestKey key, bool cancelled) noexcept {
      if (key.slot >= jobs_.size() || !started_[key.slot] || finished_[key.slot] || keys_[key.slot] != key) {
        contract_error_ = true; return;
      }
      finished_[key.slot] = true; output_->finished(key, cancelled);
    }
    void wait() noexcept { output_->wait(); }
    bool all_finished() const noexcept {
      if (contract_error_) return false;
      for (std::size_t i = 0; i < jobs_.size(); ++i) if (jobs_[i] && (!started_[i] || !finished_[i])) return false;
      return true;
    }
  private:
    GenerationDispatcher &owner_;
    std::shared_ptr<QueuedTokenConsumer> output_;
    std::vector<Job> &jobs_;
    std::vector<bool> started_, finished_;
    std::vector<RequestKey> keys_;
    std::size_t sampling_;
    bool contract_error_ = false;
  };

  void fail(std::exception_ptr error, std::vector<Job> &selected,
            const std::shared_ptr<QueuedTokenConsumer> &output) noexcept {
    {
      std::lock_guard lock(mutex_);
      const bool stopping = closed_;
      closed_ = true;
      for (const auto &job : selected) {
        if (!job) continue;
        std::lock_guard request_lock(job->mutex);
        if (job->state == detail::IngressState::kCompleted || job->state == detail::IngressState::kCancelled ||
            job->state == detail::IngressState::kFailed || job->state == detail::IngressState::kClosed) continue;
        if (!stopping) job->failure = error;
        if (job->state == detail::IngressState::kPending || job->state == detail::IngressState::kAssigned)
          job->state = stopping ? detail::IngressState::kClosed : detail::IngressState::kFailed;
        job->changed.notify_all();
      }
      for (const auto &job : waiting_) {
        std::lock_guard request_lock(job->mutex);
        if (job->state == detail::IngressState::kPending) {
          if (!stopping) job->failure = error;
          job->state = stopping ? detail::IngressState::kClosed : detail::IngressState::kFailed;
          job->changed.notify_all();
        }
      }
      waiting_.clear();
    }
    if (output) output->shutdown();
    ready_.notify_all();
  }

  void work() noexcept {
    std::vector<Job> selected;
    std::shared_ptr<QueuedTokenConsumer> output;
    try {
      selected.reserve(batch_limit_);
      std::vector<GenerationSessionSchedule> schedules; schedules.reserve(batch_limit_);
      std::vector<RequestStopPolicy> stops; stops.reserve(batch_limit_);
      std::vector<std::size_t> sampling; sampling.reserve(batch_limit_);
      for (;;) {
        selected.clear(); schedules.clear(); stops.clear(); sampling.clear(); output.reset();
        {
          std::unique_lock lock(mutex_);
          ready_.wait(lock, [&] { return closed_ || !waiting_.empty(); });
          if (closed_) return;
          while (!waiting_.empty() && selected.size() < batch_limit_) {
            const auto &job = waiting_.front();
            std::lock_guard request_lock(job->mutex);
            if (job->state != detail::IngressState::kPending) { waiting_.pop_front(); continue; }
            if (!selected.empty() && job->sampling_index != selected.front()->sampling_index) break;
            selected.push_back(job); waiting_.pop_front();
          }
          if (selected.empty()) continue;
          output = std::make_shared<QueuedTokenConsumer>(dynamic_admission_ ? batch_limit_ : selected.size(), output_capacity_);
          for (const auto &job : selected) {
            std::lock_guard request_lock(job->mutex);
            // A request cancelled after dequeue needs no model admission.
            if (job->state != detail::IngressState::kPending) continue;
            job->state = detail::IngressState::kAssigned; job->output = output;
          }
          // Remove pre-admission cancellations without changing order.
          std::erase_if(selected, [](const Job &job) {
            std::lock_guard lock(job->mutex); return job->state != detail::IngressState::kAssigned;
          });
          if (selected.empty()) continue;
          active_ = output;
        }
        // Isolate model admission rejections before any request starts. A
        // rejection is not a native failure and must not poison healthy peers.
        for (const auto &job : selected) {
          {
            std::lock_guard lock(job->mutex);
            if (job->cancel_intent) {
              job->state = detail::IngressState::kCancelled; job->changed.notify_all(); continue;
            }
          }
          try {
            executor_->validate({std::span(&job->schedule, 1), std::span(&job->sampling_index, 1),
                std::span(&job->stop, 1)});
          } catch (const RequestRejected &) {
            std::lock_guard lock(job->mutex);
            if (job->cancel_intent) job->state = detail::IngressState::kCancelled;
            else { job->failure = std::current_exception(); job->state = detail::IngressState::kFailed; }
            job->changed.notify_all();
          }
        }
        std::erase_if(selected, [](const Job &job) {
          std::lock_guard lock(job->mutex); return job->state != detail::IngressState::kAssigned;
        });
        {
          std::lock_guard lock(mutex_);
          if (closed_) {
            for (const auto &job : selected) {
              std::lock_guard request_lock(job->mutex);
              job->state = detail::IngressState::kClosed; job->changed.notify_all();
            }
            active_.reset(); return;
          }
          if (selected.empty()) { active_.reset(); continue; }
        }
        for (const auto &job : selected) {
          schedules.push_back(std::move(job->schedule)); stops.push_back(std::move(job->stop));
          sampling.push_back(job->sampling_index);
        }
        const auto selected_sampling = sampling.front();
        if (dynamic_admission_) selected.resize(batch_limit_);
        CohortConsumer consumer(*this, output, selected, selected_sampling);
        TokenConsumerRef ref(consumer);
        GenerationInputRef input(consumer);
        executor_->execute({schedules, sampling, stops, &input}, ref);
        if (!consumer.all_finished()) throw std::logic_error("model returned without all request terminal notifications");
        {
          std::lock_guard lock(mutex_);
          if (active_ == output) active_.reset();
          if (closed_) return;
        }
      }
    } catch (...) { fail(std::current_exception(), selected, output); }
  }

  std::unique_ptr<ModelExecutor> executor_;
  ModelDescription description_;
  bool dynamic_admission_ = false;
  const std::size_t batch_limit_, waiting_limit_, output_capacity_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Job> waiting_;
  std::shared_ptr<QueuedTokenConsumer> active_;
  bool closed_ = false;
  std::once_flag shutdown_once_;
  std::thread worker_;
};

} // namespace hyle::inference
