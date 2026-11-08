#pragma once

#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "generation_dispatcher.h"

namespace hyle::inference {

// Internal local transport for a frontend process. It carries token IDs, not
// protocol JSON or model state. Frames are TAB-delimited UTF-8/ASCII lines:
// G or S id output_limit comma_separated_ids [output_limit comma_separated_ids ...];
// O id output_limit comma_separated_ids opens a live input; I id sequence
// output_limit comma_separated_ids appends one turn; Z id sequence closes its input. K acknowledges
// output credit; C cancels; Q shuts down. S and O request ordered B id turn
// events; both tokens and boundaries use credit. Later turn inputs may be empty.
// Responses are READY 2 vocabulary context stop_ids, A id, T id token, B id turn,
// F id completed|cancelled|closed,
// E id kind message, C id accepted|stale and BYE. A request initially has one
// output credit; K acknowledges delivered tokens and releases that credit.
// Request IDs are nonzero unsigned integers.
// The frontend owns tokenizer, templates, protocol and disconnect detection.
class TokenWorker final {
public:
  TokenWorker(std::unique_ptr<ModelExecutor> executor, std::size_t batch_limit,
              std::size_t waiting_limit, std::size_t output_capacity,
              std::size_t max_prompt_tokens, std::istream &input, std::ostream &output)
      : description_(executor ? executor->describe() : ModelDescription{}),
        dispatcher_(std::move(executor), batch_limit, waiting_limit, output_capacity),
        max_prompt_tokens_(max_prompt_tokens), input_(input), output_(output) {
    if (!max_prompt_tokens || !description_.vocabulary || !description_.context ||
        max_prompt_tokens > description_.context)
      throw std::invalid_argument("worker requires valid model vocabulary and context");
    for (const auto id : description_.stop_ids)
      if (id < 0 || static_cast<std::size_t>(id) >= description_.vocabulary)
        throw std::invalid_argument("worker model stop ID is outside its vocabulary");
  }
  TokenWorker(const TokenWorker &) = delete;
  TokenWorker &operator=(const TokenWorker &) = delete;
  ~TokenWorker() { close(); }

  int run() {
    std::string ready = "READY\t2\t" + std::to_string(description_.vocabulary) + "\t" +
        std::to_string(description_.context) + "\t";
    if (description_.stop_ids.empty()) ready += "-";
    else for (std::size_t index = 0; index < description_.stop_ids.size(); ++index)
      ready += (index ? "," : "") + std::to_string(description_.stop_ids[index]);
    if (!emit(ready)) return 1;
    std::string line;
    while (std::getline(input_, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      reap();
      if (line == "Q") break;
      const auto fields = split(line, '\t');
      std::uint64_t id = 0;
      if (fields.size() < 2 || !number(fields[1], id) || !id) {
        if (!emit("E\t0\tprotocol\tinvalid request identity")) break;
        continue;
      }
      if (fields[0] == "C" && fields.size() == 2) {
        const auto found = calls_.find(id);
        const bool accepted = found != calls_.end() && found->second->cancel();
        if (!emit("C\t" + std::to_string(id) + (accepted ? "\taccepted" : "\tstale"))) break;
        continue;
      }
      if (fields[0] == "K" && fields.size() == 3) {
        std::size_t count = 0;
        const auto found = calls_.find(id);
        if (!number(fields[2], count) || !count || found == calls_.end() || !found->second->acknowledge(count))
          if (!emit(error(id, "protocol", "invalid output acknowledgement"))) break;
        continue;
      }
      if (fields[0] == "Z" && fields.size() == 3) {
        std::uint64_t sequence = 0;
        if (!number(fields[2], sequence) || !sequence) {
          if (!emit(error(id, "protocol", "invalid input command sequence"))) break;
          continue;
        }
        const auto found = calls_.find(id);
        const bool closed = found != calls_.end() && found->second->handle.close_input();
        if (!emit("Z\t" + std::to_string(id) + "\t" + std::to_string(sequence) +
                  (closed ? "\taccepted" : "\tclosed"))) break;
        continue;
      }
      if (fields[0] == "I" && fields.size() == 5) {
        std::uint64_t sequence = 0; std::size_t quota = 0; std::vector<std::int32_t> tokens;
        if (!number(fields[2], sequence) || !sequence) {
          if (!emit(error(id, "protocol", "invalid input command sequence"))) break;
          continue;
        }
        const auto found = calls_.find(id);
        using Result = GenerationRequestHandle::InputResult;
        Result result = Result::kClosed;
        bool failed = false;
        try {
          if (!number(fields[3], quota) || !quota || !parse_tokens(fields[4], tokens, true)) result = Result::kInvalid;
          else if (found != calls_.end()) result = found->second->handle.append({std::move(tokens), quota});
        } catch (const std::exception &) { failed = true; }
        const auto status = failed ? "failed" : result == Result::kAccepted ? "accepted" :
            result == Result::kInvalid ? "invalid" : result == Result::kCapacity ? "capacity" : "closed";
        if (!emit("I\t" + std::to_string(id) + "\t" + std::to_string(sequence) + "\t" + status)) break;
        continue;
      }
      if ((fields[0] != "G" && fields[0] != "S" && fields[0] != "O") || fields.size() < 4 ||
          (fields.size() - 2) % 2 != 0 || (fields[0] == "O" && fields.size() != 4)) {
        if (!emit(error(id, "protocol", "unknown or malformed command"))) break;
        continue;
      }
      if (calls_.contains(id)) {
        if (!emit(error(id, "protocol", "duplicate active request identity"))) break;
        continue;
      }
      std::vector<GenerationTurn> turns;
      std::size_t prompt_tokens = 0;
      const auto turn_count = (fields.size() - 2) / 2;
      bool valid = turn_count <= max_prompt_tokens_;
      if (valid) turns.reserve(turn_count);
      for (std::size_t at = 2; valid && at < fields.size(); at += 2) {
        std::size_t limit = 0;
        std::vector<std::int32_t> tokens;
        if (!number(fields[at], limit) || !limit ||
            !parse_tokens(fields[at + 1], tokens, !turns.empty()) ||
            tokens.size() > max_prompt_tokens_ - prompt_tokens) {
          valid = false;
          break;
        }
        prompt_tokens += tokens.size();
        turns.push_back({std::move(tokens), limit});
      }
      if (!valid) {
        if (!emit(error(id, "invalid", "invalid output limit or token list"))) break;
        continue;
      }
      try {
        const auto mode = fields[0] == "G" ? GenerationSessionMode::kFlat :
            fields[0] == "S" ? GenerationSessionMode::kTurns : GenerationSessionMode::kLive;
        auto handle = dispatcher_.submit({std::to_string(id), std::move(turns), mode});
        auto call = std::make_shared<Call>(std::move(handle));
        calls_.emplace(id, call);
        if (!emit("A\t" + std::to_string(id))) { call->cancel(); break; }
        try {
          call->reader = std::thread([this, call, id] { deliver(call, id); });
        } catch (...) {
          call->cancel(); calls_.erase(id); throw;
        }
      } catch (const std::exception &exception) {
        if (!emit(error(id, "admission", exception.what()))) break;
      }
    }
    close();
    return emit("BYE") ? 0 : 1;
  }
private:
  struct Call final {
    explicit Call(GenerationRequestHandle value) : handle(std::move(value)) {}
    GenerationRequestHandle handle;
    std::thread reader;
    std::atomic<bool> done = false;
    std::mutex credit_mutex;
    std::condition_variable credit_ready;
    std::size_t credit = 1, outstanding = 0;
    bool cancelled = false;

    bool cancel() noexcept {
      {
        std::lock_guard lock(credit_mutex);
        if (cancelled || done.load(std::memory_order_acquire)) return false;
        cancelled = true;
      }
      // Cancellation of delivery may still be needed after native completion
      // when accepted tokens remain buffered. Never invent a native outcome.
      (void)handle.cancel();
      credit_ready.notify_all(); return true;
    }
    bool acknowledge(std::size_t count) noexcept {
      std::lock_guard lock(credit_mutex);
      if (count > outstanding) return false;
      outstanding -= count;
      if (!cancelled) credit += count;
      credit_ready.notify_all(); return true;
    }
    bool wait_for_credit() {
      std::unique_lock lock(credit_mutex);
      credit_ready.wait(lock, [&] { return cancelled || credit; });
      if (cancelled) return false;
      --credit; return true;
    }
    bool record_event() noexcept {
      std::lock_guard lock(credit_mutex);
      if (cancelled) return false;
      ++outstanding; return true;
    }
  };
  template<class T> static bool number(std::string_view text, T &value) noexcept {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
  }
  static std::vector<std::string_view> split(std::string_view text, char separator) {
    std::vector<std::string_view> fields;
    std::size_t at = 0;
    for (;;) {
      const auto end = text.find(separator, at);
      fields.push_back(text.substr(at, end == std::string_view::npos ? end : end - at));
      if (end == std::string_view::npos) return fields;
      at = end + 1;
    }
  }
  bool parse_tokens(std::string_view value, std::vector<std::int32_t> &tokens, bool allow_empty) const {
    if (value.empty()) return allow_empty;
    for (const auto piece : split(value, ',')) {
      std::int32_t token = 0;
      if (tokens.size() == max_prompt_tokens_ || !number(piece, token) || token < 0) return false;
      tokens.push_back(token);
    }
    return !tokens.empty();
  }
  static std::string error(std::uint64_t id, std::string_view kind, std::string_view message) {
    std::string frame = "E\t" + std::to_string(id) + "\t" + std::string(kind) + "\t";
    for (const auto ch : message) frame.push_back(ch == '\r' || ch == '\n' || ch == '\t' ? ' ' : ch);
    return frame;
  }
  bool emit(const std::string &frame) {
    std::lock_guard lock(output_mutex_);
    output_ << frame << '\n' << std::flush;
    return bool(output_);
  }
  void deliver(const std::shared_ptr<Call> &call, std::uint64_t id) noexcept {
    try {
      using Kind = GenerationRequestHandle::EventKind;
      for (;;) {
        const bool deliver_token = call->wait_for_credit();
        const auto event = call->handle.read();
        if (event.kind == Kind::kToken) {
          if (deliver_token && call->record_event() &&
              !emit("T\t" + std::to_string(id) + "\t" + std::to_string(event.token))) {
            call->cancel(); break;
          }
          continue;
        }
        if (event.kind == Kind::kTurnEnd) {
          if (deliver_token && call->record_event() &&
              !emit("B\t" + std::to_string(id) + "\t" + std::to_string(event.turn))) {
            call->cancel(); break;
          }
          continue;
        }
        if (event.kind == Kind::kFailed) {
          std::string kind = "model", message = "generation failed";
          try { if (event.failure) std::rethrow_exception(event.failure); }
          catch (const RequestRejected &exception) { kind = "invalid"; message = exception.what(); }
          catch (const std::exception &exception) { message = exception.what(); }
          catch (...) { message = "unknown generation failure"; }
          (void)emit(error(id, kind, message));
        } else {
          const auto status = event.kind == Kind::kCompleted ? "completed" :
              event.kind == Kind::kCancelled ? "cancelled" : "closed";
          (void)emit("F\t" + std::to_string(id) + "\t" + status);
        }
        break;
      }
    } catch (...) {
      call->cancel();
      try { (void)emit(error(id, "internal", "output delivery failed")); } catch (...) {}
    }
    call->done.store(true, std::memory_order_release);
  }
  void reap() {
    for (auto at = calls_.begin(); at != calls_.end();) {
      if (!at->second->done.load(std::memory_order_acquire)) { ++at; continue; }
      if (at->second->reader.joinable()) at->second->reader.join();
      at = calls_.erase(at);
    }
  }
  void close() noexcept {
    for (auto &[id, call] : calls_) { (void)id; (void)call->cancel(); }
    dispatcher_.shutdown();
    for (auto &[id, call] : calls_) {
      (void)id;
      if (call->reader.joinable()) call->reader.join();
    }
    calls_.clear();
  }
  ModelDescription description_;
  GenerationDispatcher dispatcher_;
  const std::size_t max_prompt_tokens_;
  std::istream &input_;
  std::ostream &output_;
  std::mutex output_mutex_;
  std::map<std::uint64_t, std::shared_ptr<Call>> calls_;
};

} // namespace hyle::inference
