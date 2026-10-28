#pragma once

#include <cstddef>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "hyle/container/small_vector.h"
#include "batch_dispatch.h"
#include "request_window.h"

namespace hyle::inference {

// Cold-reserved host scratch, borrowed for one complete request wave. The
// original calls and their input leases travel together; this adds no model
// head, Completion, credit or native retirement authority. Every exit clears
// the temporary call records.
template<class Call, std::size_t InlineCapacity = 2>
class RequestBatchWorkspace final {
  static_assert(std::is_nothrow_move_constructible_v<Call> &&
                std::is_nothrow_move_assignable_v<Call> &&
                std::is_nothrow_destructible_v<Call>,
                "selected calls must return their actual result without throwing");
public:
  void prepare(std::size_t candidates, std::size_t batch_limit) {
    if (borrowed_ || !calls_.empty() || !bucket_.empty())
      throw std::logic_error("request batch scratch cannot change during its original wave");
    if (!candidates || !batch_limit || batch_limit > candidates)
      throw std::invalid_argument("request batch scratch requires positive candidate/batch bounds");
    if (!calls_.try_reserve(candidates) || !bucket_.try_reserve(batch_limit)) throw std::bad_alloc{};
    indices_.prepare(candidates, batch_limit);
    max_candidates_ = candidates;
  }

private:
  template<class C, std::size_t N, class Policy>
  friend void execute_request_batches(RequestBatchWorkspace<C, N> &,
      std::span<const RequestKey>, Policy &);
  container::SmallVector<Call, InlineCapacity> calls_, bucket_;
  BatchIndices indices_;
  std::size_t max_candidates_ = 0;
  bool borrowed_ = false;
};

// Policy uses direct, statically resolved stages:
// prepare(key) -> Call; limit(first); compatible(first, other); fit(batch);
// execute(batch, entire_wave); publish(call).
//
// All calls are prepared before any batch or prompt publication. Compatibility
// and admission remain real model facts. fit() may only admit the selected
// prefix; an unadmitted tail is returned immediately to its original call.
// execute owns submission/observation/retirement and maps errors using the
// actual submitted members. A pending episode must retain its own call facts,
// never a borrowed span into this scratch. Publication follows the whole wave;
// a later exception does not undo already committed model work.
template<class Call, std::size_t InlineCapacity, class Policy>
void execute_request_batches(RequestBatchWorkspace<Call, InlineCapacity> &scratch,
    std::span<const RequestKey> keys, Policy &policy) {
  if (scratch.borrowed_ || !scratch.calls_.empty() || !scratch.bucket_.empty())
    throw std::logic_error("request batch scratch is already borrowed");
  if (!scratch.max_candidates_ || keys.size() > scratch.max_candidates_)
    throw std::out_of_range("request wave exceeds its cold-prepared batch scratch");
  scratch.borrowed_ = true;
  struct Release final {
    container::SmallVector<Call, InlineCapacity> &calls, &bucket;
    bool &borrowed;
    ~Release() noexcept {
      bucket.clear(); calls.clear(); borrowed = false;
    }
  } release{scratch.calls_, scratch.bucket_, scratch.borrowed_};
  auto &calls = scratch.calls_; auto &bucket = scratch.bucket_;
  for (const auto key : keys) calls.push_back(policy.prepare(key));
  dispatch_batches(scratch.indices_, calls.size(), [](std::size_t) { return true; },
      [&](std::size_t first) -> std::size_t { return policy.limit(calls[first]); },
      [&](std::size_t first, std::size_t other) { return policy.compatible(calls[first], calls[other]); },
      [&](std::span<const std::size_t> selected) -> std::size_t {
        // The selection is borrowed from the original BatchIndices until this
        // callback returns. Every actual result, including a cancellation or a
        // partially committed error, returns before the next selection occurs.
        struct ReturnCalls final {
          container::SmallVector<Call, InlineCapacity> &calls, &bucket;
          std::span<const std::size_t> selected;
          void last() noexcept {
            const auto index = bucket.size() - 1;
            calls[selected[index]] = std::move(bucket.back()); bucket.pop_back();
          }
          ~ReturnCalls() noexcept { while (!bucket.empty()) last(); }
        } returned{calls, bucket, selected};
        for (const auto index : selected) bucket.push_back(std::move(calls[index]));
        while (bucket.size() > 1 && !policy.fit(std::span<Call>(bucket.data(), bucket.size()))) returned.last();
        const auto completed = bucket.size();
        policy.execute(std::span<Call>(bucket.data(), bucket.size()), completed == keys.size());
        return completed;
      });
  for (auto &call : calls) policy.publish(call);
}

} // namespace hyle::inference
