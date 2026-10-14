#pragma once

#include <hyle/compute/detail/copy_outcome_source.h>
#include <hyle/compute/detail/generated_result.h>
#include <hyle/core/assert.h>
#include <utility>

namespace hyle_g03_outcome_probe {
// A borrowed test observation of the cold owner, not a shared_ptr/weak_ptr.
// The caller keeps the cold slot alive through all watch/cancel operations.
struct SourceWatch final {
  const void *state = nullptr;
  bool (*available)(const void *) noexcept = nullptr;
  bool expired() const noexcept { return available(state); }
};
template<class T, class Update> class Outcome final {
public:
  Outcome(const Outcome &) = delete;
  Outcome &operator=(const Outcome &) = delete;
  Outcome(Outcome &&) = default;
  Outcome &operator=(Outcome &&) = default;
  template<class Access>
  static hyle::Result<Outcome> take(hyle::compute::OwnedResult<T> &&result,
      hyle::compute::detail::CopyOutcomeSource<Update, Access> &source,
      hyle::compute::detail::OutcomeSources protocol,
      hyle::compute::detail::DependentCompletion &observer) noexcept {
    HYLE_FATAL_ASSERT(result.valid() && protocol.valid() && observer.available());
    auto &original = hyle::compute::detail::ResultAccess::completion(result);
    auto replaced = hyle::compute::detail::ResultAccess::replace_completion(
        result, observer.publish(std::move(original), protocol));
    HYLE_FATAL_ASSERT(!replaced.valid());
    SourceWatch watch{&source, [](const void *p) noexcept {
      return static_cast<const hyle::compute::detail::CopyOutcomeSource<Update, Access> *>(p)->available();
    }};
    return Outcome(std::move(result), source.update(), watch);
  }
  const hyle::compute::Completion &completion() const noexcept { return result_.completion(); }
  hyle::Result<hyle::compute::ResultLease<T>> result() { return result_.result(); }
  hyle::Result<hyle::compute::ResultLease<T>> wait() {
    const auto terminal = completion().wait();
    if (!terminal.ok()) return std::unexpected(terminal);
    return result_.wait();
  }
  hyle::Status cancel_source() { return update_->request_cancel(); }
  SourceWatch watch() const noexcept { return watch_; }
private:
  Outcome(hyle::compute::OwnedResult<T> result, Update &update, SourceWatch watch) noexcept
      : result_(std::move(result)), update_(&update), watch_(watch) {}
  hyle::compute::OwnedResult<T> result_;
  Update *update_ = nullptr;
  SourceWatch watch_;
};
} // namespace hyle_g03_outcome_probe
