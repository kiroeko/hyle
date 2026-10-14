#pragma once

#include <hyle/compute/detail/completion_access.h>
#include <hyle/compute/result.h>
#include <atomic>
#include <memory>

// SCOPED audit fixture. It keeps the original result slot/completion and only
// adds an opt-in source-retirement owner. No executor, queue, or result copy.
namespace hyle_g03_outcome_probe {
template<class Update> struct Sources final {
  explicit Sources(Update value):update(std::move(value)){}
  Update update;
  std::atomic<bool> suffix_retired{false};
  hyle::compute::detail::CompletionRetention retention;
  std::shared_ptr<Sources> accepted;
};
template<class T,class Update> class Outcome final {
  using State=Sources<Update>;
public:
  Outcome(hyle::compute::OwnedResult<T> result,std::shared_ptr<State> state)
      :state_(std::move(state)),result_(std::move(result)){}
  Outcome(const Outcome&)=delete;
  Outcome&operator=(const Outcome&)=delete;
  Outcome(Outcome&&)=default;
  Outcome&operator=(Outcome&&)=default;
  ~Outcome()=default;

  static hyle::Result<Outcome> take(hyle::compute::OwnedResult<T>&& result,Update&& update) {
    if(!result.valid() || !update.valid())
      return std::unexpected(hyle::Status::InvalidArgument("empty outcome/source"));
    auto state=std::make_shared<State>(std::move(update));
    state->accepted=state;
    state->retention.own={state.get(),[](void*pointer,const hyle::Status&)noexcept{
      static_cast<State*>(pointer)->suffix_retired.store(true,std::memory_order_release);
    },[](void*pointer)noexcept{
      auto keep=std::move(static_cast<State*>(pointer)->accepted);
    }};
    // Fixture only. The underlying object is non-const; a product adapter
    // requires an audited trusted Result access seam, never this cast.
    auto& completion=const_cast<hyle::compute::Completion&>(result.completion());
    const auto attached=hyle::compute::detail::CompletionAccess::attach_retention(
        completion,state->retention);
    if(!attached.ok()){state->accepted={};return std::unexpected(attached);}
    return Outcome(std::move(result),std::move(state));
  }
  hyle::Result<hyle::compute::ResultLease<T>> result() {
    const auto polled=result_.poll();
    if(!state_->suffix_retired.load(std::memory_order_acquire))
      return std::unexpected(hyle::Status::Unavailable("suffix actual retirement unproved"));
    const auto source=state_->update.observe();
    if(state_->update.in_flight())
      return std::unexpected(hyle::Status::Unavailable("source actual retirement unproved"));
    if(!source.ok())return std::unexpected(source);
    if(!polled)return std::unexpected(polled.error());
    return result_.result();
  }
  hyle::Result<hyle::compute::ResultLease<T>> wait() {
    const auto native=result_.completion().wait();
    if(!state_->suffix_retired.load(std::memory_order_acquire))
      return std::unexpected(hyle::Status::Unavailable("native error did not prove retirement"));
    const auto source=state_->update.wait();
    if(!source.ok())return std::unexpected(source);
    if(!native.ok())return std::unexpected(native);
    return result_.result();
  }
  hyle::Status cancel_source(){return state_->update.request_cancel();}
  std::weak_ptr<State> watch()const{return state_;}
private:
  // Destroying the original completion before the last public source owner
  // falls is enforced by member destruction order. Its retained attachment
  // keeps sources alive through actual retirement on abandonment.
  std::shared_ptr<State> state_;
  hyle::compute::OwnedResult<T> result_;
};
} // namespace hyle_g03_outcome_probe
