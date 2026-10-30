#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/graph/executable.h"
#include "../../engine/inference/program_io.h"
#include "../../engine/inference/evaluation_failure.h"

// Test-local ordinary Graph lifecycle shared by model consumers. Compilation,
// constants, model math, phase scheduling and accepted State stay with callers.
namespace model_graph_run {
namespace c=hyle::compute;
namespace g=hyle::graph;
struct Port final {
  g::ValueId id;
  c::TensorDesc description;
};
namespace detail {
template<class T> T take(hyle::Result<T> result,const char *where) {
  if(!result)throw hyle::inference::EvaluationFailure(result.error().WithContext(where));
  return std::move(*result);
}
inline void check(hyle::Status result,const char *where) {
  if(!result.ok())throw hyle::inference::EvaluationFailure(result.WithContext(where));
}
inline hyle::inference::RetirementObservation observe_failure(g::Invocation &active,bool awaiting_completion) noexcept {
  // Drain closes the original slot. Only accepted/unobserved work uses it;
  // a recoverable pre-submit refusal still withdraws through the original reset.
  // Failed or retained-source reset refusals keep their original permissions.
  const auto status=active.in_flight()||awaiting_completion?active.drain():active.reset();
  return {!active.in_flight(),status};
}
} // namespace detail
class ColdProgram;

// Cold port policy only. It borrows one exact stable program, never native
// permissions, a binder, a Completion or per-execution binding state.
class InputPlan final {
 public:
  bool belongs_to(const ColdProgram &program) const noexcept { return program_==&program; }
  bool needs_external() const noexcept { return external_; }
  std::span<const std::size_t> ordinary() const noexcept { return ordinary_; }
 private:
  friend class ColdProgram;
  const ColdProgram *program_=nullptr;
  std::vector<std::size_t> ordinary_;
  bool external_=false;
};

// Replaces the original consumer's CompiledGraph holder. This unique cold
// object keeps that same graph and IO stable while Native/Run moves. The
// original Invocation, result pool and native retirement stay with callers.
class ColdProgram final {
 public:
  g::CompiledGraph compiled;
  hyle::inference::ProgramIo io;
  ColdProgram(const ColdProgram &)=delete;
  ColdProgram &operator=(const ColdProgram &)=delete;
  ColdProgram(ColdProgram &&)=delete;
  ColdProgram &operator=(ColdProgram &&)=delete;

  static std::unique_ptr<ColdProgram> create(g::CompiledGraph compiled,
      std::span<const Port> inputs,std::span<const Port> outputs) {
    return std::unique_ptr<ColdProgram>(new ColdProgram(std::move(compiled),inputs,outputs));
  }
  std::size_t input_count() const noexcept { return inputs_.size(); }
  std::size_t output_count() const noexcept { return outputs_.size(); }
  const hyle::inference::ProgramIo::InputFacts &input(std::size_t index) const noexcept {
    HYLE_FATAL_ASSERT(index<inputs_.size());
    return io.input_facts(inputs_[index]);
  }
  const hyle::inference::PreparedOutputFacts &output(std::size_t index) const noexcept {
    HYLE_FATAL_ASSERT(index<outputs_.size());
    return io.output_facts(outputs_[index]);
  }
  InputPlan input_plan(std::span<const std::size_t> external) const {
    std::vector<std::uint8_t> excluded(input_count(),0);
    for(const auto index:external) {
      if(index>=excluded.size()||excluded[index])
        throw std::runtime_error("ordinary model Graph resident input index");
      excluded[index]=1;
    }
    InputPlan result;result.program_=this;result.external_=!external.empty();
    result.ordinary_.reserve(input_count()-external.size());
    for(std::size_t index=0;index<excluded.size();++index)
      if(!excluded[index])result.ordinary_.push_back(index);
    return result;
  }
  const InputPlan &all_inputs() const noexcept { return all_; }
  hyle::Status bind(g::Invocation &run,std::size_t index,const c::ReadLease &lease) const noexcept {
    HYLE_FATAL_ASSERT(index<inputs_.size());
    return io.bind(run,inputs_[index],lease);
  }
  hyle::Status bind_output(g::Invocation &run,std::size_t index,c::LeasePool &pool) const noexcept {
    HYLE_FATAL_ASSERT(index<outputs_.size());
    return io.bind_output(run,outputs_[index],pool);
  }
  hyle::Result<c::ReadLease> retain(g::Invocation &run,std::size_t index) const noexcept {
    HYLE_FATAL_ASSERT(index<outputs_.size());
    return io.retain(run,outputs_[index]);
  }
 private:
  std::vector<hyle::inference::ProgramIo::Input> inputs_;
  std::vector<hyle::inference::ProgramIo::Output> outputs_;
  InputPlan all_;
  ColdProgram(g::CompiledGraph selected,std::span<const Port> inputs,std::span<const Port> outputs)
      : compiled(std::move(selected)) {
    detail::check(io.prepare(compiled,0,inputs.size(),0,outputs.size()),"prepare ordinary model Graph IO");
    inputs_.reserve(inputs.size());outputs_.reserve(outputs.size());
    for(const auto &port:inputs) {
      auto handle=detail::take(io.add_input(port.id),"register ordinary model Graph input");
      if(io.input_facts(handle).description!=port.description)
        throw std::runtime_error("ordinary model Graph concrete input description changed");
      inputs_.push_back(handle);
    }
    for(const auto &port:outputs) {
      auto handle=detail::take(io.add_output(port.id),"prepare ordinary model Graph output");
      const auto &facts=io.output_facts(handle);
      if(!facts.caller_owned())throw std::runtime_error("ordinary model Graph requires caller-owned result storage");
      if(facts.description!=port.description)
        throw std::runtime_error("ordinary model Graph concrete output description changed");
      outputs_.push_back(handle);
    }
    detail::check(io.seal(),"seal ordinary model Graph IO");
    all_=input_plan({});
  }
};

// Named views borrow the original live table and immutable port ordering only.
// Requiring references rejects projections which would copy leases per access.
template<class Inputs>
concept LeaseInputRange=std::ranges::sized_range<const Inputs> &&
    std::ranges::random_access_range<const Inputs> &&
    (std::same_as<std::ranges::range_reference_t<const Inputs>,const c::ReadLease&> ||
     std::same_as<std::ranges::range_reference_t<const Inputs>,c::ReadLease&>);

// Preserve the original eager origin check before any Graph binding. The view
// itself holds spans only; live/completed and their release points stay with the
// caller. Neither original port metadata nor lease owners may change during
// the synchronous execute call. Rvalue tables/port containers cannot bind here.
template<std::ranges::contiguous_range Values>
requires std::ranges::sized_range<Values>
auto project_inputs(Values &values,std::vector<c::ReadLease> &live) {
  using Value=std::ranges::range_value_t<Values>;
  const std::span<const Value> originals{std::ranges::data(values),std::ranges::size(values)};
  for(const auto value:originals)(void)live.at(value.ordinal());
  const std::span<const c::ReadLease> leases{live};
  return originals|std::views::transform([leases](const Value &value)->const c::ReadLease& {
    return leases[value.ordinal()];
  });
}

template<std::integral Counter,LeaseInputRange Inputs=std::span<const c::ReadLease>>
std::vector<c::ReadLease> execute(ColdProgram &program,g::Invocation &invocation,
    c::LeasePool &outputs,c::Queue &queue,const Inputs &inputs,
    Counter &completed_submissions,const InputPlan *plan=nullptr,
    const std::function<void(g::Invocation&)> &bind_external={},
    hyle::inference::RetirementObservation *failed_retirement=nullptr) {
  if(std::ranges::size(inputs)!=program.input_count())throw std::runtime_error("ordinary model Graph input count");
  const auto &selected=plan?*plan:program.all_inputs();
  if(!selected.belongs_to(program))throw std::runtime_error("ordinary model Graph input plan belongs to another program");
  if(selected.needs_external()&&!bind_external)throw std::runtime_error("ordinary model Graph resident binder missing");
  bool awaiting_completion=false;
  try {
    for(const auto index:selected.ordinary())
      detail::check(program.bind(invocation,index,std::ranges::begin(inputs)[index]),"bind ordinary model Graph input");
    if(bind_external)bind_external(invocation);
    for(std::size_t index=0;index<program.output_count();++index)
      detail::check(program.bind_output(invocation,index,outputs),"bind ordinary model Graph output writer");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
    detail::check(invocation.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}),
        "prepare ordinary model Graph");
    {
      auto done=detail::take(invocation.submit(queues),"ordinary model Graph submit");
      awaiting_completion=true;
      detail::check(done.wait(),"ordinary model Graph retirement");
      awaiting_completion=false;
    }
    ++completed_submissions;
    std::vector<c::ReadLease> result;
    result.reserve(program.output_count());
    for(std::size_t index=0;index<program.output_count();++index)
      result.push_back(detail::take(program.retain(invocation,index),"retain ordinary model Graph result lease"));
    // This synchronous runner already withdraws through its original reset.
    // Do not add a preceding unbind scan or require native binding-release.
    detail::check(invocation.reset(),"withdraw ordinary model Graph permissions");
    if(invocation.submit(queues))throw std::runtime_error("reset ordinary model Graph retained stale replay");
    return result;
  } catch(hyle::inference::EvaluationFailure &failure) {
    const auto observed=detail::observe_failure(invocation,awaiting_completion);
    if(failed_retirement)*failed_retirement=observed;
    failure.observe_retirement(observed);
    throw;
  } catch(...) {
    const auto observed=detail::observe_failure(invocation,awaiting_completion);
    if(failed_retirement)*failed_retirement=observed;
    throw;
  }
}
} // namespace model_graph_run
