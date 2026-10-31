#include "component.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "../../inference/output_pool_capacity.h"
#include "hyle/graph/cuda.h"

namespace hyle::inference::kimi_k3 {
namespace inference=hyle::inference;
using roles::detail::take;
namespace {
void check(hyle::Status status,const char *where) {
  if(!status.ok())throw EvaluationFailure(status.WithContext(where));
}
c::TensorDesc dense(std::initializer_list<std::int64_t> shape,c::DataType type=c::DataType::kFloat32) {
  return take(c::TensorDesc::dense(type,{shape.begin(),shape.size()}),"Kimi state description");
}
template<class T> Bytes bytes(const std::vector<T> &values) {
  const auto raw=std::as_bytes(std::span(values));return {raw.begin(),raw.end()};
}
} // namespace
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());check(take(lease.view(),"Kimi readonly view").read(result),"Kimi readonly bytes");return result;
}
c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> raw) {
  auto buffer=take(m::Buffer::create(owner.ref(),raw.size(),256),"Kimi immutable backing");
  check(buffer.view().write(raw),"Kimi one-time upload");return take(owner.adopt_buffer(std::move(buffer)),"Kimi original readonly permission");
}
c::ReadLease LiteralBank::bind(std::size_t slot,const c::TensorDesc &description,Bytes raw,const c::StorageOwner &owner) {
    if(slots.size()<=slot)slots.resize(slot+1);
    for(const auto &literal:slots[slot])if(literal.description==description&&literal.raw==raw)return literal.lease;
    auto lease=upload(owner,raw);slots[slot].push_back({description,std::move(raw),lease});return lease;
  }
void LiteralBank::release_host_bytes() {for(auto &slot:slots)for(auto &literal:slot)Bytes{}.swap(literal.raw);}
Pending::Pending(std::shared_ptr<Program> program,std::shared_ptr<Episode> episode)
    :owner(std::move(program)),execution(std::move(episode)) {}
Pending::~Pending() {if(owner&&execution&&owner->pending==execution)(void)owner->recover_pending();}

std::shared_ptr<Program> Program::prepare(GraphDefinition definition,std::shared_ptr<Backend> selected_backend) {
  auto result=std::make_shared<Program>();result->install(std::move(definition),std::move(selected_backend));return result;
}
void Program::install(GraphDefinition definition,std::shared_ptr<Backend> selected_backend) {
  if(backend||logical||compiled.executable)throw std::runtime_error("Kimi Program was already installed");
  backend=std::move(selected_backend);
  if(!backend)throw std::runtime_error("Kimi Backend is absent");
  auto &context=*take(backend->common_context(),"Kimi original Backend Context");
  const auto &owner=backend->storage_owner();
  logical=std::move(definition.graph);region=definition.region;
  literals.reserve(definition.constants.size());
  for(const auto &constant:definition.constants)literals.push_back(constant.lease);
  compiled=take(g::cuda::compile(logical,context,owner,{.constants=definition.constants,.workload=definition.workload}),"Kimi ordinary Graph compile");
  check(io.prepare(compiled,0,definition.inputs.size(),0,definition.outputs.size()),"prepare ordinary model Graph IO");
  inputs.reserve(definition.inputs.size());outputs.reserve(definition.outputs.size());
  for(const auto &port:definition.inputs) {
    auto handle=take(io.add_input(port.value),"register ordinary model Graph input");
    if(io.input_facts(handle).description!=port.description)throw std::runtime_error("Kimi concrete input description changed");
    inputs.push_back(handle);
  }
  for(const auto &port:definition.outputs) {
    auto handle=take(io.add_output(port.value),"prepare ordinary model Graph output");
    const auto &facts=io.output_facts(handle);
    if(!facts.caller_owned()||facts.description!=port.description)throw std::runtime_error("Kimi caller output contract changed");
    outputs.push_back(handle);
  }
  check(io.seal(),"seal ordinary model Graph IO");
  std::vector<std::size_t> output_sizes;output_sizes.reserve(output_count());
  for(std::size_t index=0;index<output_count();++index)output_sizes.push_back(take(output(index).description.required_bytes(),"Kimi output extent"));
  constexpr std::uint32_t generations=8;
  if(output_sizes.empty())throw std::runtime_error("caller output inventory must be nonempty");
  const auto output_capacity=take(inference::checked_output_pool_capacity(output_sizes,generations,256),"Kimi output pool capacity");
  pool=take(c::LeasePool::create(owner.ref(),output_capacity.backing_bytes,{.max_blocks=output_capacity.max_blocks,.backing_alignment=256}),"bounded Kimi output generations");
  check(carrier.prepare(compiled,backend->queue_bindings()),"Kimi original carrier preparation");
  check(carrier.initialize({.io=io,.invocation={.resources=g::ResourceOptions{}}}),"Kimi original Invocation installation");
}
const inference::ProgramIo::InputFacts &Program::input(std::size_t index) const noexcept {
  HYLE_FATAL_ASSERT(index<inputs.size());return io.input_facts(inputs[index]);
}
const inference::PreparedOutputFacts &Program::output(std::size_t index) const noexcept {
  HYLE_FATAL_ASSERT(index<outputs.size());return io.output_facts(outputs[index]);
}
inference::RetirementObservation Program::recover_pending() noexcept {
    if(!pending)return {true,{}};
    auto episode=pending;
    // A plain bind refusal has not touched Source/export or submission. Use
    // the original reset only with the carrier's actual retirement proof.
    if(!submit_attempted&&!episode->incoming&&!episode->sources&&episode->run&&carrier.retired()) {
      const auto reset=episode->run->reset();
      if(reset.ok()) {
        episode->run=nullptr;episode->carrier=nullptr;pending.reset();return {true,{}};
      }
    }
    episode->cleanup_pending=true;
    auto observed=episode->retire(*backend);
    auto recovered=episode->recover([]() noexcept {});
    // An unsubmitted Source consumer may retain the original predecessor
    // retirement gate. Drain only that original episode, leaving its exports
    // intact; then recover this consumer before producer cleanup can proceed.
    if(recovered&&!*recovered&&episode->predecessor_episode) {
      const auto predecessor=episode->predecessor_episode->retire(*backend);
      inference::record_retirement_status(observed.status,predecessor.status);
      recovered=episode->recover([]() noexcept {});
    }
    if(!recovered)inference::record_retirement_status(observed.status,recovered.error());
    observed.retired=carrier.retired()&&observed.retired&&recovered&&*recovered;
    if(observed.retired)pending.reset(); // A partial return retains this exact original episode.
    return observed;
  }

Pending Program::prepare_inputs(std::span<const c::ReadLease> bindings,c::WriteLease *hidden_writer) {
    if(bindings.size()!=input_count())throw std::runtime_error("Kimi original input count");
    if(pending) {
      const auto observed=recover_pending();
      if(!observed.retired)throw inference::EvaluationFailure(hyle::Status::Unavailable("Kimi original episode is still retiring"));
      check(observed.status,"Kimi previous original retirement");
    }
    if(!carrier.initialized())
      check(carrier.initialize({.io=io,.invocation={.resources=g::ResourceOptions{}}}),"Kimi original Invocation retry");
    auto episode=std::make_shared<Episode>();pending=episode;submit_attempted=false;
    try {
      auto &run=*take(carrier.borrow(),"Kimi original carrier admission");
      episode->run=&run;episode->carrier=&carrier;
      // All ordinary binding and capture preparation precede producer submit.
      // A real input refusal retains the original Invocation for a legal retry.
      for(std::size_t i=0;i<bindings.size();++i)
        check(io.bind(run,inputs[i],bindings[i]),"Kimi ordinary dynamic input");
      for(std::size_t i=0;i<output_count();++i) {
        if(i==5&&hidden_writer) {
          const auto &selected_output=this->output(i);
          check(run.bind(selected_output.value,selected_output.description,std::move(*hidden_writer)),"Kimi original reclaimed hidden writer");
        } else check(io.bind_output(run,outputs[i],pool),"Kimi original output writer");
      }
      return {shared_from_this(),std::move(episode)};
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

const char *Program::route_name(g::ReplayRoute route) noexcept {
    switch(route) {
      case g::ReplayRoute::kCommands:return "commands";
      case g::ReplayRoute::kCaptured:return "captured";
      case g::ReplayRoute::kHybrid:return "hybrid";
      default:return "unprepared";
    }
  }

g::ReplayReport Program::replay_report(const Episode &episode,const char *) const {
    const auto report=take(episode.run->replay_report(),"Kimi actual replay report");
    return report;
  }

c::WriteLease Program::prepare_source_reuse(Pending &prepared,c::ReadLease &hidden_ready) {
    if(prepared.owner.get()!=this||prepared.execution!=pending||submit_attempted)
      throw std::runtime_error("Kimi cold capture requires its original unsubmitted episode");
    auto &episode=*prepared.execution;
    const auto start=std::chrono::steady_clock::now();
    try {
      const auto original=take(hidden_ready.tensor(this->input(0).description),"Kimi cold hidden tensor");
      const auto generation=hidden_ready.generation();
      check(episode.prepare_replay({.mode=g::ReplayMode::kPreferCapture,.host_wait=g::HostWaitPolicy::kReject}),
          "Kimi complete MLA cold capture");
      const auto report=replay_report(episode,"consumer-cold-prepare");
      if((report.route!=g::ReplayRoute::kCaptured&&report.route!=g::ReplayRoute::kHybrid)||
          report.host_wait_dependency_count||(!report.recorded_capture_range_count&&!report.reused_capture_range_count))
        throw std::runtime_error("Kimi MLA cold preparation did not establish actual capture coverage");
      ++cold_capture_preparations;cold_recorded_ranges+=report.recorded_capture_range_count;
      cold_reused_ranges+=report.reused_capture_range_count;
      // Preserve every other original input, output writer and captured private
      // address. Only this complete ready input hands its one block to KDA.
      check(episode.run->unbind(this->input(0).value),"Kimi cold hidden input withdrawal");
      auto writer=take(hidden_ready.try_write(),"Kimi original hidden writer handoff");
      const auto reclaimed=take(writer.view(),"Kimi reclaimed hidden backing");
      const auto storage=original.storage();
      if(writer.generation()!=generation||reclaimed.resource()!=storage.resource()||
          reclaimed.native_handle()!=storage.native_handle()||reclaimed.offset()!=storage.offset()||
          reclaimed.size()!=storage.size())
        throw std::runtime_error("Kimi cold hidden handoff changed its original physical writer");
      cold_capture_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      return writer;
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

void Program::submit(Pending &prepared,Pending *previous,bool capture_hidden) {
    if(prepared.owner.get()!=this||prepared.execution!=pending||submit_attempted)
      throw std::runtime_error("Kimi submit requires its original prepared episode");
    auto &episode=*prepared.execution;auto &run=*episode.run;
    try {
      if(previous) {
        if(!previous->execution||!previous->execution->sources||previous->observed)
          throw std::runtime_error("Kimi MLA requires its unobserved original KDA submission");
        episode.incoming=std::make_unique<inference::IncomingExecutionSources<inference::SubmissionViews>>();
        episode.incoming->owner=previous->execution->sources;
        episode.incoming->bindings.push_back(episode.incoming->owner->primary_output.reader_binding(this->input(0).value));
        episode.predecessor_episode=previous->execution;
      }
      if(capture_hidden) {
        episode.sources=std::make_shared<inference::SubmissionViews>();
        check(episode.sources->prepare(run,std::span<const inference::SubmissionViews::StoreCapture>{},std::span<const inference::SubmissionViews::PageCapture>{},{}),"Kimi original hidden Source preparation");
      }
      check(episode.bind_consumers(logical),"Kimi original consumer bundle");
      check(episode.prepare_replay({.mode=capture_hidden?g::ReplayMode::kPreferCapture:g::ReplayMode::kCommands,
          .host_wait=(previous||capture_hidden)?g::HostWaitPolicy::kReject:g::HostWaitPolicy::kAllow}),"Kimi original replay preparation");
      last_replay=replay_report(episode,previous?"consumer-source-reuse":capture_hidden?"producer":"standalone");
      if(previous) {
        const auto source=take(run.input_source_report(),"Kimi actual dependency report");
        if(last_replay.recorded_capture_range_count||last_replay.host_wait_dependency_count||
            !last_replay.reused_capture_range_count||source.source_count!=1||source.device_wait_count!=1||source.host_wait_count)
          throw std::runtime_error("Kimi Source submit did not reuse complete device-ordered cold capture");
      }
      recorded_capture_ranges+=last_replay.recorded_capture_range_count;
      reused_capture_ranges+=last_replay.reused_capture_range_count;host_waits+=last_replay.host_wait_dependency_count;
      submit_attempted=true;
      check(episode.submit(shared_from_this()),"Kimi original Graph submit");
      if(previous)++source_submissions_before_host_observation;
      if(capture_hidden)
        take(episode.sources->capture(run,episode.completion,this->output(5).value,
            backend->storage_owner(),pool),"Kimi actual hidden OutputSource");
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

std::vector<c::ReadLease> Program::finish(Pending &submitted) {
    if(submitted.owner.get()!=this||submitted.execution!=pending||!submit_attempted)
      throw std::runtime_error("Kimi finish requires its original pending episode");
    auto &episode=*submitted.execution;
    submitted.observed=true;
    try {
      std::vector<c::ReadLease> result;
      check(episode.finish_success(0,0,{},[&]() noexcept {++submissions;},[&]() -> hyle::Status {
        result.reserve(output_count());
        for(std::size_t i=0;i<output_count();++i)
          result.push_back(take(io.retain(*episode.run,outputs[i]),"Kimi retained original result"));
        return hyle::Status::Ok();
      },[]() noexcept -> hyle::Status {return hyle::Status::Ok();}),"Kimi original successful Graph stage");
      episode.sources.reset();
      // Successful serial reuse keeps the same Invocation. Recovery alone
      // uses reset_retired(), after the original transactions have returned.
      check(episode.run->reset(),"Kimi original Invocation reset");
      episode.run=nullptr;episode.carrier=nullptr;pending.reset();
      submitted.execution.reset();submitted.owner.reset();return result;
    } catch(inference::EvaluationFailure &failure) {
      failure.observe_retirement(recover_pending());throw;
    } catch(...) { (void)recover_pending();throw; }
  }

std::vector<c::ReadLease> Program::execute(std::span<const c::ReadLease> input) {
    auto prepared=prepare_inputs(input);submit(prepared);return finish(prepared);
  }

c::ReadLease State::member(const c::StateVersion &v,Entry entry) const {return take(v.member(entries[entry]),"Kimi complete member");}

Epoch State::epoch(const c::StateVersion &version) const {
    const auto raw=read(member(version,kEpoch));
    if(raw.size()!=sizeof(Epoch))throw std::runtime_error("Kimi Epoch metadata extent");
    Epoch result{};std::memcpy(&result,raw.data(),sizeof(result));return result;
  }

State State::create(const c::StorageOwner &owner,std::size_t budget) {
    const std::array descriptions{dense({3,kChannels}),dense({1,96,128,128}),dense({kCapacity,576}),
        dense({1},c::DataType::kInt32),dense({1,7168})};
    std::vector<c::StateEntrySpec> specs;
    for(const auto &description:descriptions)specs.push_back({.description=description,.resource=owner.ref()});
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Epoch)});
    State result;result.schema=take(c::StateSchema::create(specs),"Kimi heterogeneous schema");
    result.store=take(c::StateStore::create(result.schema,{.max_versions=12,.max_drafts=3,.max_retained_blocks=128,
        .max_backing_bytes=budget,.max_host_snapshot_bytes=4096}),"one bounded heterogeneous StateStore");
    for(std::size_t i=0;i<kEntries;++i)result.entries[i]=take(result.schema.entry(i),"Kimi state entry");return result;
  }

c::StateVersion State::initial(const c::StorageOwner &owner,std::int32_t session) {
    std::vector<c::StateMember> members;
    for(std::size_t i=0;i<kEpoch;++i) {
      const auto spec=take(schema.specification(entries[i]),"Kimi initial descriptor");
      Bytes raw(take(spec.description.required_bytes(),"Kimi initial bytes"));
      if(i==kMla)raw=bytes(std::vector<float>(std::size_t(kCapacity*576),std::bit_cast<float>(std::uint32_t{0x7fc12345})));
      members.push_back({entries[i],upload(owner,raw)});
    }
    const Epoch epoch{0,0,session};members.push_back({entries[kEpoch],{},std::as_bytes(std::span(&epoch,1))});
    return take(store.import(members),"Kimi empty complete state");
  }
c::StateVersion State::candidate(const c::StateVersion &before,std::span<const c::ReadLease> kda,
    std::span<const c::ReadLease> mla,const Epoch &epoch_value) {
  if(kda.size()!=6||mla.size()!=10)throw std::runtime_error("Kimi complete candidate output inventory");
  auto draft=take(store.begin(before),"original-parameter complete draft");
  check(draft.replace(entries[kHistory],kda[0]),"original convolution history");
  check(draft.replace(entries[kRecurrent],kda[2]),"original recurrent state");
  check(draft.replace(entries[kMla],mla[0]),"original complete MLA cache");
  check(draft.replace(entries[kCursor],mla[9]),"original actual cursor");
  check(draft.replace(entries[kFeature],mla[6]),"original final feature");
  check(draft.replace_metadata(entries[kEpoch],std::as_bytes(std::span(&epoch_value,1))),"original complete Epoch");
  return take(draft.finalize(),"original complete candidate");
}
} // namespace hyle::inference::kimi_k3
