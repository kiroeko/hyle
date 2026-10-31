#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "roles.h"
#include "../../inference/backend.h"
#include "../../inference/execution_episode.h"
#include "hyle/compute/state_versions.h"

// One bounded KDA0 -> NoPE MLA3 component, not a complete Kimi model.
namespace hyle::inference::kimi_k3 {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using Bytes=std::vector<std::byte>;
inline constexpr std::int64_t kCapacity=4,kChannels=3*96*128;
struct Epoch final {std::uint64_t step;std::int32_t position,session;};
Bytes read(const c::ReadLease &);
c::ReadLease upload(const c::StorageOwner &,std::span<const std::byte>);

struct Literal final {c::TensorDesc description;Bytes raw;c::ReadLease lease;};
struct LiteralBank final {
  std::vector<std::vector<Literal>> slots;
  c::ReadLease bind(std::size_t,const c::TensorDesc &,Bytes,const c::StorageOwner &);
  void release_host_bytes();
};
struct Port final {g::ValueId value;c::TensorDesc description;};
struct GraphDefinition final {
  g::Graph graph;
  std::vector<Port> inputs,outputs;
  std::vector<g::ConstantBinding> constants;
  c::WorkloadContext workload;
  std::uint64_t region=0;
};
GraphDefinition build_kda_graph(const roles::Kda &,const c::StorageOwner &);
GraphDefinition build_mla_graph(std::int32_t,const roles::Mla &,const c::StorageOwner &,LiteralBank &);

struct Program;
using Episode=hyle::inference::ExecutionEpisode<Program,hyle::inference::SubmissionViews,0>;
struct Pending final {
  std::shared_ptr<Program> owner;
  std::shared_ptr<Episode> execution;
  // Observation is call ordering only; native completion decides retirement.
  bool observed=false;
  Pending()=default;
  Pending(std::shared_ptr<Program>,std::shared_ptr<Episode>);
  Pending(const Pending&)=delete; Pending &operator=(const Pending&)=delete;
  Pending(Pending&&)=default; Pending &operator=(Pending&&)=delete;
  ~Pending();
};
// Model-private stable holder. The Graph/compiled/io precede their borrowers.
// Public members permit finite diagnostics without an alternate runtime API.
struct Program : std::enable_shared_from_this<Program> {
  std::shared_ptr<hyle::inference::Backend> backend;
  g::Graph logical;
  std::vector<c::ReadLease> literals;
  g::CompiledGraph compiled;
  hyle::inference::ProgramIo io;
  std::vector<hyle::inference::ProgramIo::Input> inputs;
  std::vector<hyle::inference::ProgramIo::Output> outputs;
  c::LeasePool pool;
  std::shared_ptr<Episode> pending;
  bool submit_attempted=false;
  hyle::inference::ProgramCarrier<Program> carrier;
  std::uint64_t region=0;
  std::size_t submissions=0,source_submissions_before_host_observation=0;
  std::size_t recorded_capture_ranges=0,reused_capture_ranges=0,host_waits=0;
  std::size_t cold_capture_preparations=0,cold_recorded_ranges=0,cold_reused_ranges=0;
  double cold_capture_seconds=0;
  g::ReplayReport last_replay{};

  Program()=default;
  Program(const Program&)=delete; Program &operator=(const Program&)=delete;
  Program(Program&&)=delete; Program &operator=(Program&&)=delete;
  // Install only on the final stable object, including a test-derived holder.
  void install(GraphDefinition,std::shared_ptr<hyle::inference::Backend>);
  static std::shared_ptr<Program> prepare(GraphDefinition,std::shared_ptr<hyle::inference::Backend>);
  std::size_t input_count() const noexcept {return inputs.size();}
  std::size_t output_count() const noexcept {return outputs.size();}
  const hyle::inference::ProgramIo::InputFacts &input(std::size_t) const noexcept;
  const hyle::inference::PreparedOutputFacts &output(std::size_t) const noexcept;
  hyle::inference::RetirementObservation recover_pending() noexcept;
  Pending prepare_inputs(std::span<const c::ReadLease>,c::WriteLease *hidden_writer=nullptr);
  static const char *route_name(g::ReplayRoute) noexcept;
  g::ReplayReport replay_report(const Episode &,const char *) const;
  c::WriteLease prepare_source_reuse(Pending &,c::ReadLease &);
  void submit(Pending &,Pending *previous=nullptr,bool capture_hidden=false);
  std::vector<c::ReadLease> finish(Pending &);
  std::vector<c::ReadLease> execute(std::span<const c::ReadLease>);
};

enum Entry:std::size_t {kHistory,kRecurrent,kMla,kCursor,kFeature,kEpoch,kEntries};
struct State final {
  c::StateSchema schema;c::StateStore store;std::array<c::StateEntryId,kEntries> entries;
  c::ReadLease member(const c::StateVersion &,Entry) const;
  Epoch epoch(const c::StateVersion &) const;
  static State create(const c::StorageOwner &,std::size_t budget);
  c::StateVersion initial(const c::StorageOwner &,std::int32_t session);
  // Construct one unpublished complete tuple from the real KDA/MLA outputs.
  // Head publication remains with the original caller after its validation.
  c::StateVersion candidate(const c::StateVersion &,std::span<const c::ReadLease> kda,
      std::span<const c::ReadLease> mla,const Epoch &);
};
} // namespace hyle::inference::kimi_k3
