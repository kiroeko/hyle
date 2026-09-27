#include "../graph_route_cpu/d5_common.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace hyle_d5_perf;
namespace v = c::vulkan;

struct Slot final {
  std::vector<v::Tensor> values;
  v::Invocation graph;
  v::RecordedProgram direct;
};

class Route final {
public:
  static constexpr std::uint64_t kBalancedEpochIterations = 6;
  Route(const sem::Fragment &fixture, bool graph)
      : fixture_(fixture), graph_(graph), quota_(mem::system_allocator()),
        context_(take(v::Context::create({.memory_quota = &quota_}))) {
    const auto start = Clock::now();
    if (fixture.preparation || !fixture.state_ports.empty())
      throw std::runtime_error("benchmark scope excludes cold weight recipes and recurrent state");
    const auto &env = context_.environment();
    std::fprintf(stderr, "# D5_VULKAN_ENV device=%s driver=%u api=%u shader=%s\n",
                 env.device_name.c_str(), env.driver_version, env.api_version,
                 env.shader_revision.c_str());
    workload_ = d5_qwen35_fixture::workload(*fixture.profile,
        c::WorkloadPhase::kDecode, fixture.region);
    constants_.resize(fixture.descriptions.size());
    std::vector<v::Tensor> bindings;
    for (std::size_t i = 0; i < fixture.constants.size(); ++i) {
      const auto id = fixture.constants[i];
      constants_[id.ordinal()] = take(context_.upload(fixture.description(id), fixture.constant_bytes[i]));
      bindings.push_back(constants_[id.ordinal()]);
    }
    if (graph_) {
      auto logical = sem::make_graph(fixture);
      g::vulkan::CompileReport report;
      program_ = take(g::vulkan::compile(context_, logical.graph, bindings,
                                        {.report = &report, .workload = workload_}));
      std::fprintf(stderr, "# graph nodes=%zu native_dispatches=%zu managed_regions=%zu constants=%zu\n",
                   report.nodes.size(), report.native_dispatch_count, report.managed_region_count,
                   report.retained_constant_capacity_bytes);
    } else {
      for (const auto &node : fixture.nodes) {
        if (node.operation.kind() == c::OperationKind::kView) continue;
        std::vector<v::Argument> arguments;
        for (auto id : node.inputs) {
          if (id.kind() == d5_qwen38_24t_full_router::ValueKind::kConstant)
            arguments.emplace_back(constants_[id.ordinal()]);
          else arguments.emplace_back(fixture.description(id));
        }
        prepared_.push_back(take(v::prepare(context_, node.operation, arguments, workload_)));
      }
    }
    for (const auto &replay : fixture.replays) make_slot(replay);
    resident_ = quota_.used();
    std::fprintf(stderr, "# setup route=%s slots=%zu resident_bytes=%zu peak_bytes=%zu milliseconds=%.3f\n",
                 graph ? "graph" : "direct", slots_.size(), resident_, quota_.peak(),
                 std::chrono::duration<double, std::milli>(Clock::now()-start).count());
  }
  hyle::Status run() { execute(nullptr); return hyle::Status::Ok(); }
  void verify() {
    next_ = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (std::size_t i = 0; i < slots_.size(); ++i) {
        std::vector<std::byte> bytes(*fixture_.description(fixture_.outputs[0]).required_bytes());
        execute(&bytes);
        verify_output(fixture_, i, bytes);
      }
      // Original ggml allocates native auxiliary buffers on the first use of
      // each slot. Charge and report that warmup, then freeze its live budget.
      if (!warmed_) {
        std::fprintf(stderr, "# warmed route=%s cold_bytes=%zu resident_bytes=%zu\n",
                     graph_ ? "graph" : "direct", resident_, quota_.used());
        resident_ = quota_.used();
        warmed_ = true;
      } else if (quota_.used() != resident_) {
        throw std::runtime_error("Vulkan replay changed warmed retained boundary storage");
      }
    }
    next_ = 0;
  }
private:
  void make_slot(const sem::Replay &replay) {
    auto owned = std::make_unique<Slot>();
    auto &slot = *owned;
    slot.values = constants_;
    std::vector<v::Tensor> inputs;
    for (std::size_t i = 0; i < fixture_.inputs.size(); ++i) {
      const auto id = fixture_.inputs[i];
      slot.values[id.ordinal()] = take(context_.upload(fixture_.description(id), replay.inputs[i]));
      inputs.push_back(slot.values[id.ordinal()]);
    }
    if (graph_) {
      slot.graph = take(program_.create_invocation());
      require(slot.graph.bind(inputs));
    } else {
      auto recorder = take(v::Recorder::create(context_));
      std::size_t prepared = 0;
      for (const auto &node : fixture_.nodes) {
        if (const auto *view = node.operation.get_if<c::View>()) {
          slot.values[node.outputs[0].ordinal()] = take(slot.values[node.inputs[0].ordinal()].view(*view));
          continue;
        }
        std::vector<v::Tensor> arguments, outputs;
        for (auto id : node.inputs)
          if (id.kind() != d5_qwen38_24t_full_router::ValueKind::kConstant)
            arguments.push_back(slot.values[id.ordinal()]);
        for (auto id : node.outputs) {
          slot.values[id.ordinal()] = take(context_.allocate(fixture_.description(id)));
          outputs.push_back(slot.values[id.ordinal()]);
        }
        require(recorder.append(prepared_[prepared++], arguments, outputs));
      }
      slot.direct = take(recorder.finish());
    }
    slots_.push_back(std::move(owned));
  }
  void execute(std::vector<std::byte> *host) {
    auto &slot = *slots_[next_];
    next_ = (next_ + 1) % slots_.size();
    if (graph_) {
      auto work = take(slot.graph.submit());
      auto lease = take(work.wait());
      if (host) require(context_.download(lease->tensors()[0], *host));
    } else {
      auto completion = take(slot.direct.submit());
      require(completion.wait());
      if (host) require(context_.download(slot.values[fixture_.outputs[0].ordinal()], *host));
    }
  }
  const sem::Fragment &fixture_;
  bool graph_;
  mem::QuotaAllocator quota_;
  v::Context context_;
  c::WorkloadContext workload_;
  std::vector<v::Tensor> constants_;
  v::Program program_;
  std::vector<v::Program> prepared_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::size_t next_ = 0, resident_ = 0;
  bool warmed_ = false;
};
}
int main(int argc, char **argv) { return hyle_d5_perf::run_main<Route>(argc, argv); }
