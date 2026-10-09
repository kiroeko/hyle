#include "../graph_route_cpu/d5_common.h"
#include "hyle/graph/vulkan.h"
#include "hyle/compute/detail/vulkan_plan.h"

namespace {
using namespace hyle_d5_perf;
namespace v = c::vulkan;

struct Slot final {
  std::vector<v::Tensor> values;
  v::Invocation graph;
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
      // Explicit native fragment frontend, with the same owned Program,
      // Invocation, publication and native completion contract as Graph.
      v::detail::Plan plan;
      plan.workload = workload_;
      for (std::size_t i = 0; i < fixture.descriptions.size(); ++i)
        plan.values.push_back({fixture.descriptions[i], constants_[i]});
      for (auto id : fixture.inputs) plan.inputs.push_back(id.ordinal());
      for (auto id : fixture.outputs) plan.outputs.push_back(id.ordinal());
      for (const auto &node : fixture.nodes) {
        v::detail::PlanStep step;
        step.operation = node.operation;
        for (auto id : node.inputs) step.inputs.push_back(id.ordinal());
        for (auto id : node.outputs) step.outputs.push_back(id.ordinal());
        plan.steps.push_back(std::move(step));
      }
      v::detail::PlanFacts report;
      program_ = take(v::detail::prepare_plan(context_, plan, &report));
      std::fprintf(stderr, "# explicit-native-plan native_dispatches=%zu managed_regions=%zu constants=%zu\n",
                   report.native_dispatch_count, report.managed_region_count,
                   report.retained_constant_capacity_bytes);
    }
    for (const auto &replay : fixture.replays) make_slot(replay);
    resident_ = quota_.used();
    std::fprintf(stderr, "# setup route=%s slots=%zu resident_bytes=%zu peak_bytes=%zu milliseconds=%.3f\n",
                 graph ? "graph" : "explicit-native-plan", slots_.size(), resident_, quota_.peak(),
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
                     graph_ ? "graph" : "explicit-native-plan", resident_, quota_.used());
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
    slot.graph = take(program_.create_invocation());
    require(slot.graph.bind(inputs));
    slots_.push_back(std::move(owned));
  }
  void execute(std::vector<std::byte> *host) {
    auto &slot = *slots_[next_];
    next_ = (next_ + 1) % slots_.size();
    auto work = take(slot.graph.submit());
    auto lease = take(work.wait());
    if (host) require(context_.download(lease->tensors()[0], *host));
  }
  const sem::Fragment &fixture_;
  bool graph_;
  mem::QuotaAllocator quota_;
  v::Context context_;
  c::WorkloadContext workload_;
  std::vector<v::Tensor> constants_;
  v::Program program_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::size_t next_ = 0, resident_ = 0;
  bool warmed_ = false;
};
}
int main(int argc, char **argv) { return hyle_d5_perf::run_main<Route>(argc, argv); }
