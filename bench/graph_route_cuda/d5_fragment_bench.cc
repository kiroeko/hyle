// Complete, synchronous consumer tasks over the ordinary Direct and Graph APIs.
// Mathematical recipes/oracles come from the already qualified D5 fixtures.
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <gtest/gtest.h>
#include <nanobench.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../tests/compute_models/d5_deepseek_official_fragments.h"
#include "../../tests/compute_models/d5_qwen35_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/graph/cuda.h"
#include "../common/perf_process.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace nb = ankerl::nanobench;
using Clock = std::chrono::steady_clock;
using Id = sem::ValueId;
using BF16 = hyle::simd::bfloat16_t;

void require(hyle::Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
std::string environment(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0)
    throw std::runtime_error("environment read failed");
  std::string result = value == nullptr ? "" : value;
  std::free(value);
  return result;
#else
  const auto *value = std::getenv(name);
  return value == nullptr ? "" : value;
#endif
}

sem::Fragment qwen_ffn() {
  const auto root = d5_qwen35_fixture::root();
  if (!root)
    throw std::runtime_error("HYLE_D5_QWEN35_FIXTURE is required; no skip");
  auto source = d5_qwen35_fixture::load_ffn(*root);
  sem::Fragment f{.profile = &workload_test::kModel, .region = 2};
  constexpr auto h = d5_qwen35_fixture::kHidden;
  constexpr auto n = d5_qwen35_fixture::kFfnHidden;
  const auto type = c::DataType::kBFloat16;
  auto x = f.input(sem::dense({1, h}, type));
  auto scale = f.constant(sem::dense({h}, type), std::move(source.norm_scale));
  auto wg = f.constant(take(sem::dense({n, h}, type).transposed()),
                       std::move(source.weights[0]));
  auto wu = f.constant(take(sem::dense({n, h}, type).transposed()),
                       std::move(source.weights[1]));
  auto wd = f.constant(take(sem::dense({h, n}, type).transposed()),
                       std::move(source.weights[2]));
  auto normalized = f.one(c::Operation(c::RmsNorm{}), {x, scale});
  auto gate = f.one(c::Operation(c::MatMul{}), {normalized, wg});
  auto up = f.one(c::Operation(c::MatMul{}), {normalized, wu});
  auto activated = f.one(c::Operation(c::GatedActivation{
      .inputs = c::GatedInputKind::kSeparate}), {gate, up});
  f.outputs = {f.one(c::Operation(c::MatMul{}), {activated, wd})};
  for (std::size_t i = 0; i < source.activations.size(); ++i)
    f.replays.push_back({{std::move(source.activations[i])},
                         {std::move(source.direct_expected[i])}});
  return f;
}

sem::Fragment deepseek_moe() {
  auto f = d5_deepseek_official_fragments::moe_chain(
      d5_moe_execution_cases::kCases[4]);
  // The consumer publishes its full result, not the diagnostic intermediate
  // taps from the correctness suite. Both arms get exactly this same DAG.
  f.outputs = {f.outputs[5]};
  f.exact_outputs.clear();
  for (auto &replay : f.replays) {
    auto final = std::move(replay.outputs[5]);
    replay.outputs = {std::move(final)};
  }
  std::vector<bool> needed(f.descriptions.size());
  needed[f.outputs[0].ordinal()] = true;
  for (auto node = f.nodes.rbegin(); node != f.nodes.rend(); ++node)
    if (std::ranges::any_of(node->outputs,
                           [&](Id id) { return needed[id.ordinal()]; }))
      for (auto id : node->inputs)
        needed[id.ordinal()] = true;
  std::erase_if(f.nodes, [&](const auto &node) {
    return std::ranges::none_of(node.outputs,
                               [&](Id id) { return needed[id.ordinal()]; });
  });
  return f;
}

struct Call final {
  mem::Buffer workspace;
  c::OperationInvocation invocation;
  std::vector<c::ConstTensorView> inputs;
  std::vector<c::TensorView> outputs;
};
struct Slot final {
  std::vector<mem::Buffer> buffers;
  std::vector<mem::BufferView> views;
  std::vector<Call> calls;
  std::vector<c::Completion> completions;
  g::Invocation graph;
};

class Route final {
public:
  Route(const sem::Fragment &fixture, bool graph) : fixture_(fixture), graph_(graph) {
    const auto start = Clock::now();
    if (fixture.preparation || !fixture.state_ports.empty())
      throw std::runtime_error("benchmark scope excludes cold weight recipes and recurrent state");
    context_ = graph ? take(g::cuda::create_context()) : take(c::cuda::create_context());
    memory_ = take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice));
    queue_ = take(context_.create_queue());
    workload_ = d5_qwen35_fixture::workload(*fixture.profile,
                                           c::WorkloadPhase::kDecode, fixture.region);
    constants_.resize(fixture.descriptions.size());
    for (std::size_t i = 0; i < fixture.constants.size(); ++i) {
      const auto id = fixture.constants[i];
      constants_[id.ordinal()] = allocate(fixture.description(id));
      require(constants_[id.ordinal()].view().write(fixture.constant_bytes[i]));
    }
    if (graph) {
      logical_ = sem::make_graph(fixture);
      std::vector<g::ConstantBinding> bindings;
      for (std::size_t i = 0; i < fixture.constants.size(); ++i) {
        const auto id = fixture.constants[i];
        bindings.push_back({logical_.constants[i], take(c::ConstTensorView::create(
            fixture.description(id), constants_[id.ordinal()].view()))});
      }
      auto compiled = take(g::cuda::compile(logical_.graph, context_, memory_.ref(),
          {.constants = bindings, .workload = workload_}));
      std::fprintf(stderr, "# graph steps=%zu regions=%zu workspace=%zu\n",
                   compiled.report.step_count(), compiled.report.region_count(),
                   compiled.report.workspace_bytes());
      for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
        const auto step = take(compiled.report.step(i));
        print_provider(step.provider);
      }
      executable_ = std::move(compiled.executable);
    } else {
      for (const auto &node : fixture.nodes) {
        if (node.operation.kind() == c::OperationKind::kView)
          continue;
        std::vector<c::TensorDesc> in, out;
        for (auto id : node.inputs) in.push_back(fixture.description(id));
        for (auto id : node.outputs) out.push_back(fixture.description(id));
        prepared_.push_back(take(c::prepare(context_, node.operation, in, out,
                                            memory_.ref(), workload_)));
        print_provider(prepared_.back().provider());
      }
    }
    for (const auto &replay : fixture.replays)
      make_slot(replay);
    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::fprintf(stderr, "# setup route=%s slots=%zu external_device_bytes=%zu milliseconds=%.3f\n",
                 graph ? "graph" : "direct", slots_.size(), allocated_, ms);
  }

  hyle::Status run() {
    auto &slot = *slots_[next_];
    next_ = (next_ + 1) % slots_.size();
    if (graph_) {
      auto completion = slot.graph.submit(queue_);
      if (!completion) return std::move(completion).error();
      return completion->wait(); // Local completion destruction is also charged.
    }
    hyle::Status status = hyle::Status::Ok();
    std::size_t accepted = 0;
    for (auto &call : slot.calls) {
      auto completion = call.invocation.submit(queue_, call.inputs, call.outputs);
      if (!completion) {
        status.Update(std::move(completion).error());
        break;
      }
      slot.completions[accepted++] = std::move(*completion);
    }
    // All commands are already enqueued in their normal stream order. Even
    // after a failure every accepted episode is settled before owner returns.
    for (std::size_t i = 0; i < accepted; ++i) {
      status.Update(slot.completions[i].wait());
      slot.completions[i] = {};
    }
    return status;
  }

  void verify() {
    next_ = 0;
    for (std::size_t repeat = 0; repeat < 2; ++repeat)
      for (std::size_t i = 0; i < slots_.size(); ++i) {
        require(run());
        auto &slot = *slots_[i];
        if (graph_ && slot.graph.in_flight())
          throw std::runtime_error("Graph still in flight after terminal wait");
        for (const auto &call : slot.calls)
          if (call.invocation.in_flight())
            throw std::runtime_error("Direct still in flight after terminal wait");
        const auto id = fixture_.outputs[0];
        const auto size = take(fixture_.description(id).required_bytes());
        std::vector<std::byte> bytes(size);
        require(slot.views[id.ordinal()].read(bytes));
        std::vector<float> actual;
        if (fixture_.description(id).data_type() == c::DataType::kBFloat16) {
          actual.resize(size / sizeof(BF16));
          for (std::size_t j = 0; j < actual.size(); ++j) {
            BF16 value;
            std::memcpy(&value, bytes.data() + j * sizeof(value), sizeof(value));
            actual[j] = static_cast<float>(half_test::decode(value));
          }
        } else {
          actual.resize(size / sizeof(float));
          std::memcpy(actual.data(), bytes.data(), size);
        }
        const auto &expected = fixture_.replays[i].outputs[0];
        if (actual.size() != expected.size())
          throw std::runtime_error("oracle size mismatch");
        for (float value : actual)
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite result");
        if (fixture_.profile == &workload_test::kModel) {
          for (std::size_t j = 0; j < actual.size(); ++j)
            if (std::abs(actual[j] - expected[j]) > std::abs(expected[j]) * .008F + 1e-8F)
              throw std::runtime_error("Qwen BF16 FFN oracle mismatch");
        } else {
          sem::expect_output(fixture_, actual, expected);
          if (::testing::Test::HasFailure())
            throw std::runtime_error("DeepSeek MoE oracle mismatch");
        }
      }
    next_ = 0;
  }

private:
  mem::Buffer allocate(const c::TensorDesc &desc) {
    auto size = take(desc.required_bytes());
    allocated_ += size;
    return take(mem::Buffer::create(memory_.ref(), size, 256));
  }
  static void print_provider(c::ProviderIdentity provider) {
    std::fprintf(stderr, "# provider=%.*s version=%.*s\n",
                 int(provider.name.size()), provider.name.data(),
                 int(provider.version.size()), provider.version.data());
  }
  void make_slot(const sem::Replay &replay) {
    auto owned = std::make_unique<Slot>();
    auto &slot = *owned;
    slot.buffers.resize(fixture_.descriptions.size());
    slot.views.resize(fixture_.descriptions.size());
    const auto allocate_value = [&](Id id) {
      slot.buffers[id.ordinal()] = allocate(fixture_.description(id));
      slot.views[id.ordinal()] = slot.buffers[id.ordinal()].view();
    };
    for (auto id : fixture_.constants)
      slot.views[id.ordinal()] = constants_[id.ordinal()].view();
    for (std::size_t i = 0; i < fixture_.inputs.size(); ++i) {
      const auto id = fixture_.inputs[i];
      allocate_value(id);
      require(slot.views[id.ordinal()].write(replay.inputs[i]));
    }
    if (graph_) {
      slot.graph = take(executable_.create_invocation());
      for (std::size_t i = 0; i < fixture_.inputs.size(); ++i) {
        const auto id = fixture_.inputs[i];
        require(slot.graph.bind(logical_.inputs[i], take(c::ConstTensorView::create(
            fixture_.description(id), slot.views[id.ordinal()]))));
      }
      for (std::size_t i = 0; i < fixture_.outputs.size(); ++i) {
        const auto id = fixture_.outputs[i];
        allocate_value(id);
        require(slot.graph.bind(logical_.outputs[i], take(c::TensorView::create(
            fixture_.description(id), slot.views[id.ordinal()]))));
      }
      require(g::cuda::prepare_replay(slot.graph, queue_));
    } else {
      std::size_t prepared = 0;
      for (const auto &node : fixture_.nodes) {
        if (node.operation.kind() == c::OperationKind::kView) {
          const auto &view = *node.operation.get_if<c::View>();
          auto source = slot.views[node.inputs[0].ordinal()];
          if (view.kind() == c::ViewKind::kSlice) {
            const auto &input = fixture_.description(node.inputs[0]);
            const auto axis = view.axis() < 0 ? view.axis() + std::int32_t(input.rank()) : view.axis();
            source = take(source.subview(std::size_t(view.start()) *
                std::size_t(input.byte_strides()[axis]),
                take(fixture_.description(node.outputs[0]).required_bytes())));
          } else if (view.kind() != c::ViewKind::kReshape && view.kind() != c::ViewKind::kPermute) {
            throw std::runtime_error("view outside benchmark scope");
          }
          slot.views[node.outputs[0].ordinal()] = source;
          continue;
        }
        Call call;
        auto &operation = prepared_[prepared++];
        auto workspace = operation.workspace_requirement();
        if (workspace.size) {
          call.workspace = take(mem::Buffer::create(memory_.ref(), workspace.size, workspace.alignment));
          allocated_ += workspace.size;
        }
        call.invocation = take(operation.create_invocation(call.workspace.view()));
        for (auto id : node.inputs)
          call.inputs.push_back(take(c::ConstTensorView::create(
              fixture_.description(id), slot.views[id.ordinal()])));
        for (auto id : node.outputs) {
          allocate_value(id);
          call.outputs.push_back(take(c::TensorView::create(
              fixture_.description(id), slot.views[id.ordinal()])));
        }
        slot.calls.push_back(std::move(call));
      }
      slot.completions.resize(slot.calls.size());
    }
    slots_.push_back(std::move(owned));
  }
  const sem::Fragment &fixture_;
  bool graph_;
  c::Context context_;
  c::cuda::MemoryResource memory_;
  c::Queue queue_;
  c::WorkloadContext workload_;
  std::vector<mem::Buffer> constants_;
  std::vector<c::PreparedOperation> prepared_;
  sem::GraphCase logical_;
  g::Executable executable_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::size_t next_ = 0;
  std::size_t allocated_ = 0;
};

void measure(sem::Fragment fixture, const char *name, bool verify_only) {
  const auto aa = environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "selected" && aa != "peer")
    throw std::runtime_error("invalid A/A route");
  const bool allocation_first = !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
  std::unique_ptr<Route> candidate, peer;
  if (allocation_first) peer = std::make_unique<Route>(fixture, aa == "selected");
  candidate = std::make_unique<Route>(fixture, aa != "peer");
  if (!allocation_first) peer = std::make_unique<Route>(fixture, aa == "selected");
  candidate->verify();
  peer->verify();
  std::fprintf(stderr, "# verify PASS profile=%s replays=%zu aa=%s\n", name,
               fixture.replays.size(), aa.empty() ? "off" : aa.c_str());
  if (verify_only) return;
  nb::Bench bench;
  bench.title(name).relative(true).minEpochTime(std::chrono::milliseconds(8));
  const auto selected = [&] { require(candidate->run()); };
  const auto reference = [&] { require(peer->run()); };
  if (!environment("HYLE_BENCH_RIVAL_FIRST").empty())
    (void)bench.compare("Direct complete fragment", reference, "Graph complete fragment", selected);
  else
    (void)bench.compare("Graph complete fragment", selected, "Direct complete fragment", reference);
  candidate->verify();
  peer->verify();
}
} // namespace

int main(int argc, char **argv) {
  try {
    ::testing::InitGoogleTest(&argc, argv);
    const bool verify = !environment("HYLE_BENCH_VERIFY_ONLY").empty();
    // No ad-hoc run may masquerade as a qualified unified-instrument receipt.
    if (!verify && environment("HYLE_D5_DIAGNOSTIC") != "1")
      throw std::runtime_error("sampling is diagnostic-only until the A6 instrument contract is bound");
    const auto affinity = hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
    nb::doNotOptimizeAway(affinity);
    measure(qwen_ffn(), "Qwen3.5-2B official BF16 FFN M1 H2048 N6144", verify);
    measure(deepseek_moe(), "DeepSeek V4 Flash official FP4-FP8 MoE H4096 N2048 top6", verify);
    return ::testing::Test::HasFailure() ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "D5 CUDA fragment benchmark failed: %s\n", error.what());
    return 1;
  }
}
