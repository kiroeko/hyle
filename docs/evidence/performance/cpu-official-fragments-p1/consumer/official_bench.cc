#include "official_common.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/graph/cpu.h"
namespace {
using namespace hyle_d5_perf;
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
    if (environment("OMP_NUM_THREADS") != "1" || environment("MKL_NUM_THREADS") != "1")
      throw std::runtime_error("official CPU P1 requires the fixed one-thread vendor policy");
    std::fprintf(stderr, "# D5_CPU_ENV omp=1 mkl=1 libtorch=2.8.0-cpu replay=managed\n");

    if (fixture.preparation || !fixture.state_ports.empty())
      throw std::runtime_error("benchmark scope excludes cold weight recipes and recurrent state");
    context_ = graph ? take(g::cpu::create_context()) : take(c::cpu::create_context());
    memory_ = mem::system_resource();
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
      g::CompileOptions options;
      options.constants = bindings;
      options.workload = workload_;
      // Ordinary managed replay: ATen owns its native temporary tensors.
      auto compiled = take(g::compile(logical_.graph, context_, options));
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
                                            memory_, workload_)));
        print_provider(prepared_.back().provider());
      }
    }
    for (const auto &replay : fixture.replays)
      make_slot(replay);
    const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::fprintf(stderr, "# setup route=%s slots=%zu external_host_bytes=%zu milliseconds=%.3f\n",
                 graph ? "graph" : "direct", slots_.size(), allocated_, ms);
  }

  hyle::Status run() {
    // One measured call visits every changed slot exactly once in both arms.
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      auto status = run_one();
      if (!status.ok()) return status;
    }
    return hyle::Status::Ok();
  }

  hyle::Status run_one() {
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
    for (std::size_t repeat = 0; repeat < 1; ++repeat)
      for (std::size_t i = 0; i < slots_.size(); ++i) {
        require(run_one());
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
          sem::expect_output(fixture_, actual, expected, {},
              fixture_.exact_output(0), fixture_.id_validation(0),
              fixture_.bfloat_chain_output(0));
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
    return take(mem::Buffer::create(memory_, size, 256));
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
      const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue_}};
      require(slot.graph.prepare_replay(queues));
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
          call.workspace = take(mem::Buffer::create(memory_, workspace.size, workspace.alignment));
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
  mem::MemoryResourceRef memory_;
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

}
int main(int argc, char **argv) { return hyle_d5_perf::run_official_main<Route>(argc, argv); }
