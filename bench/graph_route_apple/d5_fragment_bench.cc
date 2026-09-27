#include "../graph_route_cpu/d5_common.h"
#include "../../tests/compute_route_apple/tensor_test_support.h"
#include "hyle/graph/apple.h"
namespace {
using namespace hyle_d5_perf;
namespace a=c::apple;
namespace mx=mlx::core;
using apple_tensor_test::ready;
using apple_tensor_test::stream;
mx::Shape native_shape(const c::TensorDesc &description, bool stored) {
  mx::Shape result;
  const auto dimensions = description.dimensions();
  result.reserve(dimensions.size());
  if (stored && dimensions.size() == 2) {
    result.push_back(static_cast<int>(dimensions[1]));
    result.push_back(static_cast<int>(dimensions[0]));
  } else {
    for (const auto dimension : dimensions)
      result.push_back(static_cast<int>(dimension));
  }
  return result;
}

mx::array upload(const a::Context &, const c::TensorDesc &description,
                 std::span<const std::byte> bytes) {
  if (!description.is_plain()) {
    const auto spec = a::input_spec(description).value();
    std::vector<std::uint8_t> data(bytes.size());
    std::memcpy(data.data(), bytes.data(), bytes.size());
    auto result = ready(mx::array(data.data(), spec.shape, spec.dtype));
    return result;
  }
  const bool transposed = !description.is_dense() && description.rank() == 2;
  const auto shape = native_shape(description, transposed);
  mx::array result = [&]() -> mx::array {
    switch (description.data_type()) {
    case c::DataType::kBool: {
      std::vector<std::uint8_t> data(bytes.size());
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::astype(mx::array(data.data(), shape, mx::uint8), mx::bool_,
                        stream());
    }
    case c::DataType::kFloat16: {
      std::vector<std::uint16_t> data(bytes.size() / sizeof(std::uint16_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::view(mx::array(data.data(), shape, mx::uint16), mx::float16,
                      stream());
    }
    case c::DataType::kBFloat16: {
      std::vector<std::uint16_t> data(bytes.size() / sizeof(std::uint16_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::view(mx::array(data.data(), shape, mx::uint16), mx::bfloat16,
                      stream());
    }
    case c::DataType::kFloat32: {
      std::vector<float> data(bytes.size() / sizeof(float));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::array(data.data(), shape, mx::float32);
    }
    case c::DataType::kInt32: {
      std::vector<std::int32_t> data(bytes.size() / sizeof(std::int32_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::array(data.data(), shape, mx::int32);
    }
    default:
      throw std::runtime_error("unsupported Qwen3.5 Apple fixture type");
    }
  }();
  if (transposed)
    result = mx::transpose(result, stream());
  return ready(std::move(result));
}

struct Region final {
  std::vector<Id> inputs, outputs;
  a::Program program;
};
struct Slot final {
  std::vector<std::optional<mx::array>> values;
  std::vector<a::Invocation> direct;
  a::Invocation graph;
};

class Route final {
public:
  Route(const sem::Fragment &fixture, bool graph)
      : fixture_(fixture), graph_(graph), context_(take(a::Context::create())) {
    const auto start = Clock::now();
    if (fixture.preparation || !fixture.state_ports.empty())
      throw std::runtime_error("benchmark scope excludes cold weight recipes and recurrent state");
    const auto environment = context_.environment();
    if (!environment) throw std::runtime_error(std::string(environment.error().message()));
    const auto &env = *environment;
    std::fprintf(stderr, "# D5_APPLE_ENV device=%s arch=%s os=%s mlx=%s ggml=%s sdk=%u\n",
                 env.device_name.c_str(), env.architecture.c_str(), env.os_runtime.c_str(),
                 env.mlx_version.c_str(), env.ggml_revision.c_str(), env.build_sdk);
    workload_ = d5_qwen35_fixture::workload(*fixture.profile, c::WorkloadPhase::kDecode, fixture.region);
    constants_.resize(fixture.descriptions.size());
    std::vector<mx::array> bindings;
    std::size_t constant_bytes = 0;
    for (std::size_t i = 0; i < fixture.constants.size(); ++i) {
      const auto id = fixture.constants[i];
      constants_[id.ordinal()] = upload(context_, fixture.description(id), fixture.constant_bytes[i]);
      bindings.push_back(*constants_[id.ordinal()]);
      constant_bytes += fixture.constant_bytes[i].size();
    }
    if (graph) {
      auto logical = sem::make_graph(fixture);
      g::apple::CompileReport report;
      program_ = take(g::apple::compile(context_, logical.graph, bindings,
                                       {.report = &report, .workload = workload_}));
      std::fprintf(stderr, "# graph nodes=%zu policy=%s\n", report.nodes.size(),
                   report.implementation_policy.c_str());
    } else {
      // Only the fixed recipe is partitioned. Keep consecutive typed MLX
      // calculations together; each custom ggml program stays an intact region.
      for (std::size_t begin = 0; begin < fixture.nodes.size();) {
        const bool native = fixture.nodes[begin].operation.kind() == c::OperationKind::kCustom;
        std::size_t end = begin + 1;
        if (!native)
          while (end < fixture.nodes.size() &&
                 fixture.nodes[end].operation.kind() != c::OperationKind::kCustom) ++end;
        prepare_region(begin, end, native);
        begin = end;
      }
    }
    for (const auto &replay : fixture.replays) {
      auto owned = std::make_unique<Slot>();
      owned->values = constants_;
      std::vector<mx::array> inputs;
      for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
        auto id = fixture.inputs[i];
        owned->values[id.ordinal()] = upload(context_, fixture.description(id), replay.inputs[i]);
        inputs.push_back(*owned->values[id.ordinal()]);
      }
      if (graph_) {
        owned->graph = take(program_.create_invocation());
        require(owned->graph.bind(inputs));
      } else {
        for (const auto &region : regions_)
          owned->direct.push_back(take(region.program.create_invocation()));
      }
      slots_.push_back(std::move(owned));
    }
    std::fprintf(stderr, "# setup route=%s slots=%zu direct_regions=%zu logical_constant_bytes=%zu milliseconds=%.3f\n",
                 graph ? "graph" : "direct", slots_.size(), regions_.size(), constant_bytes,
                 std::chrono::duration<double, std::milli>(Clock::now()-start).count());
  }
  hyle::Status run() { execute(nullptr); return hyle::Status::Ok(); }
  void verify() {
    next_ = 0;
    for (int repeat = 0; repeat < 2; ++repeat)
      for (std::size_t i = 0; i < slots_.size(); ++i) {
        std::vector<std::byte> bytes(*fixture_.description(fixture_.outputs[0]).required_bytes());
        execute(&bytes);
        verify_output(fixture_, i, bytes);
      }
    next_ = 0;
  }
private:
  void prepare_region(std::size_t begin, std::size_t end, bool native) {
    Region region;
    if (native) {
      const auto &node = fixture_.nodes[begin];
      region.inputs = node.inputs;
      region.outputs = node.outputs;
      std::vector<a::InputSpec> specs;
      for (auto id : region.inputs) specs.push_back(take(a::input_spec(fixture_.description(id))));
      region.program = take(context_.prepare(node.operation, specs, workload_));
    } else {
      std::vector<bool> produced(fixture_.descriptions.size()), external(produced.size());
      for (std::size_t i = begin; i < end; ++i) {
        for (auto id : fixture_.nodes[i].inputs)
          if (!produced[id.ordinal()] && !constants_[id.ordinal()] && !external[id.ordinal()]) {
            region.inputs.push_back(id);
            external[id.ordinal()] = true;
          }
        for (auto id : fixture_.nodes[i].outputs) produced[id.ordinal()] = true;
      }
      std::vector<bool> needed(produced.size());
      for (auto id : fixture_.outputs) needed[id.ordinal()] = true;
      for (std::size_t i = end; i < fixture_.nodes.size(); ++i)
        for (auto id : fixture_.nodes[i].inputs) needed[id.ordinal()] = true;
      for (std::size_t i = begin; i < end; ++i)
        for (auto id : fixture_.nodes[i].outputs)
          if (needed[id.ordinal()]) region.outputs.push_back(id);
      auto nodes = std::vector(fixture_.nodes.begin()+begin, fixture_.nodes.begin()+end);
      auto lower = [nodes = std::move(nodes), constants = constants_,
                    inputs = region.inputs, outputs = region.outputs, workload = workload_]
          (const std::vector<mx::array> &arguments, mx::Stream stream)
          -> hyle::Result<std::vector<mx::array>> {
        auto values = constants;
        for (std::size_t i = 0; i < inputs.size(); ++i) values[inputs[i].ordinal()] = arguments[i];
        for (const auto &node : nodes) {
          std::vector<mx::array> inputs;
          for (auto id : node.inputs) inputs.push_back(*values[id.ordinal()]);
          HYLE_ASSIGN_OR_RETURN(auto result, a::lower(node.operation, inputs, stream, workload));
          for (std::size_t i = 0; i < node.outputs.size(); ++i)
            values[node.outputs[i].ordinal()] = std::move(result[i]);
        }
        std::vector<mx::array> result;
        for (auto id : outputs) result.push_back(*values[id.ordinal()]);
        return result;
      };
      std::vector<a::InputSpec> specs;
      for (auto id : region.inputs) specs.push_back(take(a::input_spec(fixture_.description(id))));
      region.program = take(context_.compile(std::move(lower), specs, workload_));
    }
    regions_.push_back(std::move(region));
  }
  static void read(const mx::array &value, std::vector<std::byte> *host) {
    if (!host) return;
    const auto words = apple_tensor_test::bits(value);
    for (std::size_t i = 0; i < words.size(); ++i)
      std::memcpy(host->data()+i*value.itemsize(), &words[i], value.itemsize());
  }
  void execute(std::vector<std::byte> *host) {
    auto &slot = *slots_[next_];
    next_ = (next_ + 1) % slots_.size();
    if (graph_) {
      auto work = take(slot.graph.submit());
      auto lease = take(work.wait());
      read(lease->arrays()[0], host);
    } else {
      for (std::size_t i = 0; i < regions_.size(); ++i) {
        const auto &region = regions_[i];
        std::vector<mx::array> inputs;
        inputs.reserve(region.inputs.size());
        for (auto id : region.inputs) inputs.push_back(*slot.values[id.ordinal()]);
        require(slot.direct[i].bind(inputs));
        auto work = take(slot.direct[i].submit());
        auto lease = take(work.wait());
        for (std::size_t j = 0; j < region.outputs.size(); ++j)
          slot.values[region.outputs[j].ordinal()] = lease->arrays()[j];
      }
      read(*slot.values[fixture_.outputs[0].ordinal()], host);
      for (const auto &region : regions_)
        for (auto id : region.outputs) slot.values[id.ordinal()].reset();
    }
  }
  const sem::Fragment &fixture_;
  bool graph_;
  a::Context context_;
  c::WorkloadContext workload_;
  std::vector<std::optional<mx::array>> constants_;
  a::Program program_;
  std::vector<Region> regions_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::size_t next_ = 0;
};
}
int main(int argc, char **argv) { return hyle_d5_perf::run_main<Route>(argc, argv); }
