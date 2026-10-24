#include <hyle/compute/apple.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/compute/low_precision.h>
#include <hyle/compute/model_ops.h>
#include <hyle/graph/apple.h>
#include <hyle/graph/executable.h>
#include <hyle/graph/result.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../compute_models/gguf_nvfp4_fixture.h"

namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace mx = mlx::core;
using Weights = nvfp4_test::Fixture;

template <class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}));
}
mx::array ready(mx::array value) {
  mx::async_eval({value});
  mx::synchronize(mx::default_stream(mx::Device::gpu));
  value.wait();
  return value;
}
std::vector<std::uint32_t> bits(const mx::array &value) {
  require(value.dtype() == mx::float32, "output dtype must remain F32");
  std::vector<std::uint32_t> result(value.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = std::bit_cast<std::uint32_t>(value.data<float>()[i]);
  return result;
}

a::Program prepare(a::Context &context, bool graph, c::Operation operation,
                   std::span<const c::TensorDesc> physical,
                   std::span<const c::TensorDesc> logical, bool broadcast = false) {
  if (!graph) {
    std::vector<a::InputSpec> specs;
    for (const auto &description : logical) specs.push_back(take(a::input_spec(description)));
    return take(context.prepare(operation, specs));
  }
  auto builder = take(g::GraphBuilder::create());
  std::vector<g::ValueId> inputs;
  for (const auto &description : physical) inputs.push_back(take(builder.input(description)));
  if (broadcast) {
    const auto shape = take(c::TensorShape::create(logical[1].dimensions()));
    inputs[1] = take(builder.add(take(c::View::broadcast_to(shape)), inputs[1]));
  }
  const auto node = take(builder.add_node(operation, inputs));
  check(builder.publish(take(builder.result(node, 0))));
  return take(g::apple::compile(context, take(std::move(builder).finish())));
}

struct DotReference {
  std::vector<float> rounded;
  std::vector<double> absolute_products;
};

// The oracle uses the fixture's original logical codes, not a decoder for the
// implementation's bytes. F64 accumulation is independent of Metal's F32 tree.
DotReference reference(const Weights &weight, std::span<const float> input,
                       std::span<const std::int32_t> ids,
                       int tokens, int slots, bool shared) {
  DotReference result;
  result.rounded.resize(static_cast<std::size_t>(tokens * slots * weight.rows));
  result.absolute_products.resize(result.rounded.size());
  for (int t = 0; t < tokens; ++t)
    for (int slot = 0; slot < slots; ++slot)
      for (std::int64_t n = 0; n < weight.rows; ++n) {
        const auto expert = ids.empty() ? 0 : ids[static_cast<std::size_t>(t * slots + slot)];
        double sum = 0, absolute = 0;
        for (std::int64_t k = 0; k < weight.columns; ++k) {
          const auto x = input[static_cast<std::size_t>((t * (shared ? 1 : slots) +
              (shared ? 0 : slot)) * weight.columns + k)];
          const auto w = weight.decoded[static_cast<std::size_t>((expert * weight.rows + n) * weight.columns + k)];
          const double product = double(x) * double(w);
          sum += product;
          absolute += std::abs(product);
        }
        const auto out = static_cast<std::size_t>((t * slots + slot) * weight.rows + n);
        result.rounded[out] = static_cast<float>(sum);
        result.absolute_products[out] = absolute;
      }
  return result;
}

void expect_dot_values(std::span<const float> actual, const DotReference &expected, int depth) {
  require(actual.size() == expected.rounded.size(), "dot output size");
  // Conservative F32 product/add/tree forward bound, plus one final-output ULP.
  // This is not a cross-backend bit-equality claim. The one-term witness below
  // separately rejects pre-narrowing F32 A, which this general bound must not hide.
  const double u = std::ldexp(1.0, -24);
  const double operations = 2.0 * depth + std::ceil(std::log2(double(depth)));
  const double gamma = operations * u / (1.0 - operations * u);
  for (std::size_t i = 0; i < expected.rounded.size(); ++i) {
    const float value = actual[i], wanted = expected.rounded[i];
    const double ulp = double(std::nextafter(wanted, std::numeric_limits<float>::infinity())) - wanted;
    const double bound = gamma * expected.absolute_products[i] + std::abs(ulp) +
        depth * double(std::numeric_limits<float>::denorm_min());
    require(std::isfinite(value) && std::abs(double(value) - wanted) <= bound,
        "F32 dot forward bound at " + std::to_string(i));
  }
}
void expect_dot(const mx::array &actual, const DotReference &expected, int depth) {
  require(actual.dtype() == mx::float32, "dot output dtype");
  expect_dot_values({actual.data<float>(), actual.size()}, expected, depth);
}

std::int64_t unit_lane(const Weights &weight) {
  for (std::int64_t k = 0; k < weight.columns; ++k)
    if (weight.decoded[static_cast<std::size_t>(k)] == 1.0f) return k;
  throw std::runtime_error("fixture must contain expert0/row0 W=1");
}

void expect_unchanged(const mx::array &value, std::span<const std::uint32_t> original) {
  require(bits(value) == std::vector<std::uint32_t>(original.begin(), original.end()), "retained result changed");
}

void project_domain(bool graph, int tokens, int mode) {
  // mode -1: MatMul; 0: Grouped AS; 1: Grouped A1; 2: Grouped real slot-zero View.
  constexpr int depth = 128, rows = 5, experts = 3;
  const bool grouped = mode >= 0, shared = mode > 0;
  const int slots = grouped ? 2 : 1;
  std::printf("RUN %s %s T%d S%d mode%d\n", graph ? "Graph" : "Direct",
      grouped ? "Grouped" : "MatMul", tokens, slots, mode);
  std::fflush(stdout);
  const auto wd = grouped ? take(c::gguf_grouped_weight_description(
      c::GgufQuantizedType::kNvfp4, c::DataType::kFloat32, experts, rows, depth)) :
      take(c::gguf_weight_description(c::GgufQuantizedType::kNvfp4, c::DataType::kFloat32, depth, rows));
  const auto ad = grouped ? dense(c::DataType::kFloat32, {tokens, shared ? 1 : slots, depth}) :
      dense(c::DataType::kFloat32, {tokens, depth});
  const auto idd = dense(c::DataType::kInt32, {tokens, slots});
  auto logical_ad = ad;
  if (mode == 2) {
    const std::array<std::int64_t, 3> shape{tokens, slots, depth};
    const std::array<std::int64_t, 3> stride{depth * 4, 0, 4};
    logical_ad = take(c::TensorDesc::strided_bytes(c::DataType::kFloat32, shape, stride));
  }
  std::vector<c::TensorDesc> physical = grouped ? std::vector{wd, ad, idd} : std::vector{ad, wd};
  auto logical = physical;
  if (grouped) logical[1] = logical_ad;
  auto context = take(a::Context::create());
  auto program = prepare(context, graph, grouped ? c::model_ops::grouped_matmul() : c::Operation(c::MatMul{}),
      physical, logical, mode == 2);
  auto retained = take(program.create_invocation()), active = take(program.create_invocation());
  std::optional<c::ResultLease<a::Results>> old;
  std::vector<std::uint32_t> old_bits;
  std::vector<std::vector<mx::array>> keep_inputs;
  const float witness = 1.0f + std::ldexp(1.0f, -20);
  for (int pass = 0; pass < 3; ++pass) {
    const Weights weight(rows, depth, grouped ? experts : 1, static_cast<unsigned>(pass));
    const auto lane = unit_lane(weight);
    std::vector<float> input(static_cast<std::size_t>(tokens * (shared ? 1 : slots) * depth), 0);
    for (std::size_t row = 0; row < input.size() / depth; ++row) {
      if (pass == 0) input[row * depth + static_cast<std::size_t>(lane)] = witness;
      else for (int k = 0; k < depth; ++k)
        input[row * depth + static_cast<std::size_t>(k)] =
            float(int((row * 7 + std::size_t(k) * 3 + std::size_t(pass)) % 19) - 9) / 32.0f +
            (k == lane ? std::ldexp(1.0f, -20) : 0.0f);
    }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens * slots), 0);
    if (pass != 0) for (int t = 0; t < tokens; ++t) for (int s = 0; s < slots; ++s)
      ids[static_cast<std::size_t>(t * slots + s)] = (t + pass + (pass == 1 ? s : 0)) % experts;
    const auto xshape = grouped ? mx::Shape{tokens, shared ? 1 : slots, depth} : mx::Shape{tokens, depth};
    auto base = ready(mx::array(input.data(), xshape, mx::float32));
    auto x = base;
    if (!graph && mode == 2) {
      x = ready(mx::broadcast_to(base, {tokens, slots, depth}));
      require(x.strides()[1] == 0 && x.data<float>() == base.data<float>(), "slot-zero must be a storage View");
    }
    auto bank = ready(mx::array(weight.bytes.data(), take(a::input_spec(wd)).shape, mx::uint8));
    std::vector<mx::array> args = grouped ? std::vector{bank, x,
        ready(mx::array(ids.data(), {tokens, slots}, mx::int32))} : std::vector{x, bank};
    if (!keep_inputs.empty()) {
      const auto position = grouped ? 1U : 0U;
      require(args[position].data<float>() != keep_inputs.back()[position].data<float>(), "rebind must change A address");
      const auto wposition = grouped ? 0U : 1U;
      require(args[wposition].data<std::uint8_t>() != keep_inputs.back()[wposition].data<std::uint8_t>(), "rebind must change W address");
    }
    const auto expected = reference(weight, input, grouped ? std::span<const std::int32_t>(ids) :
        std::span<const std::int32_t>{}, tokens, slots, shared);
    auto &call = pass == 0 ? retained : active;
    check(call.bind(args));
    auto work = take(call.submit());
    auto result = take(work.wait());
    const auto &output = result.get()->arrays()[0];
    expect_dot(output, expected, depth);
    require(std::equal(weight.bytes.begin(), weight.bytes.end(), bank.data<std::uint8_t>()), "packed W was mutated");
    if (pass == 0) for (int t = 0; t < tokens; ++t) for (int s = 0; s < slots; ++s)
      require(std::bit_cast<std::uint32_t>(output.data<float>()[(t * slots + s) * rows]) ==
          std::bit_cast<std::uint32_t>(witness), "F32 activation was narrowed before W=1 dot");
    if (old) expect_unchanged(old->get()->arrays()[0], old_bits);
    if (pass == 0) { old_bits = bits(output); old.emplace(std::move(result)); }
    if (grouped) {
      require(std::equal(ids.begin(), ids.end(), args[2].data<std::int32_t>()), "IDs were mutated");
      if (pass == 2) for (const auto invalid : {-1, experts}) {
        auto bad_ids = ids; bad_ids.back() = invalid;
        auto bad_args = args;
        bad_args[2] = ready(mx::array(bad_ids.data(), {tokens, slots}, mx::int32));
        auto failed = take(program.create_invocation());
        check(failed.bind(bad_args));
        auto bad_work = take(failed.submit());
        auto bad_result = bad_work.wait();
        require(!bad_result && bad_result.error().IsOutOfRange(), "invalid expert must report OutOfRange");
        require(!bad_work.result(), "invalid expert must not publish a result");
        require(!failed.submit(), "accepted failed invocation must stay terminal");
        expect_unchanged(old->get()->arrays()[0], old_bits);
        require(std::equal(bad_ids.begin(), bad_ids.end(), bad_args[2].data<std::int32_t>()), "bad IDs were mutated");
        auto fresh = take(program.create_invocation());
        check(fresh.bind(args));
        auto recovered = take(fresh.submit());
        auto value = take(recovered.wait());
        expect_dot(value.get()->arrays()[0], expected, depth);
      }
    }
    keep_inputs.push_back(std::move(args));
  }
  check(context.close());
  expect_unchanged(old->get()->arrays()[0], old_bits);
  std::printf("PASS %s %s T%d S%d mode%d: F64 dot, F32 witness, two slots/rebind/retained result%s\n",
      graph ? "Graph" : "Direct", grouped ? "Grouped" : "MatMul", tokens, slots, mode,
      grouped ? ", checked IDs/fresh recovery" : "");
}

void gather_domain(bool graph) {
  constexpr int rows = 43, depth = 128, selections = rows + 2;
  std::printf("RUN %s Gather all legal scale codes\n", graph ? "Graph" : "Direct");
  std::fflush(stdout);
  std::array<std::uint8_t, 128> scale_codes;
  std::iota(scale_codes.begin(), scale_codes.end(), std::uint8_t{0});
  const auto wd = take(take(c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, depth, rows)).transposed());
  const std::array specs{wd, dense(c::DataType::kInt32, {selections})};
  auto context = take(a::Context::create());
  auto program = prepare(context, graph, c::Operation(c::Gather{}), specs, specs);
  auto retained = take(program.create_invocation()), active = take(program.create_invocation());
  std::optional<c::ResultLease<a::Results>> old;
  std::vector<std::uint32_t> old_bits;
  std::vector<std::vector<mx::array>> keep_inputs;
  for (int pass = 0; pass < 3; ++pass) {
    const Weights weight(rows, depth, 1, static_cast<unsigned>(pass), scale_codes);
    std::vector<std::int32_t> ids(selections);
    for (int i = 0; i < selections; ++i) ids[static_cast<std::size_t>(i)] = (i * (pass == 1 ? rows - 1 : 1) + pass) % rows;
    const std::vector<mx::array> args{
        ready(mx::array(weight.bytes.data(), take(a::input_spec(wd)).shape, mx::uint8)),
        ready(mx::array(ids.data(), {selections}, mx::int32))};
    std::vector<std::uint32_t> expected;
    for (const auto id : ids) for (int k = 0; k < depth; ++k)
      expected.push_back(std::bit_cast<std::uint32_t>(weight.decoded[static_cast<std::size_t>(id * depth + k)]));
    auto &call = pass == 0 ? retained : active;
    check(call.bind(args));
    auto work = take(call.submit());
    auto result = take(work.wait());
    require(bits(result.get()->arrays()[0]) == expected, "Gather decode/lane/scale bits, including signed zero");
    require(std::equal(weight.bytes.begin(), weight.bytes.end(), args[0].data<std::uint8_t>()), "Gather packed W was mutated");
    if (old) expect_unchanged(old->get()->arrays()[0], old_bits);
    if (pass == 0) { old_bits = expected; old.emplace(std::move(result)); }
    if (pass == 2) for (const auto invalid : {-1, rows}) {
      auto bad_ids = ids; bad_ids.back() = invalid;
      auto bad_args = args;
      bad_args[1] = ready(mx::array(bad_ids.data(), {selections}, mx::int32));
      auto failed = take(program.create_invocation());
      check(failed.bind(bad_args));
      auto bad_work = take(failed.submit());
      auto bad_result = bad_work.wait();
      require(!bad_result && bad_result.error().IsOutOfRange(), "invalid row must report OutOfRange");
      require(!bad_work.result(), "invalid row must not publish a result");
      require(!failed.submit(), "accepted failed gather must stay terminal");
      expect_unchanged(old->get()->arrays()[0], old_bits);
      require(std::equal(bad_ids.begin(), bad_ids.end(), bad_args[1].data<std::int32_t>()), "Gather IDs were mutated");
      auto fresh = take(program.create_invocation());
      check(fresh.bind(args));
      auto recovered = take(fresh.submit());
      auto value = take(recovered.wait());
      require(bits(value.get()->arrays()[0]) == expected, "fresh Gather recovery");
    }
    keep_inputs.push_back(args);
  }
  check(context.close());
  expect_unchanged(old->get()->arrays()[0], old_bits);
  std::printf("PASS %s Gather: all 128 scale codes/all nibble lanes, signed zero, checked IDs/rebind/retained result\n",
      graph ? "Graph" : "Direct");
}

template <class T>
c::ReadLease common_input(const a::MemoryResource &memory, const c::TensorDesc &description,
                          std::span<const T> values) {
  auto native = take(memory.allocate(description));
  auto host = take(hyle::mem::Buffer::create(hyle::mem::system_resource(), values.size_bytes(), alignof(T)));
  check(host.view().write(std::as_bytes(values)));
  check(memory.copy_from_host(native, take(c::ConstTensorView::create(description, host.view()))));
  return take(memory.import_readonly(native, description));
}
std::vector<float> read_tensor(c::ConstTensorView tensor) {
  require(tensor.description().data_type() == c::DataType::kFloat32, "common Graph output dtype");
  std::vector<float> values(take(tensor.description().element_count()));
  check(tensor.storage().read(std::as_writable_bytes(std::span(values))));
  return values;
}

// This is the ordinary shared Executable/Invocation journey, distinct from the
// native graph::apple::compile tests above. Both operations read original packed
// NVFP4 storage: selected-row Gather and M32 MatMul with an F32 activation.
void common_graph_domain() {
  constexpr int tokens = 32, depth = 128, rows = 5, selected = 32;
  std::puts("RUN common Graph original packed Gather and original packed MatMul");
  std::fflush(stdout);
  const Weights weight(rows, depth);
  const auto wd = take(c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, depth, rows));
  const auto table_desc = take(wd.transposed());
  const auto ad = dense(c::DataType::kFloat32, {tokens, depth});
  const auto idd = dense(c::DataType::kInt32, {selected});
  const auto yd = dense(c::DataType::kFloat32, {tokens, rows});
  auto context = take(a::Context::create());
  auto memory = take(a::MemoryResource::create(context));
  auto builder = take(g::GraphBuilder::create());
  const auto w = take(builder.constant(wd)), packed_table = take(builder.constant(table_desc));
  const auto x = take(builder.input(ad)), ids = take(builder.input(idd));
  const auto gather = take(builder.add_node(c::Operation(c::Gather{}), std::array{packed_table, ids}));
  const auto chosen = take(builder.result(gather));
  const auto output = take(builder.add(c::MatMul{}, x, w));
  check(builder.publish(chosen));
  check(builder.publish(output));
  auto host_weight = take(hyle::mem::Buffer::create(hyle::mem::system_resource(), weight.bytes.size(), 4));
  check(host_weight.view().write(std::as_bytes(std::span(weight.bytes))));
  const std::array constants{
      g::ConstantBinding{w, take(c::ConstTensorView::create(wd, host_weight.view())), g::ConstantPolicy::kMaterialize},
      g::ConstantBinding{packed_table, take(c::ConstTensorView::create(table_desc, host_weight.view())),
          g::ConstantPolicy::kMaterialize}};
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{take(g::apple::execution_source(context)), g::StorageDomainId{0}}};
  auto compiled = take(g::compile(take(std::move(builder).finish()), {domains, targets}, {.constants = constants}));
  require(compiled.executable.region_count() > 0 && compiled.executable.required_queue_count() == 0,
      "common Graph must use the original Apple managed region without a common Queue");
  check(host_weight.reset()); // Actual materialization must retain its own bytes.
  auto first = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  auto second = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  g::GraphOutput old;
  std::vector<float> old_values;
  std::vector<c::ReadLease> retained_inputs;
  c::ReadLease exported;
  DotReference exported_expected;
  const auto lane = unit_lane(weight);
  const float witness = 1.0f + std::ldexp(1.0f, -20);
  for (int pass = 0; pass < 3; ++pass) {
    std::vector<float> input(tokens * depth, 0);
    for (int t = 0; t < tokens; ++t) {
      if (pass == 0) input[static_cast<std::size_t>(t * depth + lane)] = witness;
      else for (int k = 0; k < depth; ++k)
        input[static_cast<std::size_t>(t * depth + k)] = float((t * 3 + k + pass) % 11 - 5) / 16;
    }
    std::vector<std::int32_t> indices(selected);
    for (int i = 0; i < selected; ++i) indices[static_cast<std::size_t>(i)] = (i + pass) % rows;
    const auto expected = reference(weight, input, {}, tokens, 1, false);
    std::vector<std::uint32_t> selected_bits;
    for (const auto index : indices) for (int k = 0; k < depth; ++k)
      selected_bits.push_back(std::bit_cast<std::uint32_t>(weight.decoded[static_cast<std::size_t>(index * depth + k)]));
    auto activation = common_input(memory, ad, std::span<const float>(input));
    auto selection = common_input(memory, idd, std::span<const std::int32_t>(indices));
    auto &call = pass == 0 ? first : second;
    if (pass == 2) check(call.reset());
    check(call.bind(x, ad, activation));
    check(call.bind(ids, idd, selection));
    check(call.prepare_replay({}));
    auto work = take(call.submit());
    check(work.wait());
    work = {}; // Only GraphOutput, not a retained Completion, gates reset below.
    auto selected_result = take(call.result(chosen));
    const auto selected_values = read_tensor(take(selected_result.tensor()));
    require(selected_values.size() == selected_bits.size(), "common packed Gather shape");
    for (std::size_t i = 0; i < selected_values.size(); ++i)
      require(std::bit_cast<std::uint32_t>(selected_values[i]) == selected_bits[i], "common packed Gather decode bits");
    selected_result.reset();
    auto result = take(call.result(output));
    auto values = read_tensor(take(result.tensor()));
    expect_dot_values(values, expected, depth);
    if (pass == 0) for (int t = 0; t < tokens; ++t)
      require(std::bit_cast<std::uint32_t>(values[static_cast<std::size_t>(t * rows)]) ==
          std::bit_cast<std::uint32_t>(witness), "common F32 MatMul activation was narrowed");
    require(!call.reset().ok(), "GraphOutput must block original Invocation reset");
    require(!call.bind(x, ad, activation).ok(), "GraphOutput must block rebinding");
    auto independent = take(result.storage_lease());
    expect_dot_values(read_tensor(take(independent.tensor(yd))), expected, depth);
    if (pass == 0) { old = std::move(result); old_values = values; }
    else require(read_tensor(take(old.tensor())) == old_values, "other slot changed old GraphOutput");
    if (pass == 2) { exported = std::move(independent); exported_expected = expected; }
    retained_inputs.push_back(std::move(activation));
    retained_inputs.push_back(std::move(selection));
  }
  old.reset();
  check(first.reset()); // Release of the real result observation reopens this slot.
  first = {}; second = {}; compiled.executable = {};
  check(context.close());
  expect_dot_values(read_tensor(take(exported.tensor(yd))), exported_expected, depth);
  std::puts("PASS common Graph: packed Gather/packed MM, two slots/rebind, GraphOutput gate/exported owner");
}

int main() {
  try {
    for (const bool graph : {false, true}) {
      project_domain(graph, 1, -1);   // MV, partial N=5.
      project_domain(graph, 32, -1);  // MM, F32 activation witness.
      project_domain(graph, 31, 0);   // MVID, AS and repeated/distinct experts.
      project_domain(graph, 32, 1);   // MMID, dense A1.
      project_domain(graph, 32, 2);   // MMID, real slot-zero View.
      gather_domain(graph);
    }
    common_graph_domain();
    std::puts("PASS 12 native Direct/Graph domains and one ordinary shared Graph journey");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
