#include <hyle/compute/apple.h>
#include <hyle/compute/model_ops.h>
#include <hyle/graph/apple.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace mx = mlx::core;

template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}));
}
mx::array ready(mx::array array) {
  mx::async_eval({array});
  mx::synchronize(mx::default_stream(mx::Device::gpu));
  array.wait();
  return array;
}
a::Program prepare(a::Context &context, bool graph, c::Operation operation,
    std::span<const c::TensorDesc> descriptions) {
  if (!graph) {
    std::vector<a::InputSpec> specs;
    for (const auto &description : descriptions) specs.push_back(take(a::input_spec(description)));
    return take(context.prepare(operation, specs));
  }
  auto builder = take(g::GraphBuilder::create());
  std::vector<g::ValueId> inputs;
  for (const auto &description : descriptions) inputs.push_back(take(builder.input(description)));
  auto node = take(builder.add_node(operation, inputs));
  check(builder.publish(take(builder.result(node, 0))));
  auto dag = take(std::move(builder).finish());
  return take(g::apple::compile(context, dag));
}
mx::array run(a::Program &program, std::span<const mx::array> inputs) {
  auto invocation = take(program.create_invocation());
  check(invocation.bind(inputs));
  auto work = take(invocation.submit());
  auto result = take(work.wait());
  if (result.get()->arrays().size() != 1) throw std::runtime_error("output count differs");
  return result.get()->arrays()[0];
}
void expect(const mx::array &array, const std::vector<float> &expected) {
  auto contiguous = ready(mx::contiguous(mx::astype(array, mx::float32,
      mx::default_stream(mx::Device::gpu)), false, mx::default_stream(mx::Device::gpu)));
  if (contiguous.size() != expected.size()) throw std::runtime_error("output extent differs");
  const auto *values = contiguous.data<float>();
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (values[i] != expected[i])
      throw std::runtime_error("exact source value differs at " + std::to_string(i));
}

void codecs(bool graph) {
  auto context = take(a::Context::create());
  // Real Q4_K blocks: original half scale, eight 6-bit scales and alternating
  // low/high nibble lanes. Each scale is exact, with a nonzero result.
  std::vector<std::uint8_t> packed(8 * 144);
  for (int row = 0; row < 8; ++row) {
    const auto half = static_cast<std::uint16_t>(0x3c00 + row * 0x400);
    packed[row * 144] = static_cast<std::uint8_t>(half);
    packed[row * 144 + 1] = static_cast<std::uint8_t>(half >> 8);
    std::fill_n(packed.begin() + row * 144 + 4, 12, std::uint8_t{1});
    std::fill_n(packed.begin() + row * 144 + 16, 128, std::uint8_t{0x21});
  }
  // Half scales 1,2,4,8,16,32,64,128 are represented by their exact bits.
  const auto weight = take(c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
      c::DataType::kFloat32, 256, 8));
  const auto carrier_spec = take(a::input_spec(weight));
  auto carrier = mx::array(packed.data(), carrier_spec.shape, mx::uint8);
  const std::array linear_desc{dense(c::DataType::kFloat32, {1, 256}), weight};
  auto linear = prepare(context, graph, c::Operation(c::MatMul{}), linear_desc);
  std::vector<float> activation(256, 1);
  std::vector<float> expected;
  for (int row = 0; row < 8; ++row) expected.push_back(384.0F * std::ldexp(1.0F, row));
  auto result = run(linear, std::array{mx::array(activation.data(), {1, 256}), carrier});
  expect(result, expected);
  std::fill(activation.begin(), activation.end(), 2);
  for (auto &value : expected) value *= 2;
  expect(run(linear, std::array{mx::array(activation.data(), {1, 256}), carrier}), expected);

  const std::array row_desc{take(weight.transposed()), dense(c::DataType::kInt32, {2})};
  auto gather = prepare(context, graph, c::Operation(c::Gather{}), row_desc);
  const std::array<std::int32_t, 2> ids{7, 0};
  expected.clear();
  for (int row : ids) for (int col = 0; col < 256; ++col)
    expected.push_back(std::ldexp(float(1 + (col / 32) % 2), row));
  auto retained = run(gather, std::array{carrier, mx::array(ids.data(), {2})});
  expect(retained, expected);
  const std::array<std::int32_t, 2> invalid{0, 8};
  auto invocation = take(gather.create_invocation());
  check(invocation.bind(std::array{carrier, mx::array(invalid.data(), {2})}));
  auto failed = take(invocation.submit()).wait();
  if (failed || failed.error().code() != hyle::StatusCode::OutOfRange)
    throw std::runtime_error("checked row failure was not propagated");
  check(context.close());
  expect(retained, expected);
}

void artifact(bool graph) {
  auto context = take(a::Context::create());
  const std::array descriptions{dense(c::DataType::kUInt8, {2, 32}),
      dense(c::DataType::kFloat32, {2}), dense(c::DataType::kFloat32, {16})};
  const auto operation = c::model_ops::artifact_weight_decode(
      c::model_ops::WeightArtifactFormat::kNf4, 64, c::DataType::kFloat32);
  auto decode = prepare(context, graph, operation, descriptions);
  std::vector<std::uint8_t> codes(64, 0x0f);
  const std::array<float, 2> scales{2, 4};
  const std::array<float, 16> book{-1, -0.6961928009986877F, -0.5250730514526367F,
      -0.39491748809814453F, -0.28444138169288635F, -0.18477343022823334F,
      -0.09105003625154495F, 0, 0.07958029955625534F, 0.16093020141124725F,
      0.24611230194568634F, 0.33791524171829224F, 0.44070982933044434F,
      0.5626170039176941F, 0.7229568362236023F, 1};
  auto decoded = run(decode, std::array{mx::array(codes.data(), {2, 32}),
      mx::array(scales.data(), {2}), mx::array(book.data(), {16})});
  std::vector<float> expected;
  for (float scale : scales) for (int col = 0; col < 64; ++col)
    expected.push_back(col % 2 ? scale : -scale);
  expect(decoded, expected);
  check(context.close());
  expect(decoded, expected);
}

int main() {
  try {
    for (bool graph : {false, true}) { codecs(graph); artifact(graph); }
    std::puts("PASS ordinary Direct/Graph Q4_K native matmul/replay, exact rows/failure/leases and mature NF4");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
