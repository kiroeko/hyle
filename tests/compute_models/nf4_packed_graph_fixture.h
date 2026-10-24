#pragma once

#include "ct_packed_graph_fixture.h"

namespace nf4_packed_graph_test {
namespace f = ct_packed_graph_test;
namespace c = f::c;
namespace g = f::g;
namespace m = f::m;
using Type = f::Type;

template<class Value> std::vector<std::byte> bytes(const std::vector<Value> &values) {
  const auto source = std::as_bytes(std::span(values));
  return {source.begin(), source.end()};
}

// Exercise the seven original operands through the existing Graph backend
// fixture. Independent scalar source math and the test-only RNE encoder do
// not call the packed reader or the public ArtifactWeightDecode implementation.
template<class Backend> void SourceHalfRoundsAndMutableNestedOffset() {
  constexpr std::int64_t rows = 2, n = 9, k = 2048, group = 64;
  constexpr std::size_t groups = n * k / group;
  std::vector<std::uint8_t> codes(n * k / 2), absmax(groups);
  for (std::size_t i = 0; i < codes.size(); ++i)
    codes[i] = std::uint8_t((((i * 7 + 3) % 16) << 4) | ((i * 11 + 1) % 16));
  for (std::size_t i = 0; i < groups; ++i) absmax[i] = std::uint8_t((i * 47 + 29) % 256);
  const std::vector<float> nested_scale{.015625F, .046875F};
  std::vector<float> nested_map(256);
  for (std::size_t i = 0; i < nested_map.size(); ++i) nested_map[i] = -.79F + .0061F * float(i);
  const std::vector<float> codebook{-1.F,-.6961928F,-.52507305F,-.3949175F,
      -.28444138F,-.18477343F,-.09105004F,0.F,.0795803F,.1609302F,
      .2461123F,.33791524F,.44070983F,.562617F,.72295684F,1.F};
  std::vector<double> activation(rows * k, 0);
  activation[0] = .25; activation[1] = -.125; activation[k - 1] = .5;
  activation[k] = -.5; activation[k + 1] = .125; activation[2 * k - 1] = .25;
  std::array<std::vector<std::byte>, 7> payloads{
      f::floating(Type::kFloat16, activation), bytes(codes), bytes(absmax),
      bytes(nested_scale), bytes(nested_map), bytes(std::vector<float>{.00312345F}), bytes(codebook)};
  const std::array descriptions{f::dense(Type::kFloat16, {rows,k}),
      f::dense(Type::kUInt8,{n,k/2}), f::dense(Type::kUInt8,{groups}),
      f::dense(Type::kFloat32,{2}), f::dense(Type::kFloat32,{256}),
      f::dense(Type::kFloat32,{}), f::dense(Type::kFloat32,{16})};
  auto builder = f::take(g::GraphBuilder::create(), "NF4 GraphBuilder");
  std::array<g::ValueId, 7> ports;
  for (std::size_t i = 0; i < ports.size(); ++i)
    ports[i] = f::take(builder.input(descriptions[i]), "NF4 input");
  const auto decode = f::take(builder.add_node(c::model_ops::artifact_weight_decode(
      c::model_ops::WeightArtifactFormat::kNf4DoubleQuant, group, Type::kFloat16),
      std::span(ports).subspan(1)), "NF4 Decode");
  const auto decoded = f::take(builder.result(decode), "NF4 decoded result");
  const auto transposed = f::take(builder.add(c::View::permute(
      std::array<std::int32_t,2>{1,0}).value(), decoded), "NF4 View");
  const auto output = f::take(builder.add(c::MatMul{.accumulation_type=Type::kFloat32},
      ports[0], transposed), "NF4 MatMul");
  f::check(builder.publish(output), "NF4 output publication");
  auto graph = f::take(std::move(builder).finish(), "NF4 graph");
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  auto queue = f::take(context.create_queue(), "NF4 queue");
  auto compiled = f::take(Backend::compile(graph, context, owner), "NF4 packed compile");
  ASSERT_EQ(compiled.report.step_count(), 1U);
  const auto step = f::take(compiled.report.step(0), "NF4 compiled step");
  EXPECT_EQ(step.policy, g::kArtifactPackedProjectionPolicy);
  EXPECT_EQ(step.provider.id, Backend::provider_id);
  EXPECT_EQ(step.node_count, 3U);
  for (std::size_t i = 0; i < compiled.report.physical_value_count(); ++i) {
    const auto value = f::take(compiled.report.physical_value(i), "NF4 physical value");
    EXPECT_NE(value.value, decoded);
    EXPECT_NE(value.value, transposed);
  }
  std::array<m::Buffer, 7> backing;
  for (std::size_t i = 0; i < backing.size(); ++i) {
    backing[i] = f::take(m::Buffer::create(owner.ref(), payloads[i].size(), 256), "NF4 input backing");
    f::check(backing[i].view().write(payloads[i]), "NF4 source upload");
  }
  auto destination = f::take(m::Buffer::create(owner.ref(), rows*n*2, 256), "NF4 output backing");
  auto slot = f::invocation(compiled.executable); // All caller buffers outlive the Invocation.
  for (std::size_t i = 0; i < ports.size(); ++i)
    f::check(slot.bind(ports[i], f::take(c::ConstTensorView::create(
        descriptions[i], backing[i].view()), "NF4 source view")), "NF4 source binding");
  f::check(slot.bind(output, f::take(c::TensorView::create(f::dense(Type::kFloat16,{rows,n}),
      destination.view()), "NF4 destination view")), "NF4 destination binding");
  std::size_t changed = 0, bfloat_witnesses = 0;
  std::vector<double> previous;
  for (float offset : {.00312345F, .017199F}) {
    f::check(backing[5].view().write(std::as_bytes(std::span(&offset,1))), "NF4 offset update");
    f::submit<Backend>(slot, queue, false);
    const auto actual = f::read(f::output(slot, output));
    ASSERT_EQ(actual.size(), std::size_t(rows*n));
    const auto weight = [&](std::size_t column, std::size_t depth, Type decoded_type) {
      const auto original = column * k + depth;
      const auto g0 = original / group;
      const auto product = f::t::round(Type::kFloat32,
          double(nested_map[absmax[g0]]) * double(nested_scale[g0 / 256]));
      const auto scale = f::t::round(Type::kFloat32, product + double(offset));
      const auto code = (codes[original / 2] >> (original % 2 ? 0 : 4)) & 15U;
      return f::t::round(decoded_type, f::t::round(Type::kFloat32, double(codebook[code]) * scale));
    };
    for (std::size_t row = 0; row < std::size_t(rows); ++row)
      for (std::size_t column = 0; column < std::size_t(n); ++column) {
        const auto reference = [&](Type decoded_type) {
          const auto first = f::t::round(Type::kFloat32, activation[row*k] * weight(column,0,decoded_type) +
              activation[row*k+1] * weight(column,1,decoded_type));
          return f::t::round(Type::kFloat16, f::t::round(Type::kFloat32,
              first + activation[row*k+k-1] * weight(column,k-1,decoded_type)));
        };
        const auto index = row*n+column;
        EXPECT_EQ(actual[index], reference(Type::kFloat16)) << index;
        bfloat_witnesses += actual[index] != reference(Type::kBFloat16);
        if (!previous.empty()) changed += actual[index] != previous[index];
      }
    previous = actual;
  }
  EXPECT_GT(changed, 0U);
  EXPECT_GT(bfloat_witnesses, 0U);
}
} // namespace nf4_packed_graph_test
