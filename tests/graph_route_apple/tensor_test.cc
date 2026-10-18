#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "../compute_route_apple/tensor_test_support.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;

TEST(AppleGraph, BFloatCastBoundariesSurviveFusionAndChangedStridedInputs) {
  auto builder = g::GraphBuilder::create().value();
  const auto desc = c::TensorDesc::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{2, 17}).value();
  const auto x = builder.input(desc).value();
  const auto low = builder.add(c::Cast{c::DataType::kBFloat16}, x).value();
  const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, low).value();
  ASSERT_TRUE(builder.publish(low).ok());
  ASSERT_TRUE(builder.publish(wide).ok());
  auto context = a::Context::create().value();
  auto graph = std::move(builder).finish().value();
  auto program = ga::compile(context, graph, {});
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation().value();
  const std::array<std::uint32_t, 17> seeds{
      0, 0x80000000U, 1, 0x8000U, 0x8001U, 0x10000U, 0x10001U,
      0x7fffffU, 0x800000U, 0x807fffffU, 0x80010000U,
      0x3f808000U, 0x3f818000U, 0x7f7fffffU, 0x7f800000U,
      0xff800000U, 0x7fc00001U};
  std::vector<mx::array> retained;
  std::vector<std::vector<std::uint64_t>> retained_bits;
  for (std::size_t phase = 0; phase < 3; ++phase) {
    std::array<std::uint32_t, 34> payload{};
    for (std::size_t i = 0; i < payload.size(); ++i)
      payload[i] = seeds[(i + phase) % seeds.size()];
    const auto input = ready(mx::transpose(
        mx::view(mx::array(payload.data(), {17, 2}, mx::uint32),
                 mx::float32, stream()), stream()));
    const auto before = bits(input);
    const auto source = values(input);
    ASSERT_TRUE(slot.bind(std::array{input}).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    for (const auto &output : result->get()->arrays()) {
      const auto observed = values(output);
      for (std::size_t i = 0; i < source.size(); ++i) {
        const auto expected = rounded(source[i], mx::bfloat16);
        if (std::isnan(expected))
          EXPECT_TRUE(std::isnan(observed[i])) << i;
        else {
          EXPECT_EQ(observed[i], expected) << i;
          if (expected == 0)
            EXPECT_EQ(std::signbit(observed[i]), std::signbit(expected)) << i;
        }
      }
      retained.push_back(output);
      retained_bits.push_back(bits(output));
    }
    EXPECT_EQ(bits(input), before);
  }
  *program = {};
  ASSERT_TRUE(context.close().ok());
  for (std::size_t i = 0; i < retained.size(); ++i)
    EXPECT_EQ(bits(retained[i]), retained_bits[i]);
}

TEST(AppleGraph, HalfNormGateResidualAndViewsRetainIndependentNativeResults) {
  for (auto type : {mx::float16, mx::bfloat16}) {
    for (int width : {129, 2048}) {
      for (int rows : {0, 1, 17, 23, 129}) {
        SCOPED_TRACE(::testing::Message() << static_cast<int>(logical(type))
                                          << '/' << rows << '/' << width);
        auto context = a::Context::create().value();
        auto builder = g::GraphBuilder::create().value();
        const auto desc =
            c::TensorDesc::dense(logical(type),
                                 std::array<std::int64_t, 2>{rows, width})
                .value();
        const auto mask_desc =
            c::TensorDesc::dense(c::DataType::kBool,
                                 std::array<std::int64_t, 2>{rows, 1})
                .value();
        const auto parameter =
            c::TensorDesc::dense(c::DataType::kFloat32,
                                 std::array<std::int64_t, 1>{width})
                .value();
        const auto x = builder.input(desc).value();
        const auto mask = builder.input(mask_desc).value();
        const auto weight = builder.constant(parameter).value();
        const auto norm = builder.add(c::RmsNorm{}, x, weight).value();
        const auto gate = builder
                              .add(c::Operation(c::GatedActivation{
                                       .inputs = c::GatedInputKind::kSeparate}),
                                   norm, x)
                              .value();
        const auto residual = builder.add(c::Binary{}, gate, x).value();
        const c::Fill zero{c::TensorShape::scalar(), logical(type),
                           c::ScalarValue::from_bits(logical(type), 0)};
        const auto cleared = builder.add(zero).value();
        const auto output =
            builder.add(c::Select{}, mask, residual, cleared).value();
        std::array<g::ValueId, 5> pieces;
        const std::array sizes{1, 32, 31, 17, width - 81};
        int begin = 0;
        for (std::size_t i = 0; i < pieces.size(); ++i) {
          pieces[i] = builder.add(c::View::slice(1, begin, sizes[i]).value(), x)
                          .value();
          begin += sizes[i];
        }
        const auto joined_node =
            builder.add_node(c::Operation(c::Concat{.axis = -1}), pieces)
                .value();
        const auto joined = builder.result(joined_node).value();
        ASSERT_TRUE(builder.publish(output).ok());
        ASSERT_TRUE(builder.publish(joined).ok());
        auto graph = std::move(builder).finish().value();
        auto weights = pattern(width, 1);
        for (auto &value : weights)
          value = 1.0 + value * 0.03;
        const auto native_weight = make(weights, {width}, mx::float32);
        const auto learned = values(native_weight);
        ga::CompileReport report;
        auto program = ga::compile(context, graph, std::array{native_weight},
                                   {.report = &report});
        ASSERT_TRUE(program) << program.error().message();
        EXPECT_TRUE(report.status.ok());
        EXPECT_EQ(report.inputs[0].data_type(), logical(type));
        EXPECT_EQ(report.outputs[0].data_type(), logical(type));
        EXPECT_FALSE(report.private_storage_bytes);
        graph = {};
        auto direct = context.compile(
            [native_weight, zero](const auto &inputs, mx::Stream stream)
                -> hyle::Result<std::vector<mx::array>> {
              HYLE_ASSIGN_OR_RETURN(
                  auto n,
                  a::lower(c::Operation(c::RmsNorm{}),
                           std::array{inputs[0], native_weight}, stream));
              HYLE_ASSIGN_OR_RETURN(
                  auto v, a::lower(c::Operation(c::GatedActivation{
                                       .inputs = c::GatedInputKind::kSeparate}),
                                   std::array{n[0], inputs[0]}, stream));
              HYLE_ASSIGN_OR_RETURN(
                  auto r, a::lower(c::Operation(c::Binary{}),
                                   std::array{v[0], inputs[0]}, stream));
              HYLE_ASSIGN_OR_RETURN(
                  auto z, a::lower(c::Operation(zero),
                                   std::span<const mx::array>{}, stream));
              return a::lower(c::Operation(c::Select{}),
                              std::array{inputs[1], r[0], z[0]}, stream);
            },
            std::array{a::InputSpec{{rows, width}, type},
                       a::InputSpec{{rows, 1}, mx::bool_}});
        ASSERT_TRUE(direct) << direct.error().message();
        auto first = program->create_invocation().value();
        auto second = program->create_invocation().value();
        auto direct_slot = direct->create_invocation().value();
        const auto arguments = [&](int phase) {
          const auto input =
              make(pattern(static_cast<std::size_t>(rows * width), phase),
                   {rows, width}, type);
          std::vector<std::uint8_t> flags(rows);
          for (int row = 0; row < rows; ++row)
            flags[row] = (row + phase) % 3 != 1;
          const auto mask_value =
              mx::array(flags.data(), mx::Shape{rows, 1}, mx::bool_);
          return std::array{input, mask_value};
        };
        const auto initial = arguments(0);
        const auto input_before = bits(initial[0]);
        ASSERT_TRUE(first.bind(initial).ok());
        auto first_work = first.submit().value();
        auto old = first_work.wait().value();
        const auto retained = bits(old.get()->arrays()[0]);
        EXPECT_EQ(bits(initial[0]), input_before);
        for (int phase = 1; phase <= 3; ++phase) {
          const auto input = arguments(phase);
          ASSERT_TRUE(second.bind(input).ok());
          auto work = second.submit().value();
          auto result = work.wait().value();
          const auto data = values(input[0]);
          const auto enabled = values(input[1]);
          std::vector<double> expected(data.size());
          for (int row = 0; row < rows; ++row) {
            double square = 0;
            for (int col = 0; col < width; ++col)
              square += data[row * width + col] * data[row * width + col];
            const auto inverse =
                1 / std::sqrt(square / width + static_cast<double>(1e-5F));
            for (int col = 0; col < width; ++col) {
              const auto index = static_cast<std::size_t>(row * width + col);
              const auto n =
                  rounded(data[index] * inverse * learned[col], type);
              const auto gated = rounded(
                  activation(c::UnaryKind::kSilu, n) * data[index], type);
              expected[index] =
                  enabled[row] ? rounded(gated + data[index], type) : 0;
            }
          }
          expect(result.get()->arrays()[0], expected);
          EXPECT_EQ(bits(result.get()->arrays()[1]), bits(input[0]));
          EXPECT_EQ(bits(old.get()->arrays()[0]), retained);
          ASSERT_TRUE(direct_slot.bind(input).ok());
          auto direct_work = direct_slot.submit().value();
          auto direct_result = direct_work.wait().value();
          EXPECT_EQ(bits(direct_result.get()->arrays()[0]),
                    bits(result.get()->arrays()[0]));
        }
        ASSERT_TRUE(context.close().ok());
        EXPECT_EQ(bits(old.get()->arrays()[0]), retained);
        EXPECT_FALSE(second.submit());
      }
    }
  }
}
} // namespace

#include "../compute_route_apple/sink_attention_test_support.h"

namespace {
using namespace apple_tensor_test;
namespace sink_graph=hyle::graph;
namespace sink_graph_apple=sink_graph::apple;

TEST(AppleSinkGraph, PublishedCopySinkAndNegativeCrossPrivateBoundary) {
  constexpr int queries=1025,keys=2049,heads=64;
  auto inputs=sink_ports(queries,keys,heads,0);
  auto reference=sink_reference(queries,keys,heads,0);
  auto negative=reference;
  for(auto &row:negative){row.weighted=-row.weighted;row.mass=-row.mass;}
  auto builder=sink_graph::GraphBuilder::create().value();
  std::array<sink_graph::ValueId,5> ports;
  for(std::size_t i=0;i<ports.size();++i) {
    std::vector<std::int64_t> shape(inputs[i].shape().begin(),inputs[i].shape().end());
    auto desc=c::TensorDesc::dense(c::DataType::kFloat32,shape).value();
    ports[i]=builder.input(desc).value();
  }
  auto copied=builder.add(c::Copy{},ports[0]).value();
  auto sink_ports_ids=ports;sink_ports_ids[0]=copied;
  auto sink=builder.add(c::model_ops::sink_attention(kSinkScale),
      sink_ports_ids[0],sink_ports_ids[1],sink_ports_ids[2],sink_ports_ids[3],sink_ports_ids[4]).value();
  auto neg=builder.add(c::Unary{c::UnaryKind::kNeg},sink).value();
  ASSERT_TRUE(builder.publish(copied).ok());
  ASSERT_TRUE(builder.publish(sink).ok());
  ASSERT_TRUE(builder.publish(neg).ok());
  auto graph=std::move(builder).finish().value();
  auto context=a::Context::create().value();
  auto compiled=sink_graph_apple::compile(context,graph,{});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto slot=compiled->create_invocation().value();
  ASSERT_TRUE(slot.bind(inputs).ok());
  const auto original_hash=sink_output_hash(inputs[0]);
  auto accepted=slot.submit();
  ASSERT_TRUE(accepted) << accepted.error().message();
  auto finished=accepted->wait();
  ASSERT_TRUE(finished) << finished.error().message();
  const auto outputs=finished->get()->arrays();
  ASSERT_EQ(outputs.size(),3U);
  EXPECT_EQ(outputs[0].shape(),inputs[0].shape());
  EXPECT_EQ(sink_output_hash(outputs[0]),original_hash);
  check_sink(outputs[1],reference);
  check_sink(outputs[2],negative);
}
}

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;


TEST(AppleGraph, PublishedCenteredShortQueryMathAndNegativeCrossCheckedBoundary) {
  namespace sq = ordinary_short_query_centered;
  constexpr auto layout = c::AttentionLayout::kBatchSequenceHeadDepth;
  auto inputs = sq::ports(layout, 0);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> ports;
  for (std::size_t i = 0; i < ports.size(); ++i) {
    std::vector<std::int64_t> shape(inputs[i].shape().begin(), inputs[i].shape().end());
    ports[i] = builder.input(c::TensorDesc::dense(
        i == 3 ? c::DataType::kBool : c::DataType::kFloat32, shape).value()).value();
  }
  const auto copy = builder.add(c::Copy{}, ports[0]).value();
  const auto attention = builder.add(c::ScaledDotProductAttention{
      .layout = layout, .mask = c::MaskKind::kBoolean, .scale = sq::scale},
      copy, ports[1], ports[2], ports[3]).value();
  const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, attention).value();
  ASSERT_TRUE(builder.publish(copy).ok());
  ASSERT_TRUE(builder.publish(attention).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  auto context = a::Context::create().value();
  auto program = ga::compile(context, graph, {}).value();
  auto slot = program.create_invocation().value();
  std::vector<mx::array> old_outputs;
  std::vector<std::vector<std::uint64_t>> old_bits;
  for (int phase = 0; phase < 2; ++phase) {
    inputs = sq::ports(layout, phase);
    auto expected = sq::reference(layout, phase), negated = expected;
    for (auto &value : negated) value = -value;
    ASSERT_TRUE(slot.bind(inputs).ok());
    const auto query_bits = bits(inputs[0]);
    auto accepted = slot.submit().value();
    auto finished = accepted.wait().value();
    const auto outputs = finished.get()->arrays();
    ASSERT_EQ(outputs.size(), 3U);
    EXPECT_EQ(outputs[0].shape(), inputs[0].shape());
    EXPECT_EQ(bits(outputs[0]), query_bits);
    expect(outputs[1], expected);
    expect(outputs[2], negated); // Independent signed FP64 reference, not self-comparison.
    ASSERT_FALSE(::testing::Test::HasFailure());
    if (phase == 0)
      for (const auto &output : outputs) {
        old_outputs.push_back(output);
        old_bits.push_back(bits(output));
      }
  }
  for (std::size_t i = 0; i < old_outputs.size(); ++i)
    EXPECT_EQ(bits(old_outputs[i]), old_bits[i]);
}

} // namespace
