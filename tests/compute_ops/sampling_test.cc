#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstdint>
#include <span>

#include "../compute_models/sampling_fixture.h"
#include "../../third_party/random123_counter/philox.h"

namespace {
namespace c = hyle::compute;
namespace t = sampling_test;

TEST(SamplingSchema, ExplicitCounterShapeArityDenseAndLimits) {
  const auto identity = t::dense(c::DataType::kUInt32, {13,4});
  for (const std::uint32_t count : {0U,1U,3U,4U,17U,std::uint32_t(INT32_MAX)}) {
    const auto op = c::sampling_ops::random_bits(count);
    const auto out = c::infer_single_output(op, std::array{identity});
    ASSERT_TRUE(out) << out.error().message();
    EXPECT_EQ(*out, t::dense(c::DataType::kUInt32, {13,count}));
    EXPECT_EQ(c::operation_name(op), "counter-random-bits");
    EXPECT_EQ(c::operation_traits(op).effect, c::EffectKind::kPure);
    EXPECT_EQ(c::operation_traits(op).alias, c::AliasPolicy::kNone);
  }
  EXPECT_TRUE(c::infer_single_output(c::sampling_ops::random_bits(),
      std::array{t::dense(c::DataType::kUInt32, {0,4})}));
  const auto sentinel = t::dense(c::DataType::kInt32, {7});
  for (const auto invalid : {c::TensorDesc{},
       t::dense(c::DataType::kInt32,{13,4}),
       t::dense(c::DataType::kUInt32,{52}),
       t::dense(c::DataType::kUInt32,{13,3}),
       c::TensorDesc::strided_bytes(c::DataType::kUInt32,
           std::array<std::int64_t,2>{13,4}, std::array<std::int64_t,2>{32,4}).value()}) {
    std::array out{sentinel};
    EXPECT_TRUE(c::infer_output_descriptions(c::sampling_ops::random_bits(),
        std::array{invalid}, out).IsInvalidArgument());
    EXPECT_EQ(out[0], sentinel);
  }
  std::array out{sentinel};
  EXPECT_TRUE(c::infer_output_descriptions(c::sampling_ops::random_bits(UINT32_MAX),
      std::array{identity}, out).IsOutOfRange());
  EXPECT_EQ(out[0], sentinel);
  EXPECT_FALSE(c::infer_single_output(c::sampling_ops::random_bits(),
      std::span<const c::TensorDesc>{}));
}

TEST(SamplingSchema, FiniteCategoricalDtypesOptionalPortEmptyAndInvalidLayouts) {
  for (const auto type : {c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32})
    for (const std::int64_t rows : {0,1,16}) for (const bool scored : {false,true}) {
      const auto logits = t::dense(type,{rows,248320});
      const auto keys = t::dense(c::DataType::kUInt32,{rows,4});
      const auto op = c::sampling_ops::categorical(scored);
      const auto result = t::outputs(op, std::array{logits,keys});
      ASSERT_EQ(result.size(), scored ? 2U : 1U);
      EXPECT_EQ(result[0],t::dense(c::DataType::kInt32,{rows}));
      if (scored) EXPECT_EQ(result[1],t::dense(c::DataType::kFloat32,{rows}));
      EXPECT_EQ(c::operation_traits(op).alias,c::AliasPolicy::kNone);
    }
  const auto keys = t::dense(c::DataType::kUInt32,{3,4});
  for (const auto invalid : {c::TensorDesc{},t::dense(c::DataType::kInt32,{3,7}),
       t::dense(c::DataType::kFloat32,{3,0}),t::dense(c::DataType::kFloat32,{2,7}),
       t::dense(c::DataType::kFloat32,{21}),
       c::TensorDesc::strided_bytes(c::DataType::kFloat32,
           std::array<std::int64_t,2>{3,7},std::array<std::int64_t,2>{32,4}).value()}) {
    std::array out{t::dense(c::DataType::kInt32,{3}),t::dense(c::DataType::kFloat32,{3})};
    const auto old = out;
    EXPECT_TRUE(c::infer_output_descriptions(c::sampling_ops::categorical(true),
        std::array{invalid,keys},out).IsInvalidArgument());
    EXPECT_EQ(out,old);
  }
  std::array<c::TensorDesc,2> result;
  EXPECT_TRUE(c::infer_output_descriptions(c::sampling_ops::categorical(true),
      std::array{t::dense(c::DataType::kFloat32,{3,std::int64_t(INT32_MAX)+1}),keys},
      result).IsOutOfRange());
}

TEST(SamplingCounter, ThreeOfficialRandom123TenRoundKnownAnswers) {
  struct Case { t::Words counter; std::array<std::uint32_t,2> key; t::Words expected; };
  for (const auto &item : std::array{
      Case{{0,0,0,0},{0,0},{0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8}},
      Case{{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX},{UINT32_MAX,UINT32_MAX},
            {0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd}},
      Case{{0x243f6a88,0x85a308d3,0x13198a2e,0x03707344},{0xa4093822,0x299f31d0},
            {0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1}}}) {
    EXPECT_EQ(t::philox(item.counter,item.key),item.expected);
    const auto actual = hyle::borrowed::random123_counter::philox(
        {item.counter[0],item.counter[1],item.counter[2],item.counter[3]},item.key[0],item.key[1]);
    EXPECT_EQ((t::Words{actual.x0,actual.x1,actual.x2,actual.x3}),item.expected);
  }
}
} // namespace
