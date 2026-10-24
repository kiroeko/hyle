#include <gtest/gtest.h>

#include <climits>

#include "tensor_test_support.h"
#include "../compute_models/row_cache_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace rc = row_cache_test;

mx::array source(c::RowCacheEncoding encoding, int phase) {
  const auto raw = rc::input_values(encoding, phase);
  return ready(mx::view(mx::array(raw.data(),
      {static_cast<int>(rc::kRows), static_cast<int>(encoding.depth())}, mx::uint16),
      mx::bfloat16, stream()));
}

mx::array byte_bank(std::span<const std::uint8_t> bytes, int rows, int width) {
  std::vector<std::uint8_t> backing(bytes.size() + 1, 0xa5);
  std::copy(bytes.begin(), bytes.end(), backing.begin() + 1);
  const auto physical = ready(mx::array(backing.data(),
      {static_cast<int>(backing.size())}, mx::uint8));
  const auto bank = ready(mx::reshape(mx::slice(physical, {1},
      {static_cast<int>(backing.size())}, stream()), {rows, width}, stream()));
  EXPECT_EQ(bank.data<char>() - physical.data<char>(), 1);
  return bank;
}

void expect_encoded(const mx::array &actual, const rc::Reference &expected) {
  const auto observed = bits(actual);
  ASSERT_EQ(observed.size(), expected.bytes.size());
  for (std::size_t i = 0; i < observed.size(); ++i)
    ASSERT_EQ(observed[i], expected.bytes[i]) << i;
}

void expect_selected(const mx::array &actual, const rc::Reference &expected,
                     std::span<const std::int32_t> ids, std::size_t depth) {
  const auto observed = bits(actual);
  ASSERT_EQ(observed.size(), ids.size() * depth);
  for (std::size_t i = 0; i < ids.size(); ++i)
    for (std::size_t d = 0; d < depth; ++d)
      ASSERT_EQ(observed[i * depth + d], expected.decoded[ids[i] * depth + d])
          << i << ':' << d;
}

TEST(AppleRowCache, DirectRowsMatchPackedAbiAndSelectedBfloatBits) {
  auto context = a::Context::create().value();
  for (auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                         c::RowCacheEncoding::float4_block32()}) {
    SCOPED_TRACE(encoding.row_bytes());
    const std::array encode_specs{a::InputSpec{
        {static_cast<int>(rc::kRows), static_cast<int>(encoding.depth())}, mx::bfloat16}};
    auto encoder = context.prepare(c::model_ops::row_cache_encode(encoding), encode_specs).value();
    std::vector<mx::array> retained;
    std::vector<std::uint64_t> old_encoded, old_decoded;
    for (int phase : {0, 1}) {
      const auto expected = rc::reference(encoding, rc::input_values(encoding, phase));
      const auto input = source(encoding, phase);
      auto slot = encoder.create_invocation().value();
      ASSERT_TRUE(slot.bind(std::array{input}).ok());
      auto work = slot.submit().value();
      auto encoded = work.wait();
      ASSERT_TRUE(encoded) << encoded.error().message();
      expect_encoded(encoded->get()->arrays()[0], expected);
      const auto bank = byte_bank(expected.bytes, rc::kRows, encoding.row_bytes());
      const std::array<std::int32_t, 6> selected = phase
          ? std::array<std::int32_t, 6>{3, 4, 1, 2, 0, 4}
          : std::array<std::int32_t, 6>{4, 1, 4, 0, 3, 2};
      const auto ids = ready(mx::array(selected.data(), {2, 3}, mx::int32));
      for (auto bounds : {c::IndexBoundsPolicy::kChecked, c::IndexBoundsPolicy::kUnchecked}) {
        auto decoded = execute(context, c::model_ops::row_cache_gather_decode(encoding, bounds),
            std::array{bank, ids});
        ASSERT_TRUE(decoded) << decoded.error().message();
        expect_selected(*decoded, expected, selected, encoding.depth());
        if (phase == 0 && bounds == c::IndexBoundsPolicy::kChecked) {
          retained.push_back(encoded->get()->arrays()[0]);
          retained.push_back(*decoded);
          old_encoded = bits(retained[0]);
          old_decoded = bits(retained[1]);
        }
      }
      EXPECT_EQ(bits(retained[0]), old_encoded);
      EXPECT_EQ(bits(retained[1]), old_decoded);
    }
  }
}

TEST(AppleRowCache, GraphPublishesEncodedAndDecodedValuesAcrossTwoSlots) {
  auto context = a::Context::create().value();
  for (auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                         c::RowCacheEncoding::float4_block32()}) {
    const auto source_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t, 2>{rc::kRows, encoding.depth()}).value();
    const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 2>{2, 3}).value();
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(source_desc).value();
    const auto ids = builder.input(ids_desc).value();
    const auto encoded_node = builder.add_node(c::model_ops::row_cache_encode(encoding),
        std::array{input}).value();
    const auto encoded = builder.result(encoded_node).value();
    const auto decoded_node = builder.add_node(c::model_ops::row_cache_gather_decode(encoding),
        std::array{encoded, ids}).value();
    const auto decoded = builder.result(decoded_node).value();
    ASSERT_TRUE(builder.publish(encoded).ok());
    ASSERT_TRUE(builder.publish(decoded).ok());
    const auto graph = std::move(builder).finish().value();
    auto program = g::apple::compile(context, graph).value();
    auto first = program.create_invocation().value();
    auto second = program.create_invocation().value();
    const std::array<std::int32_t, 6> selected{4, 1, 4, 0, 3, 2};
    const auto positions = ready(mx::array(selected.data(), {2, 3}, mx::int32));
    ASSERT_TRUE(first.bind(std::array{source(encoding, 0), positions}).ok());
    auto work = first.submit().value();
    auto old = work.wait().value();
    const auto old_bytes = bits(old.get()->arrays()[0]);
    const auto old_values = bits(old.get()->arrays()[1]);
    for (int phase : {0, 1}) {
      const auto expected = rc::reference(encoding, rc::input_values(encoding, phase));
      ASSERT_TRUE(second.bind(std::array{source(encoding, phase), positions}).ok());
      {
        auto next_work = second.submit().value();
        auto result = next_work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_EQ(result->get()->arrays().size(), 2U);
        expect_encoded(result->get()->arrays()[0], expected);
        expect_selected(result->get()->arrays()[1], expected, selected, encoding.depth());
      }
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_bytes);
      EXPECT_EQ(bits(old.get()->arrays()[1]), old_values);
    }
  }
}

TEST(AppleRowCache, InvalidIdsAndNonfiniteRowsPreservePublishedResults) {
  auto context = a::Context::create().value();
  for (auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                         c::RowCacheEncoding::float4_block32()}) {
    const auto expected = rc::reference(encoding, rc::input_values(encoding, 0));
    const auto bank = byte_bank(expected.bytes, rc::kRows, encoding.row_bytes());
    const auto good = ready(mx::array(std::int32_t{0}));
    const auto operation = c::model_ops::row_cache_gather_decode(encoding);
    auto program = context.prepare(operation, std::array{bank, good}).value();
    auto retained = program.create_invocation().value();
    ASSERT_TRUE(retained.bind(std::array{bank, good}).ok());
    auto old_work = retained.submit().value();
    auto old = old_work.wait().value();
    const auto old_bits = bits(old.get()->arrays()[0]);
    for (std::int32_t invalid : {-1, 5, INT32_MAX}) {
      auto failed = program.create_invocation().value();
      ASSERT_TRUE(failed.bind(std::array{bank, ready(mx::array(invalid))}).ok());
      auto work = failed.submit().value();
      auto result = work.wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
      EXPECT_FALSE(failed.submit());
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_bits);
    }
    for (std::uint16_t invalid : {std::uint16_t{0x7fc0}, std::uint16_t{0x7f80}, std::uint16_t{0xff80}}) {
      auto raw = rc::input_values(encoding, 0);
      raw.back() = invalid; // The FP8 tail is subject to the same finite check.
      const auto input = ready(mx::view(mx::array(raw.data(),
          {static_cast<int>(rc::kRows), static_cast<int>(encoding.depth())}, mx::uint16),
          mx::bfloat16, stream()));
      auto encoded = execute(context, c::model_ops::row_cache_encode(encoding), std::array{input});
      ASSERT_FALSE(encoded);
      EXPECT_TRUE(encoded.error().IsInvalidArgument());
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_bits);
    }
  }
}

TEST(AppleRowCache, ScalarEmptyAndReservedCodesKeepFormatSemantics) {
  auto context = a::Context::create().value();
  for (auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                         c::RowCacheEncoding::float4_block32()}) {
    const bool four = encoding.format() == c::RowCacheFormat::kFloat4Block32;
    const int channels = four ? 128 : 448, group = four ? 32 : 64;
    std::vector<std::uint8_t> bytes(encoding.row_bytes());
    const std::array<std::uint8_t, 7> scales = four
        ? std::array<std::uint8_t, 7>{0, 1, 254, 255, 0, 0, 0}
        : std::array<std::uint8_t, 7>{0, 1, 127, 254, 255, 0, 127};
    std::vector<std::uint16_t> expected(encoding.depth());
    for (int d = 0; d < channels; ++d) {
      const auto scale = scales[d / group];
      const unsigned magnitude = d % (four ? 8 : 128);
      const unsigned code = magnitude | ((d & 1) ? (four ? 8 : 128) : 0);
      if (four) bytes[d / 2] |= static_cast<std::uint8_t>(code << ((d & 1) * 4));
      else bytes[d] = static_cast<std::uint8_t>(code);
      const auto value = scale == 255 || (!four && magnitude == 127)
          ? std::numeric_limits<float>::quiet_NaN()
          : static_cast<float>(std::ldexp(rc::value(magnitude, four) * ((d & 1) ? -1 : 1), int(scale) - 127));
      expected[d] = rc::bfloat(value);
    }
    for (int i = 0; i < channels / group; ++i) bytes[(four ? 64 : 576) + i] = scales[i];
    if (!four) {
      constexpr std::array<std::uint16_t, 8> tail{0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0x3f80};
      for (int d = 448; d < 512; ++d) {
        expected[d] = tail[d % tail.size()];
        bytes[448 + 2 * (d - 448)] = static_cast<std::uint8_t>(expected[d]);
        bytes[449 + 2 * (d - 448)] = static_cast<std::uint8_t>(expected[d] >> 8);
      }
    }
    const auto bank = byte_bank(bytes, 1, encoding.row_bytes());
    auto result = execute(context, c::model_ops::row_cache_gather_decode(encoding),
        std::array{bank, ready(mx::array(std::int32_t{0}))});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result->shape(), mx::Shape{static_cast<int>(encoding.depth())});
    const auto actual = bits(*result);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t d = 0; d < expected.size(); ++d)
      if (d < static_cast<std::size_t>(channels) && std::isnan(rc::decode(expected[d])))
        EXPECT_TRUE(std::isnan(rc::decode(static_cast<std::uint16_t>(actual[d])))) << d;
      else EXPECT_EQ(actual[d], expected[d]) << d;
    const auto empty_bank = ready(mx::zeros({0, static_cast<int>(encoding.row_bytes())}, mx::uint8, stream()));
    const auto empty_ids = ready(mx::zeros({0, 2}, mx::int32, stream()));
    auto empty = execute(context, c::model_ops::row_cache_gather_decode(encoding),
        std::array{empty_bank, empty_ids});
    ASSERT_TRUE(empty) << empty.error().message();
    EXPECT_EQ(empty->size(), 0U);
    EXPECT_EQ(empty->shape(), (mx::Shape{0, 2, static_cast<int>(encoding.depth())}));
    const auto empty_source = ready(mx::zeros({0, static_cast<int>(encoding.depth())}, mx::bfloat16, stream()));
    auto encoded = execute(context, c::model_ops::row_cache_encode(encoding), std::array{empty_source});
    ASSERT_TRUE(encoded) << encoded.error().message();
    EXPECT_EQ(encoded->shape(), empty_bank.shape());
  }
}
} // namespace
