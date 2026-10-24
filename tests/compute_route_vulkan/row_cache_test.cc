#include <gtest/gtest.h>

#include "tensor_test_support.h"
#include "../compute_models/row_cache_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_tensor_test;
namespace g = hyle::graph;

template <class T>
std::vector<T> read(const v::Context &context, const v::Tensor &tensor) {
  const auto data = bytes(context, tensor);
  std::vector<T> result(data.size() / sizeof(T));
  std::memcpy(result.data(), data.data(), data.size());
  return result;
}

void expect_selected(const v::Context &context, const v::Tensor &tensor,
    std::span<const std::uint16_t> decoded, std::span<const std::int32_t> ids,
    std::size_t depth) {
  const auto actual = read<std::uint16_t>(context, tensor);
  ASSERT_EQ(actual.size(), ids.size() * depth);
  for (std::size_t row = 0; row < ids.size(); ++row)
    for (std::size_t d = 0; d < depth; ++d)
      ASSERT_EQ(actual[row * depth + d], decoded[std::size_t(ids[row]) * depth + d])
          << row << '/' << d;
}

v::Tensor odd_bank(const v::Context &context, const c::TensorDesc &description,
    std::span<const std::uint8_t> payload) {
  std::vector<std::uint8_t> storage(payload.size() + 1, 0xa5);
  std::copy(payload.begin(), payload.end(), storage.begin() + 1);
  const auto root = context.upload(dense(c::DataType::kUInt8,
      {static_cast<std::int64_t>(storage.size())}), std::as_bytes(std::span(storage))).value();
  const auto slice = root.view(c::View::slice(0, 1,
      static_cast<std::int64_t>(payload.size())).value()).value();
  const auto shape = c::TensorShape::create(description.dimensions()).value();
  return slice.view(c::View::reshape(shape).value()).value();
}

TEST(VulkanRowCache, EncodedBytesAndSelectedBfloatBitsReplayThroughPublicGraph) {
  auto context = vulkan_test::context().value();
  for (const auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                              c::RowCacheEncoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    const auto input_desc = dense(c::DataType::kBFloat16,
        {row_cache_test::kRows, encoding.depth()});
    const auto ids_desc = dense(c::DataType::kInt32, {2, 3});
    auto builder = g::GraphBuilder::create().value();
    const auto input_id = builder.input(input_desc).value();
    const auto ids_id = builder.input(ids_desc).value();
    const auto encoded_node = builder.add_node(c::model_ops::row_cache_encode(encoding),
                                               std::array{input_id}).value();
    const auto bank_id = builder.result(encoded_node).value();
    const auto selected_node = builder.add_node(c::model_ops::row_cache_gather_decode(encoding),
                                                std::array{bank_id, ids_id}).value();
    const auto selected_id = builder.result(selected_node).value();
    ASSERT_TRUE(builder.publish(bank_id).ok());
    ASSERT_TRUE(builder.publish(selected_id).ok());
    const auto logical = std::move(builder).finish().value();
    auto made = g::vulkan::compile(context, logical);
    ASSERT_TRUE(made) << made.error().message();
    auto program = std::move(*made);
    auto slots = std::array{program.create_invocation().value(), program.create_invocation().value()};
    v::Tensor old_bank, old_selected;
    std::vector<std::uint8_t> old_bytes;
    std::vector<std::uint16_t> old_bits;
    for (int phase = 0; phase < 2; ++phase) {
      const auto input = row_cache_test::input_values(encoding, phase);
      const auto reference = row_cache_test::reference(encoding, input);
      const std::array<std::int32_t, 6> ids{4, 1, 4, phase, 2, 0};
      const auto source = context.upload(input_desc, std::as_bytes(std::span(input))).value();
      const auto index = context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
      ASSERT_TRUE(slots[phase].bind(std::array{source, index}).ok());
      auto submitted = slots[phase].submit();
      ASSERT_TRUE(submitted) << submitted.error().message();
      auto result = submitted->wait();
      ASSERT_TRUE(result) << result.error().message();
      ASSERT_EQ(result->get()->tensors().size(), 2u);
      const auto &bank = result->get()->tensors()[0];
      const auto &selected = result->get()->tensors()[1];
      EXPECT_EQ(read<std::uint8_t>(context, bank), reference.bytes);
      expect_selected(context, selected, reference.decoded, ids, encoding.depth());
      if (phase == 0) {
        old_bank = bank; old_selected = selected;
        old_bytes = read<std::uint8_t>(context, old_bank);
        old_bits = read<std::uint16_t>(context, old_selected);
      } else {
        EXPECT_EQ(read<std::uint8_t>(context, old_bank), old_bytes);
        EXPECT_EQ(read<std::uint16_t>(context, old_selected), old_bits);
      }
    }
  }
}

TEST(VulkanRowCache, CheckedFailureRetainsWholeOutputAndByteOriginsAreExplicit) {
  auto context = vulkan_test::context().value();
  const auto ids_desc = dense(c::DataType::kInt32, {2, 3});
  for (const auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                              c::RowCacheEncoding::float4_block32()}) {
    SCOPED_TRACE(encoding.depth());
    const auto input_desc = dense(c::DataType::kBFloat16,
        {row_cache_test::kRows, encoding.depth()});
    const auto bank_desc = c::row_cache_carrier_description(encoding, row_cache_test::kRows).value();
    const auto input = row_cache_test::input_values(encoding, 1);
    const auto reference = row_cache_test::reference(encoding, input);
    const auto source = context.upload(input_desc, std::as_bytes(std::span(input))).value();
    const auto encoded = execute(context, c::model_ops::row_cache_encode(encoding), std::array{source});
    ASSERT_TRUE(encoded) << encoded.error().message();
    EXPECT_EQ(read<std::uint8_t>(context, *encoded), reference.bytes);
    auto bank = odd_bank(context, bank_desc, reference.bytes);
    EXPECT_EQ(bank.native_offset() & 3u, 1u);
    auto gather = v::prepare(context, c::model_ops::row_cache_gather_decode(encoding),
        std::array{v::Argument(bank_desc), v::Argument(ids_desc)}).value();
    const auto output_desc = gather.output_descriptions()[0];
    const std::vector<std::uint16_t> sentinel(6 * encoding.depth(), 0x3f00);
    for (const auto bad : {-1, 5, INT32_MAX}) {
      const std::array<std::int32_t, 6> ids{4, 1, 4, 0, 2, bad};
      const auto index = context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
      const auto output = context.upload(output_desc, std::as_bytes(std::span(sentinel))).value();
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(recorder.append(gather, std::array{bank, index}, std::array{output}).ok());
      auto recorded = recorder.finish().value();
      auto work = recorded.submit();
      ASSERT_TRUE(work) << work.error().message();
      EXPECT_TRUE(work->wait().IsOutOfRange());
      EXPECT_EQ(read<std::uint16_t>(context, output), sentinel);
      EXPECT_FALSE(recorded.submit());
    }
    const std::array<std::int32_t, 6> ids{4, 1, 4, 0, 2, 0};
    const auto index = context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
    for (const auto bounds : {c::IndexBoundsPolicy::kChecked, c::IndexBoundsPolicy::kUnchecked}) {
      const auto decoded = execute(context, c::model_ops::row_cache_gather_decode(encoding, bounds),
                                    std::array{bank, index});
      ASSERT_TRUE(decoded) << decoded.error().message();
      expect_selected(context, *decoded, reference.decoded, ids, encoding.depth());
    }
    const std::vector<std::uint8_t> previous(reference.bytes.size(), 0xa5);
    auto encoder = v::prepare(context, c::model_ops::row_cache_encode(encoding),
                              v::Argument(input_desc)).value();
    for (const std::uint16_t bad : {std::uint16_t{0x7fc0}, std::uint16_t{0x7f80}, std::uint16_t{0xff80}}) {
      auto corrupt = input; corrupt.back() = bad;
      const auto invalid = context.upload(input_desc, std::as_bytes(std::span(corrupt))).value();
      const auto output = context.upload(bank_desc, std::as_bytes(std::span(previous))).value();
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(recorder.append(encoder, std::array{invalid}, std::array{output}).ok());
      auto recorded = recorder.finish().value();
      auto work = recorded.submit();
      ASSERT_TRUE(work) << work.error().message();
      EXPECT_TRUE(work->wait().IsInvalidArgument());
      EXPECT_EQ(read<std::uint8_t>(context, output), previous);
    }
    const auto scalar_desc = dense(c::DataType::kInt32, {});
    const std::array<std::int32_t, 1> scalar{4};
    const auto scalar_index = context.upload(scalar_desc, std::as_bytes(std::span(scalar))).value();
    const auto scalar_output = execute(context, c::model_ops::row_cache_gather_decode(encoding),
                                        std::array{bank, scalar_index});
    ASSERT_TRUE(scalar_output) << scalar_output.error().message();
    EXPECT_EQ(scalar_output->description().rank(), 1u);
    expect_selected(context, *scalar_output, reference.decoded, scalar, encoding.depth());
  }
}

TEST(VulkanRowCache, EmptyInputsAndReservedPayloadUseTheSameByteContract) {
  auto context = vulkan_test::context().value();
  for (const auto encoding : {c::RowCacheEncoding::float8_block64_tail64(),
                              c::RowCacheEncoding::float4_block32()}) {
    const auto input_desc = dense(c::DataType::kBFloat16, {0, encoding.depth()});
    const auto input = context.upload(input_desc, {}).value();
    const auto encoded = execute(context, c::model_ops::row_cache_encode(encoding), std::array{input});
    ASSERT_TRUE(encoded) << encoded.error().message();
    const auto ids_desc = dense(c::DataType::kInt32, {0});
    const auto index = context.upload(ids_desc, {}).value();
    const auto result = execute(context, c::model_ops::row_cache_gather_decode(encoding),
                                 std::array{*encoded, index});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result->description().element_count().value(), 0u);
    const bool four = encoding.format() == c::RowCacheFormat::kFloat4Block32;
    auto payload = std::vector<std::uint8_t>(encoding.row_bytes());
    const auto bank_desc = c::row_cache_carrier_description(encoding, 1).value();
    const auto scalar_desc = dense(c::DataType::kInt32, {});
    const std::array<std::int32_t, 1> zero{0};
    const auto id = context.upload(scalar_desc, std::as_bytes(std::span(zero))).value();
    if (four) payload[0] = 0x21; // .5 and1, scale2^-127
    else { payload[0] = 0x40; payload[1] = 0x80; payload[2] = 0x7f; }
    const auto bank = odd_bank(context, bank_desc, payload);
    auto decoded = execute(context, c::model_ops::row_cache_gather_decode(encoding), std::array{bank, id});
    ASSERT_TRUE(decoded) << decoded.error().message();
    auto actual = read<std::uint16_t>(context, *decoded);
    EXPECT_EQ(actual[0], four ? 0x0020u : 0x0080u);
    EXPECT_EQ(actual[1], four ? 0x0040u : 0x8000u);
    if (!four) EXPECT_TRUE(std::isnan(row_cache_test::decode(actual[2])));
    payload[four ? 64 : 576] = 255;
    const auto reserved = context.upload(bank_desc, std::as_bytes(std::span(payload))).value();
    decoded = execute(context, c::model_ops::row_cache_gather_decode(encoding), std::array{reserved, id});
    ASSERT_TRUE(decoded) << decoded.error().message();
    actual = read<std::uint16_t>(context, *decoded);
    EXPECT_TRUE(std::isnan(row_cache_test::decode(actual[0])));
  }
}
} // namespace
