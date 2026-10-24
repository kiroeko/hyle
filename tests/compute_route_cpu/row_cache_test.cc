#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "../compute_models/row_cache_fixture.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using B = hyle::simd::bfloat16_t;
using row_cache_test::kRows;
using row_cache_test::input_values;
using row_cache_test::reference;

void exercise(c::RowCacheEncoding encoding) {
  auto stats = m::StatsAllocator::create(m::system_allocator()).value();
  auto context = c::cpu::create_context({.metadata_allocator=stats.get()}).value();
  auto queue = context.create_queue().value();
  const auto depth = encoding.depth();
  const auto source_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t,2>{kRows,depth}).value();
  const auto bank_desc = c::row_cache_carrier_description(encoding,kRows).value();
  const auto id_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t,2>{2,3}).value();
  const std::array decode_desc{bank_desc,id_desc};
  const auto encode_op = c::model_ops::row_cache_encode(encoding);
  auto encoder = c::prepare(context,encode_op,std::array{source_desc},std::array{bank_desc},m::system_resource()).value();
  EXPECT_EQ(encoder.provider().id,c::native::kProviderId);
  EXPECT_EQ(encoder.workspace_requirement().size,0u);
  auto source = half_test::buffer<std::uint16_t>(input_values(encoding,0));
  auto bank = half_test::buffer<std::uint8_t>(std::vector<std::uint8_t>(*bank_desc.required_bytes(),0xa5));
  auto ids = half_test::buffer<std::int32_t>(std::array<std::int32_t,6>{4,1,4,0,3,2});
  const auto source_view = c::ConstTensorView::create(source_desc,source.view()).value();
  const auto bank_output = c::TensorView::create(bank_desc,bank.view()).value();
  const auto bank_input = c::ConstTensorView::create(bank_desc,bank.view()).value();
  const auto id_view = c::ConstTensorView::create(id_desc,ids.view()).value();
  for (auto bounds : {c::IndexBoundsPolicy::kChecked,c::IndexBoundsPolicy::kUnchecked}) {
    const auto op = c::model_ops::row_cache_gather_decode(encoding,bounds);
    const auto output_desc = c::infer_single_output(op,decode_desc).value();
    auto decoder = c::prepare(context,op,decode_desc,std::array{output_desc},m::system_resource()).value();
    EXPECT_EQ(decoder.provider().id,c::native::kProviderId);
    EXPECT_EQ(decoder.workspace_requirement().size,0u);
    auto output = half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(6 * depth));
    const auto output_view = c::TensorView::create(output_desc,output.view()).value();
    auto encode = encoder.create_invocation().value();
    auto decode = decoder.create_invocation().value();
    for (int phase : {0,1}) {
      auto data = input_values(encoding,phase);const auto expected = reference(encoding,data);
      const std::array<std::int32_t,6> selected = phase ? std::array<std::int32_t,6>{3,4,1,2,0,4}
                                                      : std::array<std::int32_t,6>{4,1,4,0,3,2};
      ASSERT_TRUE(source.view().write(std::as_bytes(std::span(data))).ok());
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      const auto before = stats.snapshot().total_allocs;
      { auto done=encode.submit(queue,source_view,bank_output);ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok()); }
      { auto done=decode.submit(queue,std::array{bank_input,id_view},std::array{output_view});ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok()); }
      EXPECT_EQ(stats.snapshot().total_allocs,before);
      EXPECT_EQ(half_test::read<std::uint8_t>(bank),expected.bytes);
      const auto actual = half_test::read<std::uint16_t>(output);
      for (std::size_t i=0;i<selected.size();++i)
        for (std::size_t d=0;d<depth;++d)
          ASSERT_EQ(actual[i*depth+d],expected.decoded[std::size_t(selected[i])*depth+d])<<i<<":"<<d;
    }
    if (bounds == c::IndexBoundsPolicy::kChecked) for (std::int32_t invalid : {-1,5,INT32_MAX}) {
      const std::array<std::int32_t,6> selected{4,1,4,0,3,invalid};
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      const auto sentinel = std::vector<std::uint16_t>(6*depth,0x5a5a);
      ASSERT_TRUE(output.view().write(std::as_bytes(std::span(sentinel))).ok());
      auto failed=decoder.create_invocation().value();
      auto done=failed.submit(queue,std::array{bank_input,id_view},std::array{output_view});
      ASSERT_TRUE(done);EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(half_test::read<std::uint16_t>(output),sentinel);
    }
  }
  const auto sentinel=std::vector<std::uint8_t>(bank.size(),0xa5);
  for (std::uint16_t invalid : {std::uint16_t{0x7fc0},std::uint16_t{0x7f80},std::uint16_t{0xff80}}) {
    auto nonfinite=input_values(encoding,0);nonfinite.back()=invalid;
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(nonfinite))).ok());
    ASSERT_TRUE(bank.view().write(std::as_bytes(std::span(sentinel))).ok());
    auto failed=encoder.create_invocation().value();auto done=failed.submit(queue,source_view,bank_output);
    ASSERT_TRUE(done);EXPECT_TRUE(done->wait().IsInvalidArgument());
    EXPECT_EQ(half_test::read<std::uint8_t>(bank),sentinel);
  }
}

TEST(CpuRowCache, PackedRowsRetainQatBytesAndSelectedBfloatBoundary) {
  exercise(c::RowCacheEncoding::float8_block64_tail64());
  exercise(c::RowCacheEncoding::float4_block32());
}

TEST(CpuRowCache, ScalarEmptyAndReservedPayloadUseFormatSemantics) {
  auto context=c::cpu::create_context().value();auto queue=context.create_queue().value();
  for(auto encoding:{c::RowCacheEncoding::float8_block64_tail64(),c::RowCacheEncoding::float4_block32()}) {
    const auto bank_desc=c::row_cache_carrier_description(encoding,1).value();
    const auto scalar=c::TensorDesc::dense(c::DataType::kInt32,{}).value();
    const auto op=c::model_ops::row_cache_gather_decode(encoding);
    const std::array inputs{bank_desc,scalar};const auto output_desc=c::infer_single_output(op,inputs).value();
    auto prepared=c::prepare(context,op,inputs,std::array{output_desc},m::system_resource()).value();
    auto bytes=std::vector<std::uint8_t>(encoding.row_bytes());
    const bool four=encoding.format()==c::RowCacheFormat::kFloat4Block32;
    bytes[four?64:576]=255;
    if(!four){bytes[64]=0x7f;bytes[577]=127;}
    auto bank=half_test::buffer<std::uint8_t>(bytes);
    auto id=half_test::buffer<std::int32_t>(std::array<std::int32_t,1>{0});
    auto out=half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(encoding.depth()));
    const std::array views{c::ConstTensorView::create(bank_desc,bank.view()).value(),c::ConstTensorView::create(scalar,id.view()).value()};
    auto invocation=prepared.create_invocation().value();auto done=invocation.submit(queue,views,std::array{c::TensorView::create(output_desc,out.view()).value()});
    ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok());
    const auto actual=half_test::read<std::uint16_t>(out);
    EXPECT_TRUE(std::isnan(half_test::decode_bits<B>(actual[0])));
    if(!four)EXPECT_TRUE(std::isnan(half_test::decode_bits<B>(actual[64])));
    const auto empty_bank=c::row_cache_carrier_description(encoding,0).value();
    const auto empty_ids=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{0,2}).value();
    const std::array empty_inputs{empty_bank,empty_ids};const auto empty_output=c::infer_single_output(op,empty_inputs).value();
    auto empty_program=c::prepare(context,op,empty_inputs,std::array{empty_output},m::system_resource()).value();
    auto empty=m::Buffer::create(m::system_resource(),0).value();
    const std::array empty_views{c::ConstTensorView::create(empty_bank,empty.view()).value(),c::ConstTensorView::create(empty_ids,empty.view()).value()};
    auto empty_invocation=empty_program.create_invocation().value();auto empty_done=empty_invocation.submit(queue,empty_views,std::array{c::TensorView::create(empty_output,empty.view()).value()});
    ASSERT_TRUE(empty_done);EXPECT_TRUE(empty_done->wait().ok());
    const auto empty_source=c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{0,encoding.depth()}).value();
    auto empty_encoder=c::prepare(context,c::model_ops::row_cache_encode(encoding),std::array{empty_source},std::array{empty_bank},m::system_resource()).value();
    auto encode_empty=empty_encoder.create_invocation().value();auto encoded=encode_empty.submit(queue,c::ConstTensorView::create(empty_source,empty.view()).value(),c::TensorView::create(empty_bank,empty.view()).value());
    ASSERT_TRUE(encoded);EXPECT_TRUE(encoded->wait().ok());
  }
}
} // namespace
