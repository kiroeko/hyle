#include <gtest/gtest.h>
#include "../compute_models/row_cache_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
namespace {
namespace c=hyle::compute;namespace m=hyle::mem;
TEST(CpuRowCachePadded, OnlyMinusOnePadsAndInvalidIdsNeverPublish){
  auto context=c::cpu::create_context().value();auto queue=context.create_queue().value();
  for(auto encoding:{c::RowCacheEncoding::float4_block32(),c::RowCacheEncoding::float8_block64_tail64()}){
    SCOPED_TRACE(encoding.depth());const auto bank_desc=c::row_cache_carrier_description(encoding,5).value();
    const auto ids_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{2,3}).value();
    const auto operation=c::model_ops::row_cache_gather_decode_padded(encoding);
    const auto output_desc=c::infer_single_output(operation,std::array{bank_desc,ids_desc}).value();
    auto prepared=c::prepare(context,operation,std::array{bank_desc,ids_desc},std::array{output_desc},m::system_resource()).value();
    std::array outputs{half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(6*encoding.depth())),half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(6*encoding.depth()))};
    std::vector<std::uint16_t> old;
    for(int phase:{0,1}){const auto source=row_cache_test::reference(encoding,row_cache_test::input_values(encoding,phase));
      auto bank=half_test::buffer<std::uint8_t>(source.bytes);auto ids=half_test::buffer<std::int32_t>(std::array<std::int32_t,6>{-1,4,0,2,-1,1});
      const std::array input{c::ConstTensorView::create(bank_desc,bank.view()).value(),c::ConstTensorView::create(ids_desc,ids.view()).value()};
      auto slot=prepared.create_invocation().value();auto done=slot.submit(queue,input,std::array{c::TensorView::create(output_desc,outputs[phase].view()).value()});
      ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok());const auto actual=half_test::read<std::uint16_t>(outputs[phase]);
      const std::array<std::int32_t,6> selected{-1,4,0,2,-1,1};for(std::size_t row=0;row<6;++row)for(std::size_t d=0;d<encoding.depth();++d)
        EXPECT_EQ(actual[row*encoding.depth()+d],selected[row]<0?std::uint16_t{0}:source.decoded[std::size_t(selected[row])*encoding.depth()+d]);
      if(!phase)old=actual;else EXPECT_EQ(half_test::read<std::uint16_t>(outputs[0]),old);
      for(std::int32_t invalid:{-2,5}){auto bad=half_test::buffer<std::int32_t>(std::array<std::int32_t,6>{-1,4,0,2,-1,invalid});
        const std::vector<std::uint16_t> sentinel(6*encoding.depth(),0x5a5a);auto output=half_test::buffer<std::uint16_t>(sentinel);
        auto failed=prepared.create_invocation().value();auto completion=failed.submit(queue,std::array{input[0],c::ConstTensorView::create(ids_desc,bad.view()).value()},std::array{c::TensorView::create(output_desc,output.view()).value()});
        ASSERT_TRUE(completion);EXPECT_TRUE(completion->wait().IsOutOfRange());EXPECT_EQ(half_test::read<std::uint16_t>(output),sentinel);}
    }
    // Empty physical bank, nonempty padded IDs: the decoder may not map/read
    // a source row or form a bank pointer for these positive-zero outputs.
    const auto empty_desc=c::row_cache_carrier_description(encoding,0).value();auto empty_bank=half_test::buffer<std::uint8_t>(std::vector<std::uint8_t>{});
    auto ids=half_test::buffer<std::int32_t>(std::vector<std::int32_t>(6,-1));
    auto padded=c::prepare(context,operation,std::array{empty_desc,ids_desc},std::array{output_desc},m::system_resource()).value();
    auto invocation=padded.create_invocation().value();auto done=invocation.submit(queue,std::array{c::ConstTensorView::create(empty_desc,empty_bank.view()).value(),c::ConstTensorView::create(ids_desc,ids.view()).value()},std::array{c::TensorView::create(output_desc,outputs[1].view()).value()});
    ASSERT_TRUE(done);EXPECT_TRUE(done->wait().ok());EXPECT_EQ(half_test::read<std::uint16_t>(outputs[1]),std::vector<std::uint16_t>(6*encoding.depth(),0));
    EXPECT_FALSE(c::infer_single_output(c::model_ops::row_cache_gather_decode(encoding),std::array{empty_desc,ids_desc}));
  }
}
}
