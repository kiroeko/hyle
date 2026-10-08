#pragma once
#include "tensor_test_support.h"
#include "gguf_coverage_fixture.h"

namespace apple_exact_row_resource_test {
using namespace apple_tensor_test;
template <class Prepare>
void check(Prepare prepare) {
  const cpu_c5_inputs::Case item{"exact-q4-resource",cpu_c5_inputs::Kind::kEmbedding,
      GGML_TYPE_Q4_K,c::GgufQuantizedType::kQ4K,true,256,8,1,0,1};
  const cpu_c5_gguf_fixture::Bank bank(item);
  const auto table=c::gguf_weight_description(item.quant,c::DataType::kFloat32,256,8)->transposed().value();
  const auto ids_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{2}).value();
  const std::array descriptions{table,ids_desc};
  auto context=a::Context::create().value();
  auto program=prepare(context,c::Operation(c::Gather{}),descriptions);
  ASSERT_TRUE(program)<<program.error().message();
  const auto spec=a::input_spec(table).value();
  const auto bytes=bank.rows(8);std::vector<std::uint8_t> stored(bytes.size());
  std::memcpy(stored.data(),bytes.data(),bytes.size());
  std::vector<mx::array> inputs{ready(mx::array(stored.data(),spec.shape,mx::uint8)),
      ready(mx::array({std::int32_t{7},std::int32_t{0}},{2}))};
  auto slot=program->create_invocation().value();
  ASSERT_TRUE(slot.bind(inputs).ok());auto work=slot.submit();ASSERT_TRUE(work);
  auto result=work->wait();ASSERT_TRUE(result)<<result.error().message();
  const auto retained=result->get()->arrays()[0];const auto before=values(retained);
  for(const auto bad:{std::int32_t{-1},std::int32_t{8}}) {
    auto failed_slot=program->create_invocation().value();
    inputs[1]=ready(mx::array({bad,std::int32_t{0}},{2}));
    ASSERT_TRUE(failed_slot.bind(inputs).ok());auto failed_work=failed_slot.submit();ASSERT_TRUE(failed_work);
    const auto failed=failed_work->wait();ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsOutOfRange());EXPECT_EQ(values(retained),before);
  }
  auto recovered=program->create_invocation().value();
  inputs[1]=ready(mx::array({std::int32_t{1},std::int32_t{2}},{2}));
  ASSERT_TRUE(recovered.bind(inputs).ok());auto good_work=recovered.submit();ASSERT_TRUE(good_work);
  auto good=good_work->wait();ASSERT_TRUE(good)<<good.error().message();
  ASSERT_TRUE(context.close().ok());EXPECT_EQ(values(retained),before);
  const auto empty=c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{0}).value();
  auto empty_context=a::Context::create().value();
  auto empty_program=prepare(empty_context,c::Operation(c::Gather{}),std::array{table,empty});
  ASSERT_TRUE(empty_program)<<empty_program.error().message();
  auto empty_slot=empty_program->create_invocation().value();
  const std::array empty_inputs{inputs[0],ready(mx::zeros({0},mx::int32,stream()))};
  ASSERT_TRUE(empty_slot.bind(empty_inputs).ok());auto empty_work=empty_slot.submit();ASSERT_TRUE(empty_work);
  auto empty_result=empty_work->wait();ASSERT_TRUE(empty_result)<<empty_result.error().message();
  EXPECT_EQ(empty_result->get()->arrays()[0].size(),0U);
}
} // namespace apple_exact_row_resource_test
