#include <bit>
#include <cmath>
#include <numeric>
#include "region_test_support.h"
#include "tensor_test_support.h"
#include "../compute_models/batched_sink_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/providers/ggml_cuda.h"

namespace {
using namespace ggml_cuda_graph_test;
using namespace ggml_cuda_tensor_test;
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::mem::Buffer;

void near(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_NEAR(actual[i], expected[i], 1e-5F) << i;
}
std::vector<float> attention_reference(std::span<const float> q,
    std::span<const float> k, std::span<const float> v, int rows, float scale,
    std::span<const float> mask, bool batched_mask, std::span<const float> sinks) {
  constexpr int batch=2, heads=64, kv_heads=1, keys=11, depth=512;
  std::vector<float> out(batch*rows*heads*depth);
  for(int b=0;b<batch;++b)for(int t=0;t<rows;++t)for(int h=0;h<heads;++h){
    std::array<double,keys> weights{};double maximum=sinks[h],denominator=0;
    const int qbase=((b*rows+t)*heads+h)*depth,kh=h/(heads/kv_heads);
    for(int s=0;s<keys;++s){double dot=0;for(int d=0;d<depth;++d)
      dot+=double(q[qbase+d])*k[((b*keys+s)*kv_heads+kh)*depth+d];
      weights[s]=dot*scale+mask[((batched_mask?b:0)*rows+t)*keys+s];maximum=std::max(maximum,weights[s]);}
    denominator=std::exp(double(sinks[h])-maximum);
    for(auto&w:weights){w=std::exp(w-maximum);denominator+=w;}
    for(int d=0;d<depth;++d){double sum=0;for(int s=0;s<keys;++s)
      sum+=weights[s]*v[((b*keys+s)*kv_heads+kh)*depth+d];out[qbase+d]=float(sum/denominator);}
  }
  return out;
}

// The diagnostic build changes only the private tile selector to two rows.
// This keeps scalar/dense/checked-leaf tests small while running the real
// native CONCAT and gallocr lifecycle, not a mock allocator or kernel.
void small_regions(bool checked) {
  constexpr int batch=2, queries=7, other_queries=5, heads=64, depth=512;
  auto context=c::Context::create(c::cuda::device(0).value(),g::ggml_cuda::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  const auto qdesc=dense(c::DataType::kFloat32,{batch,queries,heads,depth});
  const auto q2desc=dense(c::DataType::kFloat32,{batch,other_queries,heads,depth});
  const auto kvdesc=dense(c::DataType::kFloat32,{batch,11,1,depth});
  const auto idesc=dense(c::DataType::kInt32,{queries,11});
  auto builder=g::GraphBuilder::create().value();
  std::vector<c::TensorDesc> descriptions{qdesc,kvdesc,kvdesc,q2desc,
      checked ? dense(c::DataType::kFloat32,{queries,11}) : dense(c::DataType::kFloat32,{batch,queries,11}),
      dense(c::DataType::kFloat32,{other_queries,11}),
      dense(c::DataType::kFloat32,{heads})};
  if(checked)descriptions.push_back(idesc);
  std::vector<g::ValueId> ports;for(const auto&d:descriptions)ports.push_back(builder.input(d).value());
  auto q=ports[0];
  auto mask=ports[4];
  if(checked)mask=builder.add(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),mask,ports[7]).value();
  q=builder.add(c::Unary{c::UnaryKind::kRsqrt},q).value();
  q=builder.add(c::Hadamard{},q).value();
  const auto first=builder.add(c::model_ops::sink_attention(.25F),q,ports[1],ports[2],mask,ports[6]).value();
  const auto second=builder.add(c::model_ops::sink_attention(.125F),ports[3],ports[1],ports[2],ports[5],ports[6]).value();
  const auto neg1=builder.add(c::Unary{c::UnaryKind::kNeg},first).value();
  const auto neg2=builder.add(c::Unary{c::UnaryKind::kNeg},second).value();
  const std::array result_ids{first,neg1,second,neg2};
  for(auto id:result_ids)ASSERT_TRUE(builder.publish(id).ok());
  auto logical=std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{.resource=memory.ref()}};
  const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(context).value(),.default_storage=g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target=g::ExecutionTargetId{0}}};
  auto compiled=g::compile(logical,{.storage_domains=domains,.targets=targets,.lanes=lanes});
  ASSERT_TRUE(compiled)<<compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(),1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,c::ggml_cuda::kProviderId);
  std::array calls{compiled->executable.create_invocation().value(),compiled->executable.create_invocation().value()};
  std::array queues{context.create_queue().value(),context.create_queue().value()};
  std::vector<Buffer> inputs;for(const auto&d:descriptions)inputs.push_back(output_buffer(memory.ref(),d));
  std::vector<Buffer> outputs;for(int slot=0;slot<2;++slot)for(int i=0;i<4;++i)
    outputs.push_back(output_buffer(memory.ref(),i<2?qdesc:q2desc));
  for(int slot=0;slot<2;++slot){for(std::size_t i=0;i<ports.size();++i)
    ASSERT_TRUE(calls[slot].bind(ports[i],c::ConstTensorView::create(descriptions[i],inputs[i].view()).value()).ok());
    for(int i=0;i<4;++i)ASSERT_TRUE(calls[slot].bind(result_ids[i],c::TensorView::create(i<2?qdesc:q2desc,outputs[slot*4+i].view()).value()).ok());}
  std::vector<std::vector<float>> old_outputs;
  for(int phase=0;phase<2;++phase){
    std::vector<std::vector<float>> data;
    for(int i=0;i<7;++i){data.emplace_back(descriptions[i].element_count().value());
      for(std::size_t j=0;j<data.back().size();++j){
        if(i==0)data.back()[j]=1.F+float((j/depth+j%8+phase*7)%19)/16.F;
        else if(i==4||i==5)data.back()[j]=j%11==0||(j/11+j%11+phase)%3? -float(j%3)/8.F:-INFINITY;
        else if(i==6)data.back()[j]=float(int((j*7+phase*3)%17)-8)/8.F;
        else data.back()[j]=(i==2?.25F:0.F)+float(int((j*7+phase*11+i)%23)-11)/32.F;
      }
      ASSERT_TRUE(inputs[i].view().write(std::as_bytes(std::span(data.back()))).ok());}
    std::vector<std::int32_t> ids(queries*11);
    for(std::size_t j=0;j<ids.size();++j)ids[j]=int((j+phase*3)%11);
    if(checked)ASSERT_TRUE(inputs[7].view().write(std::as_bytes(std::span(ids))).ok());
    auto query=data[0];
    auto selected_mask=data[4];if(checked)for(std::size_t j=0;j<selected_mask.size();++j)selected_mask[j]=data[4][(j/11)*11+ids[j]];
    for(auto&v:query)v=1.F/std::sqrt(v);
    const auto untransformed=query;const float scale=1.F/std::sqrt(float(depth));
    // Source rows repeat every8 lanes, so H512 has only its first8 nonzero
    // frequencies. Evaluate the real normalized matrix in FP64 analytically.
    std::fill(query.begin(),query.end(),0.F);
    for(std::size_t row=0;row<query.size()/depth;++row)for(int col=0;col<8;++col){double sum=0;
      for(int d=0;d<8;++d)sum+=double(untransformed[row*depth+d])*(std::popcount(unsigned(col&d))%2?-scale:scale);
      query[row*depth+col]=float(sum*64); }
    const auto a=attention_reference(query,data[1],data[2],queries,.25F,selected_mask,!checked,data[6]);
    const auto b=attention_reference(data[3],data[1],data[2],other_queries,.125F,data[5],false,data[6]);
    std::array expected{a,a,b,b};for(int i:{1,3})for(auto&v:expected[i])v=-v;
    for(int slot=phase==0?0:1;slot<2;++slot){auto accepted=calls[slot].submit(queues[slot]);ASSERT_TRUE(accepted)<<accepted.error().message();ASSERT_TRUE(accepted->wait().ok());
      for(int i=0;i<4;++i)near(half_test::read<float>(outputs[slot*4+i]),expected[i]);
      ASSERT_FALSE(::testing::Test::HasFailure());}
    if(phase==0)for(int i=0;i<4;++i)old_outputs.push_back(half_test::read<float>(outputs[i]));
    else for(int i=0;i<4;++i)EXPECT_EQ(half_test::read<float>(outputs[i]),old_outputs[i]);
    if(checked&&phase==1){std::vector<std::vector<float>> before;for(const auto&v:outputs)before.push_back(half_test::read<float>(v));
      ids.back()=11;ASSERT_TRUE(inputs[7].view().write(std::as_bytes(std::span(ids))).ok());
      auto bad=calls[1].submit(queues[1]);ASSERT_TRUE(bad)<<bad.error().message();EXPECT_TRUE(bad->wait().IsOutOfRange());
      for(std::size_t i=0;i<outputs.size();++i)EXPECT_EQ(half_test::read<float>(outputs[i]),before[i]);
      EXPECT_EQ(half_test::read<std::int32_t>(inputs[7]),ids);}
  }
}
}

TEST(GgmlCudaAttentionRegion, RealBatchedSinkTilesFeedPublishedNegative) {
  namespace bs=batched_sink_test;
  using namespace ggml_cuda_graph_test;
  using namespace ggml_cuda_tensor_test;
  auto context=c::Context::create(c::cuda::device(0).value(),g::ggml_cuda::provider()).value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  bs::Fixture f(128,true,3,8,2049);
  auto builder=g::GraphBuilder::create().value();std::array<g::ValueId,5> ports;
  for(std::size_t i=0;i<ports.size();++i)ports[i]=builder.input(f.descriptions[i]).value();
  auto y=builder.add(f.operation,ports[0],ports[1],ports[2],ports[3],ports[4]).value();
  auto neg=builder.add(c::Unary{c::UnaryKind::kNeg},y).value();
  ASSERT_TRUE(builder.publish(y).ok());ASSERT_TRUE(builder.publish(neg).ok());
  auto graph=std::move(builder).finish().value();const std::array domains{g::StorageDomain{.resource=memory.ref()}};
  const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(context).value(),.default_storage=g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target=g::ExecutionTargetId{0}}};
  auto compiled=g::compile(graph,{.storage_domains=domains,.targets=targets,.lanes=lanes});
  ASSERT_TRUE(compiled)<<compiled.error().message();ASSERT_EQ(compiled->report.step_count(),1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,c::ggml_cuda::kProviderId);
  auto call=compiled->executable.create_invocation().value();auto queue=context.create_queue().value();
  std::array<Buffer,5> inputs;for(std::size_t i=0;i<inputs.size();++i){inputs[i]=output_buffer(memory.ref(),f.descriptions[i]);
    ASSERT_TRUE(call.bind(ports[i],c::ConstTensorView::create(f.descriptions[i],inputs[i].view()).value()).ok());}
  auto first=output_buffer(memory.ref(),f.descriptions[0]),second=output_buffer(memory.ref(),f.descriptions[0]);
  ASSERT_TRUE(call.bind(y,c::TensorView::create(f.descriptions[0],first.view()).value()).ok());
  ASSERT_TRUE(call.bind(neg,c::TensorView::create(f.descriptions[0],second.view()).value()).ok());
  for(int phase=0;phase<2;++phase){f.update(phase);for(std::size_t i=0;i<inputs.size();++i)
    ASSERT_TRUE(inputs[i].view().write(std::as_bytes(std::span(f.payloads[i]))).ok());
    auto accepted=call.submit(queue);ASSERT_TRUE(accepted)<<accepted.error().message();ASSERT_TRUE(accepted->wait().ok());
    auto expected=f.expected();bs::check(half_test::read<float>(first),expected);for(auto&v:expected)v=-v;
    bs::check(half_test::read<float>(second),expected);ASSERT_FALSE(::testing::Test::HasFailure());}
}

TEST(GgmlCudaAttentionRegion, ColdScalarsDenseConstantsAndDifferentQueriesReplay) { small_regions(false); }
TEST(GgmlCudaAttentionRegion, CheckedIdsKeepEveryOutputPrivateAcrossAttention) { small_regions(true); }
