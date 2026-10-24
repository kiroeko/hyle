#pragma once

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <span>
#include <vector>
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

namespace ggml_constant_test {
inline void dense_rhs(hyle::compute::Context &context,
                       hyle::mem::MemoryResourceRef resource,
                       hyle::compute::StorageOwner owner = {}) {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  namespace m = hyle::mem;
  constexpr std::int64_t batches=2, rows=3, k=37, n=7;
  const auto a_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{batches,rows,k}).value();
  const auto w_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{k,n}).value();
  const auto y_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{batches,rows,n}).value();
  const auto buffer=[&](const c::TensorDesc &d){return m::Buffer::create(resource,d.required_bytes().value(),256).value();};
  for(bool constant:{true,false}) {
    SCOPED_TRACE(constant);
    auto a=buffer(a_desc),w=buffer(w_desc);
    std::array y{buffer(y_desc),buffer(y_desc)}, negative_weights{buffer(w_desc),buffer(w_desc)};
    std::vector<float> weight(k*n);
    const auto write_weights=[&](int phase){
      for(std::size_t i=0;i<weight.size();++i)weight[i]=float((int((i*19+phase*13)%47)-23)*.01371);
      return w.view().write(std::as_bytes(std::span(weight)));
    };
    ASSERT_TRUE(write_weights(0).ok());
    auto builder=g::GraphBuilder::create().value();
    const auto x=builder.input(a_desc).value();
    const auto weights=(constant?builder.constant(w_desc):builder.input(w_desc)).value();
    const auto product=builder.add(c::MatMul{},x,weights).value();
    // This consumer needs the original logical layout of the same constant.
    const auto neg=builder.add(c::Unary{c::UnaryKind::kNeg},weights).value();
    ASSERT_TRUE(builder.publish(product).ok());ASSERT_TRUE(builder.publish(neg).ok());
    const auto logical=std::move(builder).finish().value();
    c::ReadLease ready;
    if (constant && owner) ready = owner.adopt_buffer(std::move(w)).value();
    const auto binding = ready ? ready.tensor(w_desc).value()
                              : c::ConstTensorView::create(w_desc,w.view()).value();
    const std::array constants{g::ConstantBinding{weights,binding,
        g::ConstantPolicy::kBorrow, ready ? owner : c::StorageOwner{}, ready}};
    const std::array storage{g::StorageDomain{.resource=resource,.owner=owner}};
    const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(context).value(),.default_storage=g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target=g::ExecutionTargetId{0}}};
    auto compiled=g::compile(logical,
        {.storage_domains=storage,.targets=targets,.lanes=lanes},
        {.constants=constant?std::span(constants):std::span<const g::ConstantBinding>{}});
    ASSERT_TRUE(compiled)<<compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(),1u);
    EXPECT_EQ(compiled->report.packed_constant_count(),constant?1u:0u);
    EXPECT_EQ(compiled->report.packed_constant_bytes(),constant?w_desc.required_bytes().value():0u);
    std::array invocation{compiled->executable.create_invocation().value(),compiled->executable.create_invocation().value()};
    std::array queue{context.create_queue().value(),context.create_queue().value()};
    for(std::size_t slot=0;slot<2;++slot){
      ASSERT_TRUE(invocation[slot].bind(x,c::ConstTensorView::create(a_desc,a.view()).value()).ok());
      if(!constant)ASSERT_TRUE(invocation[slot].bind(weights,c::ConstTensorView::create(w_desc,w.view()).value()).ok());
      ASSERT_TRUE(invocation[slot].bind(product,c::TensorView::create(y_desc,y[slot].view()).value()).ok());
      ASSERT_TRUE(invocation[slot].bind(neg,c::TensorView::create(w_desc,negative_weights[slot].view()).value()).ok());
      const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue[slot]}};
      ASSERT_TRUE(invocation[slot].prepare_replay(queues).ok());
    }
    std::vector<float> old;
    for(int phase=0;phase<2;++phase){
      if(!constant&&phase)ASSERT_TRUE(write_weights(phase).ok());
      std::vector<float> input(batches*rows*k);
      for(std::size_t i=0;i<input.size();++i)input[i]=float((int((i*11+phase*7)%43)-21)*.01913);
      ASSERT_TRUE(a.view().write(std::as_bytes(std::span(input))).ok());
      auto done=invocation[phase].submit(queue[phase]);ASSERT_TRUE(done)<<done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::vector<float> actual(batches*rows*n),actual_weights(k*n);
      ASSERT_TRUE(y[phase].view().read(std::as_writable_bytes(std::span(actual))).ok());
      ASSERT_TRUE(negative_weights[phase].view().read(std::as_writable_bytes(std::span(actual_weights))).ok());
      for(std::size_t row=0;row<std::size_t(batches*rows);++row)
        for(std::size_t col=0;col<std::size_t(n);++col){
          double expected=0;for(std::size_t inner=0;inner<std::size_t(k);++inner)
            expected+=double(input[row*k+inner])*weight[inner*n+col];
          EXPECT_NEAR(actual[row*n+col],expected,2e-6+std::abs(expected)*6e-6)<<row<<'/'<<col;
        }
      for(std::size_t i=0;i<weight.size();++i)EXPECT_EQ(actual_weights[i],-weight[i]);
      if(!phase)old=actual;
      else{std::vector<float> previous(old.size());
        ASSERT_TRUE(y[0].view().read(std::as_writable_bytes(std::span(previous))).ok());EXPECT_EQ(previous,old);}
    }
  }
}
} // namespace ggml_constant_test
