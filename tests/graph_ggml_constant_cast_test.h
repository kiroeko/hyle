#pragma once

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <vector>
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

namespace ggml_constant_cast_test {
inline void exercise_incoming_control(hyle::compute::Context &first,
                                      hyle::compute::Context &second,
                                      hyle::mem::MemoryResourceRef resource,
                                      hyle::compute::StorageOwner owner = {}) {
  namespace c=hyle::compute; namespace g=hyle::graph; namespace m=hyle::mem;
  const auto f=c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{2,9}).value();
  const auto w=c::TensorDesc::dense(c::DataType::kBFloat16,f.dimensions()).value();
  const auto alloc=[&](const c::TensorDesc &d){return m::Buffer::create(
      resource,d.required_bytes().value(),256).value();};
  auto source=alloc(f),weight=alloc(w);
  std::array<std::uint16_t,18> bits;
  std::array<float,18> weights;
  for(std::size_t i=0;i<bits.size();++i){
    bits[i]=static_cast<std::uint16_t>(0x3f00+i*8);
    weights[i]=std::bit_cast<float>(std::uint32_t(bits[i])<<16);
  }
  ASSERT_TRUE(weight.view().write(std::as_bytes(std::span(bits))).ok());
  auto builder=g::GraphBuilder::create().value();
  const auto x=builder.input(f).value(),constant=builder.constant(w).value();
  // TopK is outside the ggml region. The first TopK precedes Cast only by
  // control; the second reads only Cast, never the independent region output.
  const auto before=builder.add_node(c::Operation(c::TopK{.count=3}),
      std::array{x}).value();
  const auto cast=builder.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),
      std::array{constant}).value();
  ASSERT_TRUE(builder.depends_on(cast,before).ok());
  const auto independent=builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{x}).value();
  const auto after=builder.add_node(c::Operation(c::TopK{.count=3}),
      std::array{builder.result(cast).value()}).value();
  const auto neg=builder.result(independent).value(),selected=builder.result(after).value();
  ASSERT_TRUE(builder.publish(neg).ok());ASSERT_TRUE(builder.publish(selected).ok());
  const auto graph=std::move(builder).finish().value();
  const std::array storage{g::StorageDomain{.resource=resource,.owner=owner}};
  const std::array targets{
      g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(first).value(),.default_storage=g::StorageDomainId{0}},
      g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(second).value(),.default_storage=g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target=g::ExecutionTargetId{0}},
                         g::ExecutionLane{.target=g::ExecutionTargetId{1}}};
  const std::array placements{
      g::NodePlacement{before,g::ExecutionTargetId{0},g::PlacementStrength::kRequired},
      g::NodePlacement{cast,g::ExecutionTargetId{1},g::PlacementStrength::kRequired},
      g::NodePlacement{independent,g::ExecutionTargetId{1},g::PlacementStrength::kRequired},
      g::NodePlacement{after,g::ExecutionTargetId{0},g::PlacementStrength::kRequired}};
  c::ReadLease ready;
  if (owner) ready = owner.adopt_buffer(std::move(weight)).value();
  const auto binding = ready ? ready.tensor(w).value()
      : c::ConstTensorView::create(w,weight.view()).value();
  const std::array constants{g::ConstantBinding{constant,binding,
      g::ConstantPolicy::kBorrow, owner, ready}};
  auto compiled=g::compile(graph,{.storage_domains=storage,.targets=targets,.lanes=lanes},
      {.constants=constants,.node_placements=placements});
  ASSERT_TRUE(compiled)<<compiled.error().message();
  ASSERT_EQ(compiled->report.packed_constant_count(),1u);
  ASSERT_EQ(compiled->report.region_count(),1u);
  const auto segment_for=[&](g::NodeId node){
    for(std::size_t s=0;s<compiled->report.segment_count();++s){
      const auto segment=compiled->report.segment(s).value();
      for(std::size_t i=segment.step_begin;i<segment.step_begin+segment.step_count;++i){
        const auto step=compiled->report.step(i).value();
        for(std::size_t n=0;n<step.node_count;++n)
          if(compiled->report.step_node(i,n).value()==node)return s;
      }
    }
    return SIZE_MAX;
  };
  const auto a=segment_for(before),b=segment_for(cast),d=segment_for(after);
  ASSERT_NE(a,SIZE_MAX);ASSERT_NE(b,SIZE_MAX);ASSERT_NE(d,SIZE_MAX);
  const auto depends=[&](std::size_t destination,std::size_t source_segment){
    for(std::size_t i=0;i<compiled->report.segment(destination)->dependency_count;++i)
      if(compiled->report.segment_dependency(destination,i)->source_segment==source_segment)return true;
    return false;
  };
  ASSERT_NE(a,d); // A -> region -> data-only consumer must split target 0.
  ASSERT_TRUE(depends(b,a));ASSERT_TRUE(depends(d,b));
  auto negative=alloc(f);
  const auto out_desc=graph.value_description(selected).value();auto output=alloc(out_desc);
  auto invocation=compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(x,c::ConstTensorView::create(f,source.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(neg,c::TensorView::create(f,negative.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(selected,c::TensorView::create(out_desc,output.view()).value()).ok());
  auto first_queue=first.create_queue().value(),second_queue=second.create_queue().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&first_queue},
                          g::QueueBinding{g::ExecutionLaneId{1},&second_queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for(int phase=0;phase<2;++phase){
    std::array<float,18> input;for(std::size_t i=0;i<input.size();++i)input[i]=float(i+phase);
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(input))).ok());
    auto done=invocation.submit(queues);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
    std::array<float,18> actual_neg;std::array<float,6> actual;
    ASSERT_TRUE(negative.view().read(std::as_writable_bytes(std::span(actual_neg))).ok());
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for(std::size_t i=0;i<input.size();++i)EXPECT_EQ(actual_neg[i],-input[i]);
    for(std::size_t r=0;r<2;++r)for(std::size_t i=0;i<3;++i)EXPECT_EQ(actual[r*3+i],weights[r*9+8-i]);
  }
}

inline void exercise(hyle::compute::Context &context,
                     hyle::mem::MemoryResourceRef resource,
                     hyle::compute::StorageOwner owner = {}) {
  namespace c=hyle::compute; namespace g=hyle::graph; namespace m=hyle::mem;
  constexpr std::int64_t rows=2,k=37,n=7;
  const auto a_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{rows,k}).value();
  const auto w_desc=c::TensorDesc::dense(c::DataType::kBFloat16,std::array{n,k}).value();
  const auto f_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{n,k}).value();
  const auto s_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{k}).value();
  const auto y_desc=c::TensorDesc::dense(c::DataType::kFloat32,std::array{rows,n}).value();
  const auto alloc=[&](const c::TensorDesc &d){return m::Buffer::create(resource,d.required_bytes().value(),256).value();};
  // 0 immutable; 1 dynamic; 2/3 published Cast/View; 4 second BF16 consumer;
  // 5 two derived Casts share the same original constant artifact.
  for(int mode=0;mode<6;++mode) {
    SCOPED_TRACE(mode);
    auto a=alloc(a_desc),w=alloc(w_desc),s=alloc(s_desc);
    std::array y{alloc(y_desc),alloc(y_desc)},extra{alloc(f_desc),alloc(f_desc)};
    std::array y2{alloc(y_desc),alloc(y_desc)},bf_copy{alloc(w_desc),alloc(w_desc)};
    std::vector<std::uint16_t> bits(n*k);
    std::vector<float> weights(n*k),input(rows*k),scales(k,1.f);
    const auto write_weight=[&](int phase){
      for(std::size_t i=0;i<bits.size();++i){
        auto raw=std::bit_cast<std::uint32_t>(float((int((i*19+phase*13)%47)-23)*.01371));
        raw+=0x7fff+((raw>>16)&1u);bits[i]=static_cast<std::uint16_t>(raw>>16);
        weights[i]=std::bit_cast<float>(std::uint32_t(bits[i])<<16);
      }
      return w.view().write(std::as_bytes(std::span(bits)));
    };
    ASSERT_TRUE(write_weight(0).ok());ASSERT_TRUE(s.view().write(std::as_bytes(std::span(scales))).ok());
    auto b=g::GraphBuilder::create().value();
    const auto x=b.input(a_desc).value();
    const auto weight=(mode==1?b.input(w_desc):b.constant(w_desc)).value();
    const auto scale=b.constant(s_desc).value();
    const auto norm=b.add(c::RmsNorm{},x,scale).value();
    const std::array cast_inputs{weight};
    const auto cast_node=b.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),cast_inputs).value();
    const auto cast=b.result(cast_node).value();
    const auto transposed=b.add(c::View::permute(std::array<std::int32_t,2>{1,0}).value(),cast).value();
    const std::array product_inputs{norm,transposed};
    const auto product_node=b.add_node(c::Operation(c::MatMul{}),product_inputs).value();
    const auto product=b.result(product_node).value();
    ASSERT_TRUE(b.publish(product).ok());
    if(mode==2)ASSERT_TRUE(b.publish(cast).ok());
    if(mode==3)ASSERT_TRUE(b.publish(transposed).ok());
    g::ValueId second;
    if(mode==4){second=b.add(c::Copy{},weight).value();ASSERT_TRUE(b.publish(second).ok());}
    if(mode==5){
      const auto cast2=b.add(c::Cast{c::DataType::kFloat32},weight).value();
      const auto t2=b.add(c::View::permute(std::array<std::int32_t,2>{1,0}).value(),cast2).value();
      second=b.add(c::MatMul{},norm,t2).value();ASSERT_TRUE(b.publish(second).ok());
    }
    // This metadata output reads only x: its dependency on Cast is a real
    // control edge rather than the weight's existing data dependency.
    const std::array control_inputs{x};
    const auto control_node=b.add_node(c::Operation(c::View::reshape(
        c::TensorShape::create(a_desc.dimensions()).value()).value()),control_inputs).value();
    const auto control=b.result(control_node).value();
    ASSERT_TRUE(b.depends_on(control_node,cast_node).ok());
    ASSERT_TRUE(b.publish(control).ok());
    const auto logical=std::move(b).finish().value();
    c::ReadLease ready;
    if (mode!=1 && owner) ready = owner.adopt_buffer(std::move(w)).value();
    const auto weight_binding = ready ? ready.tensor(w_desc).value()
        : c::ConstTensorView::create(w_desc,w.view()).value();
    const auto scale_binding=c::ConstTensorView::create(s_desc,s.view()).value();
    std::vector<g::ConstantBinding> constants{{scale,scale_binding}};
    if(mode!=1)constants.push_back({weight,weight_binding,
        mode==4?g::ConstantPolicy::kMaterialize:g::ConstantPolicy::kBorrow,
        ready ? owner : c::StorageOwner{}, ready});
    const std::array storage{g::StorageDomain{.resource=resource,.owner=owner}};
    const std::array targets{g::ExecutionTarget{.source = hyle::graph::ExecutionSource::acquire(context).value(),.default_storage=g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target=g::ExecutionTargetId{0}}};
    auto compiled=g::compile(logical,{.storage_domains=storage,.targets=targets,.lanes=lanes},{.constants=constants});
    ASSERT_TRUE(compiled)<<compiled.error().message();
    const bool packed=mode==0||mode==4||mode==5;
    EXPECT_EQ(compiled->report.packed_constant_count(),packed?1u:0u);
    EXPECT_EQ(compiled->report.packed_constant_bytes(),packed?f_desc.required_bytes().value():0u);
    EXPECT_EQ(logical.node_dependency_count(control_node).value(),1u);
    if(mode==4){
      EXPECT_EQ(compiled->report.materialized_constant_bytes(),w_desc.required_bytes().value());
      ready.reset();
      constants.clear();
      ASSERT_TRUE(w.reset().ok()); // Both cold F32 and the BF16 consumer outlive this source.
    }
    std::array invocation{compiled->executable.create_invocation().value(),compiled->executable.create_invocation().value()};
    std::array queue{context.create_queue().value(),context.create_queue().value()};
    std::vector<float> old;
    for(int phase=0;phase<2;++phase){
      if(mode==1)ASSERT_TRUE(write_weight(phase).ok());
      for(std::size_t i=0;i<input.size();++i)input[i]=float((int((i*11+phase*7)%43)-21)*.01913);
      ASSERT_TRUE(a.view().write(std::as_bytes(std::span(input))).ok());
      auto &v=invocation[phase];
      ASSERT_TRUE(v.bind(x,c::ConstTensorView::create(a_desc,a.view()).value()).ok());
      if(mode==1)ASSERT_TRUE(v.bind(weight,weight_binding).ok());
      ASSERT_TRUE(v.bind(product,c::TensorView::create(y_desc,y[phase].view()).value()).ok());
      if(mode==2)ASSERT_TRUE(v.bind(cast,c::TensorView::create(f_desc,extra[phase].view()).value()).ok());
      if(mode==4)ASSERT_TRUE(v.bind(second,c::TensorView::create(w_desc,bf_copy[phase].view()).value()).ok());
      if(mode==5)ASSERT_TRUE(v.bind(second,c::TensorView::create(y_desc,y2[phase].view()).value()).ok());
      auto done=v.submit(queue[phase]);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());
      std::vector<float> actual(rows*n);ASSERT_TRUE(y[phase].view().read(std::as_writable_bytes(std::span(actual))).ok());
      if(phase){auto repeated=v.submit(queue[phase]);ASSERT_TRUE(repeated)<<repeated.error().message();ASSERT_TRUE(repeated->wait().ok());
        std::vector<float> again(rows*n);ASSERT_TRUE(y[phase].view().read(std::as_writable_bytes(std::span(again))).ok());EXPECT_EQ(again,actual);}
      for(std::int64_t row=0;row<rows;++row){
        double square=0;for(std::int64_t j=0;j<k;++j)square+=double(input[row*k+j])*input[row*k+j];
        const double factor=1/std::sqrt(square/k+1e-5);
        for(std::int64_t col=0;col<n;++col){double expected=0;
          for(std::int64_t j=0;j<k;++j)expected+=double(input[row*k+j])*factor*weights[col*k+j];
          EXPECT_NEAR(actual[row*n+col],expected,2e-6+std::abs(expected)*6e-6);
        }
      }
      if(mode==2){std::vector<float> out(n*k);ASSERT_TRUE(extra[phase].view().read(std::as_writable_bytes(std::span(out))).ok());EXPECT_EQ(out,weights);}
      if(mode==3){auto out=v.output(transposed);ASSERT_TRUE(out)<<out.error().message();
        EXPECT_EQ(out->description(),f_desc.transposed().value());
        std::vector<float> raw(n*k);ASSERT_TRUE(out->storage().read(std::as_writable_bytes(std::span(raw))).ok());EXPECT_EQ(raw,weights);}
      {auto out=v.output(control);ASSERT_TRUE(out)<<out.error().message();
        std::vector<float> raw(rows*k);ASSERT_TRUE(out->storage().read(std::as_writable_bytes(std::span(raw))).ok());EXPECT_EQ(raw,input);}
      if(mode==4){std::vector<std::uint16_t> out(n*k);ASSERT_TRUE(bf_copy[phase].view().read(std::as_writable_bytes(std::span(out))).ok());EXPECT_EQ(out,bits);}
      if(mode==5){std::vector<float> out(rows*n);ASSERT_TRUE(y2[phase].view().read(std::as_writable_bytes(std::span(out))).ok());EXPECT_EQ(out,actual);}
      if(!phase)old=actual;else{std::vector<float> previous(rows*n);ASSERT_TRUE(y[0].view().read(std::as_writable_bytes(std::span(previous))).ok());EXPECT_EQ(previous,old);}
    }
  }
}
} // namespace ggml_constant_cast_test
