#pragma once

#include <cstring>
#include "nvfp4_reference_fixture.h"
#include "mlx_affine_bitstream.h"
#include "hyle/simd/dtype.h"

namespace mlx_orientation_test {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace scalar=nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type=c::DataType;
using Direction=c::model_ops::MlxMatmulOrientation;

template<class T> std::vector<std::byte> bytes(std::span<const T> values) {
  const auto raw=std::as_bytes(values);return {raw.begin(),raw.end()};
}
inline std::vector<std::byte> floating(Type type,std::span<const float> values) {
  if(type==Type::kFloat32)return bytes<float>(values);
  std::vector<std::uint16_t> result;result.reserve(values.size());
  for(float v:values)result.push_back(type==Type::kBFloat16?scalar::bf16(v):hyle::simd::f32_to_f16(v).bits);
  return bytes<std::uint16_t>(result);
}
struct Bank final {
  std::int64_t experts,n,k;
  unsigned bits,group;
  Type metadata;
  std::vector<std::uint32_t> codes;
  std::vector<float> scales,biases;
  std::array<c::TensorDesc,3> descriptions;
  std::array<std::vector<std::byte>,3> payloads;
  Bank(std::int64_t e,std::int64_t rows,std::int64_t columns,unsigned width,unsigned grouping,Type type)
      :experts(e),n(rows),k(columns),bits(width),group(grouping),metadata(type),
       codes(std::size_t(e*n*k)),scales(std::size_t(e*n*(k/group))),biases(scales.size()) {
    for(std::int64_t expert=0;expert<e;++expert)for(std::int64_t row=0;row<n;++row) {
      for(std::int64_t col=0;col<k;++col)
        codes[std::size_t((expert*n+row)*k+col)]=unsigned(expert*11+row*7+col*5)&((1u<<bits)-1);
      for(std::int64_t block=0;block<k/group;++block) {
        const auto i=std::size_t((expert*n+row)*(k/group)+block);
        scales[i]=std::ldexp(1.F,-int(bits)-3+int((expert+row+block)%3));
        biases[i]=-std::ldexp(float(1+(expert+row+block)%3),-int(bits)-4);
      }
    }
    // Independent single-bit writer; oracle reads original codes, not words.
    payloads={bytes<std::uint32_t>(mlx_affine_test::pack(codes,std::size_t(e*n),std::size_t(k),bits)),
        floating(type,scales),floating(type,biases)};
  }
  float value(std::int32_t e,std::int64_t row,std::int64_t col) const {
    const auto i=std::size_t((std::int64_t(e)*n+row)*(k/group)+col/group);
    return scalar::f32(double(scalar::multiply(float(codes[std::size_t((std::int64_t(e)*n+row)*k+col)]),scales[i]))+double(biases[i]));
  }
};
enum class Layout {kLinear,kShared,kZeroStride,kIndependent,kHeads};

struct Case final {
  Bank bank;
  Layout layout;
  Type activation_type,output_type;
  std::int64_t tokens,slots;
  bool grouped,both;
  g::Graph graph;
  std::vector<c::TensorDesc> input_desc,output_desc;
  std::vector<g::ValueId> inputs,outputs;
  std::array<g::ValueId,3> constants;
  Case(Bank source,Type activation,Layout kind,bool two_directions)
      :bank(std::move(source)),layout(kind),activation_type(activation),
       output_type(activation==bank.metadata?activation:Type::kFloat32),
       tokens(kind==Layout::kHeads?3:2),slots(kind==Layout::kHeads?bank.experts:3),
       grouped(kind!=Layout::kLinear),both(two_directions) {
    const auto shape=[&](Type type,std::int64_t width) {
      return grouped?dense(type,{bank.experts,bank.n,width}):dense(type,{bank.n,width});
    };
    bank.descriptions={shape(Type::kUInt32,(bank.k/32)*bank.bits),
        shape(bank.metadata,bank.k/bank.group),shape(bank.metadata,bank.k/bank.group)};
    auto builder=take(g::GraphBuilder::create(),"orientation graph");
    for(unsigned direction=0;direction<(both?2u:1u);++direction) {
      const auto reduction=direction?bank.k:bank.n;
      c::TensorDesc desc;
      const auto width=std::int64_t(c::data_type_size(activation_type));
      if(!grouped)desc=dense(activation_type,{tokens,reduction});
      else if(direction&&layout==Layout::kHeads)
        desc=take(c::TensorDesc::strided_bytes(activation_type,std::array{tokens,slots,reduction},
            std::array{slots*(reduction+64)*width,(reduction+64)*width,width}),"padded head query");
      else if(layout==Layout::kZeroStride)
        desc=take(c::TensorDesc::strided_bytes(activation_type,std::array{tokens,slots,reduction},
            std::array{(reduction+5)*width,std::int64_t{0},width}),"shared pitched A");
      else desc=dense(activation_type,{tokens,layout==Layout::kIndependent?slots:std::int64_t{1},reduction});
      input_desc.push_back(desc);inputs.push_back(take(builder.input(desc),"A input"));
    }
    if(grouped) {
      input_desc.push_back(dense(Type::kInt32,{tokens,slots}));
      inputs.push_back(take(builder.input(input_desc.back()),"head/expert IDs"));
    }
    for(unsigned i=0;i<3;++i)constants[i]=take(builder.constant(bank.descriptions[i]),"packed bank");
    for(unsigned direction=0;direction<(both?2u:1u);++direction) {
      auto a=inputs[direction];
      if(direction&&layout==Layout::kHeads)a=take(builder.add(c::Copy{},a),"explicit small padded-query Copy");
      const auto orientation=direction?Direction::kTransposeWeight:Direction::kWeightAsStored;
      const auto op=grouped?c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine,bank.bits,bank.group,orientation):
          c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine,bank.bits,bank.group,orientation);
      const auto y=grouped?take(builder.add(op,a,constants[0],constants[1],constants[2],inputs.back()),"selected packed projection"):
          take(builder.add(op,a,constants[0],constants[1],constants[2]),"packed projection");
      outputs.push_back(y);check(builder.publish(y),"publish projection");
      output_desc.push_back(grouped?dense(output_type,{tokens,slots,direction?bank.n:bank.k}):
          dense(output_type,{tokens,direction?bank.n:bank.k}));
    }
    graph=take(std::move(builder).finish(),"finish orientation graph");
  }
  float pulse(unsigned phase,std::int64_t token,std::int64_t slot,unsigned position,unsigned direction) const {
    const bool shared=grouped&&!(direction&&layout==Layout::kHeads)&&layout!=Layout::kIndependent;
    const auto actual_slot=shared?0:slot;
    const auto value=float(1+(phase+unsigned(token)+unsigned(actual_slot)+position+direction)%3)*.25F;
    return position==1?-value:value;
  }
  std::vector<std::int32_t> ids(unsigned phase) const {
    std::vector<std::int32_t> result(std::size_t(tokens*slots));
    constexpr std::array<std::int32_t,6> small{2,0,2,1,1,0};
    for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<slots;++s)
      result[std::size_t(t*slots+s)]=layout==Layout::kHeads?std::int32_t(s):
          std::int32_t((small[std::size_t(t*slots+s)]+phase)%unsigned(bank.experts));
    return result;
  }
  std::vector<std::vector<std::byte>> data(unsigned phase) const {
    std::vector<std::vector<std::byte>> result;
    for(unsigned direction=0;direction<(both?2u:1u);++direction) {
      const auto &desc=input_desc[direction];const auto strides=desc.byte_strides();
      const auto item=c::data_type_size(activation_type);
      std::vector<float> values(take(desc.required_bytes(),"input extent")/item,std::numeric_limits<float>::quiet_NaN());
      const auto reduction=desc.dimensions().back();
      const auto a_slots=grouped?desc.dimensions()[1]:1;
      const std::array<std::int64_t,3> positions{0,reduction/2,reduction-1};
      for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<a_slots;++s) {
        const auto offset=std::size_t(t*strides[0]+(grouped?s*strides[1]:0))/item;
        std::fill_n(values.begin()+static_cast<std::ptrdiff_t>(offset),std::size_t(reduction),0.F);
        for(unsigned i=0;i<3;++i)values[offset+std::size_t(positions[i])]=pulse(phase,t,s,i,direction);
      }
      result.push_back(floating(activation_type,values));
    }
    if(grouped)result.push_back(bytes<std::int32_t>(ids(phase)));
    return result;
  }
  std::vector<std::vector<std::byte>> oracle(unsigned phase) const {
    std::vector<std::vector<std::byte>> result;const auto selected=grouped?ids(phase):std::vector<std::int32_t>{};
    for(unsigned direction=0;direction<(both?2u:1u);++direction) {
      const auto reduction=direction?bank.k:bank.n, width=direction?bank.n:bank.k;
      const std::array<std::int64_t,3> positions{0,reduction/2,reduction-1};
      std::vector<float> expected;
      for(std::int64_t t=0;t<tokens;++t)for(std::int64_t s=0;s<(grouped?slots:1);++s)
        for(std::int64_t out=0;out<width;++out) {
          float value=0;const auto e=grouped?selected[std::size_t(t*slots+s)]:0;
          for(unsigned i=0;i<3;++i) {
            const auto w=direction?bank.value(e,out,positions[i]):bank.value(e,positions[i],out);
            value=scalar::f32(double(value)+double(scalar::multiply(pulse(phase,t,s,i,direction),w)));
          }
          expected.push_back(value);
        }
      result.push_back(floating(output_type,expected));
    }
    return result;
  }
};

template<class Compile,class Write> auto compile_case(Case &f,m::MemoryResourceRef resource,Compile compile,Write write,c::StorageOwner owner) {
  std::array<c::ReadLease,3> leases;std::array<g::ConstantBinding,3> constants;
  for(unsigned i=0;i<3;++i) {
    auto storage=take(m::Buffer::create(resource,f.bank.payloads[i].size(),256),"packed constant storage");
    check(write(storage.view(),f.bank.payloads[i]),"packed constant upload");
    leases[i]=owner?take(owner.adopt_buffer(std::move(storage)),"owned constant"):take(c::ReadLease::adopt(std::move(storage)),"constant lease");
    constants[i]={f.constants[i],take(leases[i].tensor(f.bank.descriptions[i]),"constant tensor"),g::ConstantPolicy::kMaterialize,owner,owner?leases[i]:c::ReadLease{}};
  }
  return take(compile(f.graph,g::CompileOptions{.constants=constants}),"compile original graph");
}
template<class Read> void expect(const g::GraphOutput &output,const std::vector<std::byte> &expected,Read read) {
  auto tensor=take(output.tensor(),"retained output");std::vector<std::byte> actual(expected.size());
  check(read(take(tensor.storage().subview(0,actual.size()),"exact output range"),actual),"read result");
  const auto mismatch=std::mismatch(actual.begin(),actual.end(),expected.begin()).first;
  EXPECT_TRUE(mismatch==actual.end())<<"first different byte="<<std::distance(actual.begin(),mismatch)<<" / "<<actual.size();
}
template<class Compile,class Write,class Read,class Prepare>
void exercise(Case f,m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner) {
  auto compiled=compile_case(f,resource,compile,write,owner);
  const auto ownership=take(compiled.executable.output_ownership(),"output ownership");
  std::array runs{take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"slot0"),
                 take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"slot1")};
  std::size_t capacity=0;for(const auto &d:f.output_desc)capacity+=2*((d.required_bytes().value()+255)/256*256);
  c::LeasePool pool;
  if(ownership==g::OutputOwnership::kCaller)
    pool=take(c::LeasePool::create(resource,capacity,{.max_blocks=std::uint32_t(2*f.outputs.size())}),"two-slot outputs");
  std::array<std::vector<m::Buffer>,2> inputs;
  for(unsigned slot=0;slot<2;++slot) {
    for(unsigned i=0;i<f.inputs.size();++i) {
      inputs[slot].push_back(take(m::Buffer::create(resource,f.input_desc[i].required_bytes().value(),256),"input storage"));
      check(runs[slot].bind(f.inputs[i],take(c::ConstTensorView::create(f.input_desc[i],inputs[slot].back().view()),"input tensor")),"bind input");
    }
    if(ownership==g::OutputOwnership::kCaller)for(unsigned i=0;i<f.outputs.size();++i)
      check(runs[slot].bind(f.outputs[i],f.output_desc[i],take(pool.allocate(f.output_desc[i].required_bytes().value()),"output writer")),"bind output");
    check(prepare(runs[slot],queues),"prepare original invocation");
  }
  compiled={};f.graph={};
  std::array<std::vector<g::GraphOutput>,2> retained;
  std::array<std::vector<std::vector<std::byte>>,2> expected;
  for(unsigned phase=0;phase<2;++phase)for(unsigned slot=0;slot<2;++slot) {
    if(phase) {
      for(unsigned i=0;i<f.outputs.size();++i)expect(retained[slot][i],expected[slot][i],read);
      EXPECT_FALSE(runs[slot].submit(queues));retained[slot].clear();
    }
    const auto payloads=f.data(phase*2+slot);
    for(unsigned i=0;i<payloads.size();++i)check(write(inputs[slot][i].view(),payloads[i]),"current input bytes");
    auto done=take(runs[slot].submit(queues),"projection submit");check(done.wait(),"projection Completion");
    expected[slot]=f.oracle(phase*2+slot);
    for(unsigned i=0;i<f.outputs.size();++i) {
      retained[slot].push_back(take(runs[slot].result(f.outputs[i]),"projection result"));
      expect(retained[slot][i],expected[slot][i],read);
      if(!retained[1-slot].empty())expect(retained[1-slot][i],expected[1-slot][i],read);
    }
  }
  runs={};for(unsigned slot=0;slot<2;++slot)for(unsigned i=0;i<f.outputs.size();++i)expect(retained[slot][i],expected[slot][i],read);
}
template<class Compile,class Write,class Read,class Prepare>
void run_dense(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  struct Shape {unsigned bits,group;std::int64_t n,k;Type a,s;};
  constexpr std::array shapes{Shape{5,32,7,96,Type::kFloat32,Type::kFloat32},
      Shape{6,128,13,256,Type::kBFloat16,Type::kFloat16},
      Shape{4,64,9,192,Type::kBFloat16,Type::kBFloat16},
      Shape{8,32,67,64,Type::kFloat16,Type::kFloat16}};
  for(const auto shape:shapes) {
    SCOPED_TRACE(::testing::Message()<<"bits="<<shape.bits<<" N="<<shape.n<<" K="<<shape.k);
    exercise(Case(Bank(1,shape.n,shape.k,shape.bits,shape.group,shape.s),shape.a,Layout::kLinear,true),resource,queues,compile,write,read,prepare,owner);
  }
  ::testing::Test::RecordProperty("successful_submissions",16);
}
template<class Compile,class Write,class Read,class Prepare>
void run_heads(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  exercise(Case(Bank(64,512,192,4,64,Type::kBFloat16),Type::kBFloat16,Layout::kHeads,true),resource,queues,compile,write,read,prepare,owner);
  for(const auto layout:{Layout::kShared,Layout::kZeroStride,Layout::kIndependent})
    exercise(Case(Bank(3,7,96,5,32,Type::kFloat16),Type::kBFloat16,layout,false),resource,queues,compile,write,read,prepare,owner);
  ::testing::Test::RecordProperty("successful_submissions",16);
  ::testing::Test::RecordProperty("actual_heads",64);
  ::testing::Test::RecordProperty("physical_N",512);
  ::testing::Test::RecordProperty("physical_K",192);
}
template<class Compile,class Write,class Read,class Prepare>
void run_failures(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  Case f(Bank(3,7,96,6,32,Type::kBFloat16),Type::kBFloat16,Layout::kIndependent,false);
  auto compiled=compile_case(f,resource,compile,write,owner);
  const auto ownership=take(compiled.executable.output_ownership(),"failure output ownership");
  for(unsigned phase=0;phase<5;++phase) {
    std::vector<m::Buffer> buffers;
    for(const auto &desc:f.input_desc)buffers.push_back(take(m::Buffer::create(resource,desc.required_bytes().value(),256),"failure input"));
    auto out=take(m::Buffer::create(resource,f.output_desc[0].required_bytes().value(),256),"failure output");
    auto run=take(compiled.executable.create_invocation(),"fresh failure/recovery invocation");
    auto payloads=f.data(phase);
    if(phase==1||phase==3) {
      auto invalid=f.ids(phase);invalid.back()=phase==1?-1:std::int32_t(f.bank.experts);
      payloads.back()=bytes<std::int32_t>(invalid);
    }
    for(unsigned i=0;i<f.inputs.size();++i) {
      check(write(buffers[i].view(),payloads[i]),"failure input write");
      check(run.bind(f.inputs[i],take(c::ConstTensorView::create(f.input_desc[i],buffers[i].view()),"failure input tensor")),"failure bind");
    }
    if(ownership==g::OutputOwnership::kCaller)
      check(run.bind(f.outputs[0],take(c::TensorView::create(f.output_desc[0],out.view()),"failure output tensor")),"failure output bind");
    check(prepare(run,queues),"failure prepare");auto done=take(run.submit(queues),"failure/recovery submit");
    const auto status=done.wait();
    if(phase==1||phase==3){EXPECT_FALSE(status.ok());EXPECT_FALSE(run.result(f.outputs[0]));}
    else {check(status,"fresh recovery");expect(take(run.result(f.outputs[0]),"recovery result"),f.oracle(phase)[0],read);}
  }
  ::testing::Test::RecordProperty("successful_submissions",3);
  ::testing::Test::RecordProperty("expected_failed_completions",2);
}
} // namespace mlx_orientation_test
