#pragma once

#include "static_fp8_reference_fixture.h"
#include "../../src/compute_packed_shared/static_fp8_linear.h"

// Generic stored-scale software equation, not an assertion about every Quark
// reader/kernel epilogue. Preserve the two original ordered scale products.
namespace fp8_outer_scale_test {
namespace f=static_fp8_reference_test;
namespace c=f::c;
namespace g=f::g;
namespace m=f::m;
using f::take;
using f::check;
using f::dense;
using f::Type;
inline constexpr std::int64_t rows=3,columns=5,depth=128;
inline constexpr c::RoutePolicyId kPackedPolicy{0x48594c4553465038ULL,1,2};
inline void selected(const g::CompileReport &report,bool expected){
  std::size_t count=0;for(std::size_t i=0;i<report.step_count();++i)
    count+=take(report.step(i),"outer-scale step").policy==kPackedPolicy;
  EXPECT_EQ(count,expected?1U:0U);
}
struct Data {
  std::vector<double> a;
  std::vector<std::uint8_t> w;
  std::vector<float> sa,sw;
};
struct Geometry { std::int64_t rows=3,columns=5,depth=128; };
enum class ScaleFormation { kExternal, kDivideWithFloor, kMultiplyReciprocal };
struct Case {
  std::int64_t rows,columns,depth;
  bool prologue;
  bool weight_first;
  ScaleFormation formation;
  std::array<c::TensorDesc,4> descriptions;
  Type output_type;
  c::TensorDesc output;
  c::TensorDesc encoded;
  explicit Case(bool dynamic_scale=false,bool reverse_scales=false):Case(Geometry{},
      dynamic_scale?ScaleFormation::kDivideWithFloor:ScaleFormation::kExternal,
      reverse_scales,reverse_scales?Type::kFloat32:Type::kBFloat16){}
  Case(Geometry geometry,ScaleFormation scale_formation,bool reverse_scales,Type result_type):
      rows(geometry.rows),columns(geometry.columns),depth(geometry.depth),
      prologue(scale_formation!=ScaleFormation::kExternal),weight_first(reverse_scales),
      formation(scale_formation),descriptions{
      dense(Type::kBFloat16,{rows,depth}),dense(Type::kFloat8E4M3FN,{columns,depth}),
      dense(Type::kFloat32,{rows,1}),scale_formation==ScaleFormation::kDivideWithFloor?
          dense(Type::kFloat32,{1,columns}):dense(Type::kFloat32,{columns})},
      output_type(result_type),output(dense(output_type,{rows,columns})),
      encoded(dense(Type::kFloat8E4M3FN,{rows,depth})){}
  static Case source_role(std::int64_t token_count){
    return Case({token_count,128,7168},ScaleFormation::kMultiplyReciprocal,true,Type::kBFloat16);
  }
  float row_maximum(const Data &d,std::int64_t row)const{
    float maximum=0;
    for(std::int64_t k=0;k<depth;++k)
      maximum=std::max(maximum,std::abs(float(d.a[std::size_t(row*depth+k)])));
    return maximum;
  }
  void form_scales(Data &d,bool force_divide=false)const{
    for(std::int64_t r=0;r<rows;++r){
      const auto maximum=row_maximum(d,r);
      if(formation==ScaleFormation::kMultiplyReciprocal&&!force_divide)
        d.sa[std::size_t(r)]=f::multiply(maximum,std::bit_cast<float>(0x3b124925U));
      else {volatile float value=(formation==ScaleFormation::kDivideWithFloor?
          std::max(maximum,1.e-10F):maximum)/448.F;d.sa[std::size_t(r)]=value;}
    }
  }
  Data data(unsigned phase)const{
    Data out;out.a.assign(std::size_t(rows*depth),0.);out.w.assign(std::size_t(columns*depth),0);
    if(formation==ScaleFormation::kMultiplyReciprocal){
      // Finite artificial code/scale bytes at a fixed primary-header role
      // geometry. No original checkpoint numerical payload is represented.
      out.sa.resize(std::size_t(rows));out.sw.resize(std::size_t(columns));
      for(std::int64_t r=0;r<rows;++r){
        const auto base=std::size_t(r*depth);
        out.a[base]=.006866455078125;out.a[base+1]=1.;out.a[base+2]=.375;
        out.a[base+32]=double(r+1)/8.+double(phase)/16.;
        out.a[base+128]=double(r+2)/4.;out.a[base+std::size_t(depth-2)]=-.5;
        out.a[base+std::size_t(depth-1)]=double(r?896:450)*double(1+phase%2);
      }
      constexpr std::array<std::int64_t,6> lanes{0,1,2,32,128,-2};
      for(std::int64_t n=0;n<columns;++n){
        const auto selected=lanes[std::size_t(n)%lanes.size()];
        const auto k=n==columns-1?depth-1:selected<0?depth+selected:selected;
        out.w[std::size_t(n*depth+k)]=n<3||n%2==0?0x38U:0xb8U;
        out.sw[std::size_t(n)]=n==0?1.F:n==1?1.01171875F:n==2?1.25F:
            .5F+float(n%9)/8.F;
        if(phase>=2)out.sw[std::size_t(n)]=f::multiply(out.sw[std::size_t(n)],n%2?2.F:.5F);
      }
      form_scales(out);return out;
    }
    out.sa={1.00390625F,.5F,2.F};
    out.sw={1.00390625F,-1.125F,1.75F,.125F,std::bit_cast<float>(0x3f812345U)};
    for(std::int64_t r=0;r<rows;++r){
      for(std::int64_t k=0;k<12;++k)out.a[std::size_t(r*depth+k)]=f::round(Type::kBFloat16,
          double((k*3+r*5+std::int64_t(phase)*7)%13-5)/8.);
      out.a[std::size_t(r*depth)]=r==0?1.:double(r+1)/2.;
      out.a[std::size_t(r*depth+64)]=f::round(Type::kBFloat16,double(r+1)/8.+double(phase)/16.);
      out.a[std::size_t(r*depth+127)]=double(448*(r+1))*double(1+phase%2);
      if(!prologue)out.sa[std::size_t(r)]=f::multiply(out.sa[std::size_t(r)],phase%2?2.F:1.F);
    }
    out.a[4]=.375; // Exact F32 scale-order witness at output row0/column4.
    for(std::int64_t n=0;n<columns;++n){
      out.w[std::size_t(n*depth+n)]=n%2?0xb8U:0x38U;
      if(n&&n!=4){out.w[std::size_t(n*depth+64)]=0x30U;out.w[std::size_t(n*depth+127)]=0x28U;}
      if(phase>=2)out.sw[std::size_t(n)]=f::multiply(out.sw[std::size_t(n)],n%2?2.F:.5F);
    }
    if(prologue)form_scales(out);
    return out;
  }
  std::array<std::vector<std::byte>,4> payloads(const Data &d)const{
    return {f::floating(Type::kBFloat16,d.a),f::bytes(d.w),f::bytes(d.sa),f::bytes(d.sw)};
  }
  std::vector<std::uint8_t> codes(const Data &d,bool wrong_row=false)const{
    std::vector<std::uint8_t> out(d.a.size());
    for(std::int64_t r=0;r<rows;++r){const auto inverse=f::reciprocal(d.sa[wrong_row?0:std::size_t(r)]);
      for(std::int64_t k=0;k<depth;++k){const auto i=std::size_t(r*depth+k);out[i]=f::encode_e4m3(f::multiply(float(d.a[i]),inverse));}}
    return out;
  }
  std::vector<double> oracle(const Data &d,bool wrong_row=false,bool wrong_column=false,bool operand_bf16=false,bool reverse_order=false)const{
    const auto q=codes(d,wrong_row);std::vector<double> out(std::size_t(rows*columns));
    for(std::int64_t r=0;r<rows;++r)for(std::int64_t n=0;n<columns;++n){
      const auto sa=d.sa[wrong_row?0:std::size_t(r)],sw=d.sw[wrong_column?0:std::size_t(n)];float sum=0;
      for(std::int64_t k=0;k<depth;++k){
        auto a=f::e4m3(q[std::size_t(r*depth+k)]),w=f::e4m3(d.w[std::size_t(n*depth+k)]);
        if(operand_bf16){a=float(f::round(Type::kBFloat16,f::multiply(a,sa)));w=float(f::round(Type::kBFloat16,f::multiply(w,sw)));}
        volatile float term=a*w;volatile float next=sum+term;sum=next;
      }
      const bool wf=weight_first!=reverse_order;
      out[std::size_t(r*columns+n)]=f::round(output_type,operand_bf16?sum:
          f::multiply(f::multiply(sum,wf?sw:sa),wf?sa:sw));
    }
    return out;
  }
};
struct Logical {
  std::array<g::ValueId,4> ports{};g::ValueId output{},encoded{};g::Graph graph;
  explicit Logical(const Case &test,bool escape=false){
    auto b=take(g::GraphBuilder::create(),"outer-scale ordinary graph");
    ports[0]=take(b.input(test.descriptions[0]),"original BF16 A");
    ports[1]=take(b.constant(test.descriptions[1]),"original E4 bank");
    ports[3]=take(b.input(test.descriptions[3]),"current column scales");
    const auto number=[&](float value){return take(b.add(c::Fill{
        .output_shape=take(c::TensorShape::create(std::array<std::int64_t,1>{1}),"literal shape"),
        .output_type=Type::kFloat32,.value=c::ScalarValue(value)}),"F32 prologue literal");};
    if(test.formation==ScaleFormation::kMultiplyReciprocal){
      // BF16 abs is exact in this declared finite domain. Widen that result
      // before the F32 reduction, keeping the contracted A Cast single-reader.
      const auto absolute=take(b.add(c::Unary{c::UnaryKind::kAbs},ports[0]),"finite BF16 abs");
      const auto wide=take(b.add(c::Cast{Type::kFloat32},absolute),"F32 absolute values");
      const auto maximum=take(b.add(c::Reduce{.kind=c::ReductionKind::kMaximum,
          .axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"row reduction axis"),
          .keep_dimensions=true,.accumulation_type=Type::kFloat32},wide),"F32 row maximum");
      ports[2]=take(b.add(c::Binary{c::BinaryKind::kMultiply},maximum,
          number(std::bit_cast<float>(0x3b124925U))),"no-floor F32 multiply-reciprocal SA");
    }else if(test.prologue){
      const auto absolute=take(b.add(c::Unary{c::UnaryKind::kAbs},ports[0]),"finite BF16 abs");
      const auto maximum=take(b.add(c::Reduce{.kind=c::ReductionKind::kMaximum,
          .axes=take(c::Axes::from(std::array<std::int32_t,1>{1}),"row reduction axis"),
          .keep_dimensions=true,.accumulation_type=Type::kFloat32},absolute),"exact BF16 row maximum");
      const auto wide=take(b.add(c::Cast{Type::kFloat32},maximum),"F32 maximum");
      const auto bounded=take(b.add(c::Binary{c::BinaryKind::kMaximum},wide,number(1.e-10F)),"declared F32 floor");
      ports[2]=take(b.add(c::Binary{c::BinaryKind::kDivide},bounded,number(448.F)),"ordinary dynamic row SA");
    }else ports[2]=take(b.input(test.descriptions[2]),"current row scales");
    const auto a=take(b.add(c::Cast{Type::kFloat32},ports[0]),"original A cast");
    const auto one=take(b.add(c::Fill{.output_shape=take(c::TensorShape::create(test.descriptions[2].dimensions()),"SA one shape"),
        .output_type=Type::kFloat32,.value=c::ScalarValue(1.F)}),"one with exact SA shape");
    const auto inverse=take(b.add(c::Binary{c::BinaryKind::kDivide},one,ports[2]),"row reciprocal");
    const auto normalized=take(b.add(c::Binary{c::BinaryKind::kMultiply},a,inverse),"row normalized F32 A");
    encoded=take(b.add(c::Cast{Type::kFloat8E4M3FN},normalized),"SATFINITE E4 A");
    const auto qa=take(b.add(c::Cast{Type::kFloat32},encoded),"code-domain A");
    const auto qw=take(b.add(c::Cast{Type::kFloat32},ports[1]),"code-domain W");
    const auto wt=take(b.add(take(c::View::permute(std::array<std::int32_t,2>{1,0}),"weight transpose spec"),qw),"code W transpose");
    auto y=take(b.add(c::MatMul{.accumulation_type=Type::kFloat32},qa,wt),"F32 code dot");
    y=take(b.add(c::Binary{c::BinaryKind::kMultiply},y,ports[test.weight_first?3:2]),"first explicit scale product");
    y=take(b.add(c::Binary{c::BinaryKind::kMultiply},y,ports[test.weight_first?2:3]),"second explicit scale product");
    output=take(b.add(c::Cast{test.output_type},y),"final declared output");
    if(escape)check(b.publish(encoded),"publish original activation codes");
    check(b.publish(output),"publish outer-scale result");graph=take(std::move(b).finish(),"outer-scale graph");
  }
};
inline void admission(){
  using Op=c::packed::internal::StaticFloat8Linear;Case test;
  std::array<c::TensorDesc,1> output;
  EXPECT_TRUE(Op{}.infer_output_descriptions(test.descriptions,output).ok());
  auto invalid=test.descriptions;invalid[2]=dense(Type::kFloat32,{rows});
  EXPECT_FALSE(Op{}.infer_output_descriptions(invalid,output).ok());
  invalid=test.descriptions;invalid[3]=dense(Type::kFloat32,{columns,1});
  EXPECT_FALSE(Op{}.infer_output_descriptions(invalid,output).ok());
  invalid=test.descriptions;invalid[2]=dense(Type::kFloat32,{1,rows});
  EXPECT_FALSE(Op{}.infer_output_descriptions(invalid,output).ok());
  invalid=test.descriptions;invalid[3]=dense(Type::kFloat32,{1,columns});
  EXPECT_TRUE(Op{}.infer_output_descriptions(invalid,output).ok());
  Op unknown;unknown.scale_order=static_cast<c::packed::internal::StaticFloat8ScaleOrder>(255);
  EXPECT_FALSE(unknown.infer_output_descriptions(test.descriptions,output).ok());
}
inline void witnesses(){
  Case test;const auto d=test.data(0);const auto expected=test.oracle(d);
  EXPECT_NE(expected,test.oracle(d,true));EXPECT_NE(expected,test.oracle(d,false,true));
  EXPECT_NE(expected,test.oracle(d,false,false,true));
  EXPECT_EQ(half_test::encode<f::BF16>(expected[0]).bits,0x3f81U);
  EXPECT_EQ(half_test::encode<f::BF16>(test.oracle(d,false,false,true)[0]).bits,0x3f80U);
  Case ordered(false,true);const auto data=ordered.data(0);const auto correct=ordered.oracle(data);
  const auto wrong=ordered.oracle(data,false,false,false,true);EXPECT_NE(correct,wrong);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(float(correct[4])),0x3ec2769dU);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(float(wrong[4])),0x3ec2769cU);
}
inline void source_role_witnesses(){
  auto test=Case::source_role(2);const auto data=test.data(0);const auto expected=test.oracle(data);
  auto divided=data;test.form_scales(divided,true);const auto wrong_divide=test.oracle(divided);
  const auto wrong_operands=test.oracle(data,false,false,true);
  const auto wrong_row=test.oracle(data,true);const auto wrong_column=test.oracle(data,false,true);
  EXPECT_NE(expected,wrong_divide);EXPECT_NE(expected,wrong_operands);
  EXPECT_NE(expected,wrong_row);EXPECT_NE(expected,wrong_column);
  const auto code=[](double x){return half_test::encode<f::BF16>(x).bits;};
  EXPECT_EQ(code(expected[0]),0x3bc1U);EXPECT_EQ(code(wrong_divide[0]),0x3c01U);
  EXPECT_EQ(code(expected[1]),0x3f82U);EXPECT_EQ(code(wrong_operands[1]),0x3f83U);
  EXPECT_EQ(code(expected[std::size_t(test.columns+2)]),0x3ef0U);
  EXPECT_EQ(code(wrong_row[std::size_t(test.columns+2)]),0x3ef1U);
  EXPECT_EQ(code(expected[2]),0x3ef1U);EXPECT_EQ(code(wrong_column[2]),0x3ec1U);
}
template<class Compile,class Write,class Read,class Prepare>
void run_case(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,const Case &test,c::StorageOwner owner={}){
  Logical logical(test);const auto first=test.payloads(test.data(0));
  auto bank=take(m::Buffer::create(resource,first[1].size(),256),"original immutable E4 bank");check(write(bank.view(),first[1]),"write original bank");
  auto ready=owner?take(owner.adopt_buffer(std::move(bank)),"owned E4 source"):take(c::ReadLease::adopt(std::move(bank)),"ready original E4 source");
  std::array constants{g::ConstantBinding{logical.ports[1],take(ready.tensor(test.descriptions[1]),"original bank view"),g::ConstantPolicy::kMaterialize,owner,owner?ready:c::ReadLease{}}};
  auto compiled=take(compile(logical.graph,g::CompileOptions{.constants=constants}),"compile bounded outer-scale graph");selected(compiled.report,true);
  std::array runs{take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"outer slot0"),take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"outer slot1")};
  const bool caller=take(compiled.executable.output_ownership(),"outer output ownership")==g::OutputOwnership::kCaller;
  constants={};ready={};compiled={};logical.graph={};
  c::LeasePool pool;const auto bytes=test.output.required_bytes().value();
  if(caller)pool=take(c::LeasePool::create(resource,2*((bytes+255)/256*256),{.max_blocks=2}),"two original result blocks");
  const std::vector<std::size_t> dynamic=test.prologue?std::vector<std::size_t>{0,3}:std::vector<std::size_t>{0,2,3};
  std::array<std::vector<m::Buffer>,2> input;
  for(std::size_t slot=0;slot<2;++slot){
    for(auto port:dynamic){auto buffer=take(m::Buffer::create(resource,first[port].size(),256),"current dynamic outer port");check(write(buffer.view(),first[port]),"initial dynamic bytes");check(runs[slot].bind(logical.ports[port],take(c::ConstTensorView::create(test.descriptions[port],buffer.view()),"typed dynamic view")),"bind once");input[slot].push_back(std::move(buffer));}
    if(caller)check(runs[slot].bind(logical.output,test.output,take(pool.allocate(bytes),"outer writer")),"bind output");
    check(prepare(runs[slot],queues),"prepare outer-scale replay");
  }
  std::array<g::GraphOutput,2> retained;std::array<std::vector<double>,2> old;
  for(unsigned repeat=0;repeat<2;++repeat)for(std::size_t slot=0;slot<2;++slot){
    if(repeat){f::expect_result(retained[slot],test.output_type,old[slot],read);EXPECT_FALSE(runs[slot].submit(queues));retained[slot].reset();}
    const auto data=test.data(repeat*2+static_cast<unsigned>(slot));const auto payload=test.payloads(data);
    for(std::size_t i=0;i<dynamic.size();++i)check(write(input[slot][i].view(),payload[dynamic[i]]),"update row/channel values");
    auto done=take(runs[slot].submit(queues),"outer-scale submit");check(done.wait(),"outer completion");
    retained[slot]=take(runs[slot].result(logical.output),"retained outer result");old[slot]=test.oracle(data);f::expect_result(retained[slot],test.output_type,old[slot],read);
    if(retained[1-slot].valid())f::expect_result(retained[1-slot],test.output_type,old[1-slot],read);
  }
  runs={};pool={};for(std::size_t slot=0;slot<2;++slot)f::expect_result(retained[slot],test.output_type,old[slot],read);
  ::testing::Test::RecordProperty("successful_submissions",4);
  ::testing::Test::RecordProperty("scope",test.formation==ScaleFormation::kMultiplyReciprocal?
      "fixed-header N128/K7168 finite artificial E4/F32 bytes;no-floor maximum*F32reciprocal448;strict reciprocal;SW then SA;BF16;no checkpoint payload/AMD rcpf bit claim":
      test.prologue?"generic dynamic row SA prologue;SW[1,N];SA then SW;BF16;not deployed Quark math":"explicit SA[M,1]/SW[N];SW then SA;F32 order witness");
  ::testing::Test::RecordProperty("scale_order_counterfactual", "F32 row0col4: SW->SA 0x3ec2769d,SA->SW 0x3ec2769c");
}
template<class Compile,class Write,class Read,class Prepare>
void run(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,bool prologue,c::StorageOwner owner={}){
  admission();witnesses();run_case(resource,queues,compile,write,read,prepare,Case(prologue,!prologue),owner);
}
template<class Compile,class Write,class Read,class Prepare>
void run_lifetimes(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  run(resource,queues,compile,write,read,prepare,false,owner);
}
template<class Compile,class Write,class Read,class Prepare>
void run_prologue(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  run(resource,queues,compile,write,read,prepare,true,owner);
}
template<class Compile,class Write,class Read,class Prepare>
void run_source_role(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  source_role_witnesses();
  for(const auto token_count:std::array<std::int64_t,2>{1,2})
    run_case(resource,queues,compile,write,read,prepare,Case::source_role(token_count),owner);
  ::testing::Test::RecordProperty("successful_submissions",8);
  ::testing::Test::RecordProperty("source_role_shapes","M1/M2,N128,K7168;layer0 f_a_proj fixed-header geometry");
  ::testing::Test::RecordProperty("source_role_counterfactuals","scale formation,early BF16 operands,wrong row,wrong column all change final BF16");
}
template<class Compile,class Write,class Read,class Prepare>
void run_escape(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}){
  Case test(false,true);Logical logical(test,true);const auto first=test.payloads(test.data(0));
  auto bank=take(m::Buffer::create(resource,first[1].size(),256),"escape original bank");check(write(bank.view(),first[1]),"escape bank bytes");
  auto ready=owner?take(owner.adopt_buffer(std::move(bank)),"owned escape bank"):take(c::ReadLease::adopt(std::move(bank)),"escape bank lease");
  std::array constants{g::ConstantBinding{logical.ports[1],take(ready.tensor(test.descriptions[1]),"escape typed bank"),g::ConstantPolicy::kMaterialize,owner,owner?ready:c::ReadLease{}}};
  auto compiled=take(compile(logical.graph,g::CompileOptions{.constants=constants}),"compile published code fallback");selected(compiled.report,false);
  auto slot=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"escape original invocation");std::array<m::Buffer,3> buffers;
  constexpr std::array<std::size_t,3> dynamic{0,2,3};
  for(std::size_t i=0;i<dynamic.size();++i){const auto port=dynamic[i];buffers[i]=take(m::Buffer::create(resource,first[port].size(),256),"escape dynamic bytes");check(write(buffers[i].view(),first[port]),"escape initial bytes");check(slot.bind(logical.ports[port],take(c::ConstTensorView::create(test.descriptions[port],buffers[i].view()),"escape typed dynamic")),"escape bind once");}
  std::array<c::LeasePool,2> pools;const std::array ports{logical.encoded,logical.output};const std::array desc{test.encoded,test.output};
  for(std::size_t i=0;i<2;++i)if(take(compiled.executable.output_ownership(i),"escape ownership")==g::OutputOwnership::kCaller){const auto n=desc[i].required_bytes().value();pools[i]=take(c::LeasePool::create(resource,(n+255)/256*256,{.max_blocks=1}),"escape output pool");check(slot.bind(ports[i],desc[i],take(pools[i].allocate(n),"escape writer")),"bind actual escape output");}
  check(prepare(slot,queues),"prepare ordinary escaped graph");compiled={};constants={};ready={};logical.graph={};
  for(unsigned phase=0;phase<2;++phase){const auto data=test.data(phase);const auto payload=test.payloads(data);for(std::size_t i=0;i<dynamic.size();++i)check(write(buffers[i].view(),payload[dynamic[i]]),"escape current bytes");auto done=take(slot.submit(queues),"escape submit");check(done.wait(),"escape completion");auto y=take(slot.result(logical.output),"ordinary final result");f::expect_result(y,test.output_type,test.oracle(data),read);auto aq=take(slot.result(logical.encoded),"ordinary encoded A result");const auto tensor=take(aq.tensor(),"actual code tensor");std::vector<std::uint8_t> actual(std::size_t(test.rows*test.depth));check(read(tensor.storage(),std::as_writable_bytes(std::span(actual))),"read original E4 codes");EXPECT_EQ(actual,test.codes(data));}
  ::testing::Test::RecordProperty("successful_submissions",2);
}
} // namespace fp8_outer_scale_test
