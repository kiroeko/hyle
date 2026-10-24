#pragma once

#include "nvfp4_reference_fixture.h"

namespace ct_block128_qdq_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace storage = nvfp4_reference_test;
using Type = c::DataType;
using storage::take;
using storage::check;
using storage::dense;
using storage::bytes;
inline float bf(double x) { return storage::wide(storage::bf16(storage::f32(x))); }
inline std::uint8_t encode(float x) {
  if (std::isnan(x)) return std::uint8_t(0x7f | (std::signbit(x)?0x80:0));
  const auto value=std::min(std::abs(double(x)),448.);
  unsigned best=0;double distance=value;
  for(unsigned i=1;i<=126;++i) {
    const auto next=std::abs(value-double(storage::e4m3(std::uint8_t(i))));
    if(next<distance || (next==distance && !(i&1))) {best=i;distance=next;}
  }
  return std::uint8_t(best|(std::signbit(x)?0x80:0));
}
enum class Output { kLinear, kActivation, kWeight };
struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::int32_t> ids;
};
struct Case final {
  static constexpr std::int64_t tokens=2,depth=256,groups=2;
  bool grouped;
  Output kind;
  std::int64_t columns,experts,slots,blocks;
  c::TensorDesc activation_description=dense(Type::kBFloat16,{tokens,depth}),output_description;
  std::array<c::TensorDesc,7> descriptions;
  std::array<std::vector<std::byte>,7> payloads;
  std::vector<std::uint8_t> w;
  std::vector<std::uint16_t> scales;
  std::array<g::ValueId,7> constants{};
  g::ValueId activation{},ids{},output{};g::Graph graph;

  Case(bool expert=false,Output requested=Output::kLinear)
      :grouped(expert),kind(requested),columns(expert?128:256),experts(expert?3:1),slots(expert?3:1),blocks(columns/128) {
    descriptions[0]=grouped?dense(Type::kFloat8E4M3FN,{experts,columns,depth}):dense(Type::kFloat8E4M3FN,{columns,depth});
    descriptions[1]=grouped?dense(Type::kBFloat16,{experts,blocks,groups}):dense(Type::kBFloat16,{blocks,groups});
    descriptions[2]=dense(Type::kInt32,{columns});
    for(std::size_t i=3;i<descriptions.size();++i) descriptions[i]=dense(Type::kBFloat16,{});
    w.resize(std::size_t(experts*columns*depth));scales.resize(std::size_t(experts*blocks*groups));
    constexpr std::array<std::uint16_t,4> scale_values{0x3f81,0x3ef1,0x3f03,0x3fdd};
    for(std::int64_t e=0;e<experts;++e) for(std::int64_t b=0;b<blocks;++b) for(std::int64_t k=0;k<groups;++k)
      scales[std::size_t((e*blocks+b)*groups+k)]=scale_values[std::size_t((e*3+b+k)%4)];
    for(std::int64_t e=0;e<experts;++e) for(std::int64_t n=0;n<columns;++n) for(std::int64_t k=0;k<depth;++k)
      w[std::size_t((e*columns+n)*depth+k)]=std::uint8_t(0x38+(e*7+n*3+k*5)%8+((n+k+e)%5==0?0x80:0));
    std::vector<std::int32_t> row_ids(std::size_t(columns),0);
    for(std::int64_t n=0;n<columns;++n) row_ids[std::size_t(n)]=std::int32_t(n/128);
    const std::array<std::uint16_t,1> high{0x43e0},low{0xc3e0},epsilon{0x3c00},zero{0};
    payloads={bytes<std::uint8_t>(w),bytes<std::uint16_t>(scales),bytes<std::int32_t>(row_ids),
      bytes<std::uint16_t>(high),bytes<std::uint16_t>(low),bytes<std::uint16_t>(epsilon),bytes<std::uint16_t>(zero)};
    auto b=take(g::GraphBuilder::create(),"CT block128 builder");
    activation=take(b.input(activation_description),"original BF16 A");
    for(std::size_t i=0;i<constants.size();++i) constants[i]=take(b.constant(descriptions[i]),"original CT block constant");
    auto reshape=[&](g::ValueId v,std::initializer_list<std::int64_t> shape) {
      std::vector<g::Dimension> dimensions;for(const auto n:shape) dimensions.push_back(g::Dimension::fixed(n));
      return take(b.add(take(g::ViewSpec::reshape(dimensions),"reshape spec"),v),"metadata reshape");
    };
    g::ValueId aq{};
    if(kind!=Output::kWeight) {
    const auto grouped_a=reshape(activation,{tokens,groups,128});
    const auto absolute=take(b.add(c::Unary{c::UnaryKind::kAbs},grouped_a),"BF16 abs");
    const auto maximum=take(b.add(c::Reduce{.kind=c::ReductionKind::kMaximum,
      .axes=take(c::Axes::from(std::array<std::int32_t,1>{2}),"group reduce axis"),.keep_dimensions=true,.accumulation_type=Type::kFloat32},absolute),"BF16 group128 maximum");
    const auto raw_scale=take(b.add(c::Binary{c::BinaryKind::kDivide},maximum,constants[3]),"BF16 amax/448");
    const auto nonzero=take(b.add(c::Cast{Type::kBool},raw_scale),"scale nonzero");
    const auto scale=take(b.add(c::Select{},nonzero,raw_scale,constants[5]),"only zero scales become BF16 epsilon");
    const auto normalized=take(b.add(c::Binary{c::BinaryKind::kDivide},grouped_a,scale),"BF16 A/SA");
    // The fixed dynamic producer explicitly supplies a symmetric +0 zero point.
    // Keep both nodes: the first changes the sign of a source negative zero.
    const auto shifted=take(b.add(c::Binary{c::BinaryKind::kAdd},normalized,constants[6]),"BF16 normalized+0");
    // Public E4 Cast is SATFINITE RNE: it includes the producer clamp while
    // preserving NaN class, unlike ordinary fmin/fmax's single-NaN policy.
    const auto encoded=take(b.add(c::Cast{Type::kFloat8E4M3FN},shifted),"SATFINITE E4 RNE includes producer clamp");
    const auto widened=take(b.add(c::Cast{Type::kBFloat16},encoded),"E4 code to BF16");
    const auto centered=take(b.add(c::Binary{c::BinaryKind::kSubtract},widened,constants[6]),"BF16 code-0");
    const auto restored=take(b.add(c::Binary{c::BinaryKind::kMultiply},centered,scale),"BF16 activation QDQ");
    aq=reshape(restored,{tokens,depth});
    }
    if(kind==Output::kActivation) {
      output=aq;output_description=activation_description;
      check(b.publish(output),"publish only activation QDQ");
      graph=take(std::move(b).finish(),"finish activation-only CT graph");return;
    }
    auto raw_w=constants[0],sf=constants[1];
    if(grouped) {
      ids=take(b.input(dense(Type::kInt32,{tokens,slots})),"current expert IDs");
      const auto flat_ids=reshape(ids,{tokens*slots});
      raw_w=take(b.add(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},raw_w,flat_ids),"select original E4 expert rows before any floating decode");
      sf=take(b.add(c::Gather{.axis=0,.bounds=c::IndexBoundsPolicy::kChecked},sf,flat_ids),"select original BF16 expert scale grids");
    }
    const auto selected_sf=take(b.add(c::Gather{.axis=grouped?1:0,.bounds=c::IndexBoundsPolicy::kChecked},sf,constants[2]),"broadcast original block128 SF by logical row");
    const auto scales_view=grouped?reshape(selected_sf,{tokens*slots,columns,groups,1}):reshape(selected_sf,{columns,groups,1});
    const auto codes=take(b.add(c::Cast{Type::kBFloat16},raw_w),"selected weight codes to BF16");
    const auto codes_view=grouped?reshape(codes,{tokens*slots,columns,groups,128}):reshape(codes,{columns,groups,128});
    const auto restored_w=take(b.add(c::Binary{c::BinaryKind::kMultiply},codes_view,scales_view),"BF16 block128 weight multiply");
    const auto weight=grouped?reshape(restored_w,{tokens*slots,columns,depth}):reshape(restored_w,{columns,depth});
    if(kind==Output::kWeight) {
      output=weight;output_description=grouped?dense(Type::kBFloat16,{tokens*slots,columns,depth}):dense(Type::kBFloat16,{columns,depth});
    } else if(grouped) {
      const auto rhs=reshape(weight,{tokens,slots,columns,depth});
      const auto transposed=take(b.add(take(g::ViewSpec::permute(std::array<std::int32_t,4>{0,1,3,2}),"batched transpose"),rhs),"selected BF16 W transpose");
      const auto lhs=reshape(aq,{tokens,1,1,depth});
      const auto product=take(b.add(c::MatMul{.accumulation_type=Type::kFloat32},lhs,transposed),"ordinary BF16 batched projection with shared A");
      output=reshape(product,{tokens,slots,columns});output_description=dense(Type::kBFloat16,{tokens,slots,columns});
    } else {
      const auto rhs=take(b.add(take(g::ViewSpec::permute(std::array<std::int32_t,2>{1,0}),"dense transpose"),weight),"original BF16 W transpose");
      output=take(b.add(c::MatMul{.accumulation_type=Type::kFloat32},aq,rhs),"ordinary BF16 block128 projection");
      output_description=dense(Type::kBFloat16,{tokens,columns});
    }
    check(b.publish(output),"publish requested CT result");graph=take(std::move(b).finish(),"finish CT block graph");
  }
  Data data(unsigned phase=0) const {
    Data d;d.a.resize(std::size_t(tokens*depth),0);
    for(std::int64_t t=0;t<tokens;++t) for(std::int64_t g=0;g<groups;++g) {
      if(t==0 && g==0) {d.a[std::size_t(t*depth+g*128)]=0x8000;continue;}
      if(g==0) {
        for(const auto lane:{0,3,32,127}) {
          const auto v=lane==0?1.F:lane==3?-.5F:lane==32?.75F:-.25F;
          d.a[std::size_t(t*depth+lane)]=storage::bf16(std::ldexp(v,-20));
        }
      } else {
        const auto factor=std::ldexp(1.F,int((phase+std::size_t(t))%4));
        d.a[std::size_t(t*depth+128)]=storage::bf16(storage::wide(0x3bad)*factor);
        d.a[std::size_t(t*depth+131)]=storage::bf16(.5F*factor);
        d.a[std::size_t(t*depth+160)]=storage::bf16(.125F*factor);
        d.a[std::size_t(t*depth+255)]=storage::bf16(-.25F*factor);
      }
    }
    if(grouped) for(std::int64_t t=0;t<tokens;++t) for(std::int64_t s=0;s<slots;++s)
      d.ids.push_back(std::int32_t((std::size_t(t+s)+phase)%2)); // unselected expert2
    return d;
  }
  std::vector<std::uint16_t> activation_oracle(const Data &d,bool wrong_floor=false) const {
    auto out=d.a;
    for(std::int64_t t=0;t<tokens;++t) for(std::int64_t g=0;g<groups;++g) {
      float maximum=0;
      for(std::int64_t i=0;i<128;++i) maximum=std::max(maximum,std::abs(storage::wide(d.a[std::size_t(t*depth+g*128+i)])));
      float scale=bf(double(maximum)/448.);
      scale=wrong_floor?std::max(scale,1.F/128.F):(scale==0.F?1.F/128.F:scale);
      for(std::int64_t i=0;i<128;++i) {
        const auto index=std::size_t(t*depth+g*128+i);
        const auto norm=bf(double(storage::wide(d.a[index]))/double(scale));
        const auto shifted=bf(double(norm)+0.);
        const auto code=encode(std::clamp(shifted,-448.F,448.F));
        const auto centered=bf(double(storage::e4m3(code))-0.);
        out[index]=storage::bf16(bf(double(centered)*double(scale)));
      }
    }
    return out;
  }
  float decoded(std::int64_t e,std::int64_t n,std::int64_t k,bool omit_round=false) const {
    const auto code=w[std::size_t((e*columns+n)*depth+k)];
    const auto sf=storage::wide(scales[std::size_t((e*blocks+n/128)*groups+k/128)]);
    const auto value=storage::f32(double(storage::e4m3(code))*double(sf));
    return omit_round?value:bf(value);
  }
  std::vector<std::uint16_t> oracle(const Data &d,bool wrong_floor=false,bool omit_weight_round=false) const {
    const auto a=activation_oracle(d,wrong_floor);
    if(kind==Output::kActivation) return a;
    if(kind==Output::kWeight) {
      std::vector<std::uint16_t> y(std::size_t((grouped?tokens*slots:1)*columns*depth));
      for(std::int64_t row=0;row<(grouped?tokens*slots:1);++row) for(std::int64_t n=0;n<columns;++n) for(std::int64_t k=0;k<depth;++k)
        y[std::size_t((row*columns+n)*depth+k)]=storage::bf16(decoded(grouped?d.ids[std::size_t(row)]:0,n,k));
      return y;
    }
    std::vector<std::uint16_t> y(std::size_t(tokens*slots*columns));
    for(std::int64_t t=0;t<tokens;++t) for(std::int64_t s=0;s<slots;++s) for(std::int64_t n=0;n<columns;++n) {
      const auto e=grouped?d.ids[std::size_t(t*slots+s)]:0;double dot=0;
      for(std::int64_t k=0;k<depth;++k) dot+=double(storage::wide(a[std::size_t(t*depth+k)]))*double(decoded(e,n,k,omit_weight_round));
      y[std::size_t((t*slots+s)*columns+n)]=storage::bf16(storage::f32(dot));
    }
    return y;
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={},bool probes=false) {
  unsigned submissions=0;
  const std::array linear_cases{std::pair{false,Output::kLinear},std::pair{true,Output::kLinear}};
  const std::array probe_cases{std::pair{false,Output::kActivation},std::pair{false,Output::kWeight},std::pair{true,Output::kWeight}};
  const std::span cases=probes?std::span<const std::pair<bool,Output>>(probe_cases):std::span<const std::pair<bool,Output>>(linear_cases);
  for(const auto [grouped,kind]:cases) {
    SCOPED_TRACE(::testing::Message()<<"grouped="<<grouped<<" output="<<int(kind));Case f(grouped,kind);
    bool floor_witness=false,weight_witness=false;
    for(unsigned phase=0;phase<4;++phase) {
      const auto data=f.data(phase);
      floor_witness |= f.activation_oracle(data)!=f.activation_oracle(data,true);
    }
    for(std::int64_t e=0;e<f.experts;++e) for(std::int64_t n=0;n<f.columns;++n) for(std::int64_t k=0;k<f.depth;++k)
      weight_witness |= f.decoded(e,n,k)!=f.decoded(e,n,k,true);
    if(kind!=Output::kWeight)ASSERT_TRUE(floor_witness);
    if(kind!=Output::kActivation)ASSERT_TRUE(weight_witness);
    std::array<c::ReadLease,7> leases;std::array<g::ConstantBinding,7> constants;
    for(std::size_t i=0;i<constants.size();++i) {
      auto buffer=take(m::Buffer::create(resource,f.payloads[i].size(),256),"original constant bytes");check(write(buffer.view(),f.payloads[i]),"upload CT constants");
      leases[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned CT bytes"):take(c::ReadLease::adopt(std::move(buffer)),"CT byte lease");
      constants[i]={f.constants[i],take(leases[i].tensor(f.descriptions[i]),"original CT constant"),g::ConstantPolicy::kMaterialize,owner,owner?leases[i]:c::ReadLease{}};
    }
    auto compiled=compile(f.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
    const auto ownership=take(compiled->executable.output_ownership(),"CT output ownership");
    ASSERT_TRUE(ownership==g::OutputOwnership::kCaller || ownership==g::OutputOwnership::kInvocation);
    std::array runs{take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"CT slot0"),take(compiled->executable.create_invocation({.resources=g::ResourceOptions{}}),"CT slot1")};
    constants={};leases={};compiled=g::CompiledGraph{};
    const auto size=f.output_description.required_bytes().value();c::LeasePool pool;
    if(ownership==g::OutputOwnership::kCaller) pool=take(c::LeasePool::create(resource,2*((size+255)/256*256),{.max_blocks=2}),"two output writers");
    std::array<std::array<m::Buffer,2>,2> inputs;
    for(std::size_t slot=0;slot<2;++slot) {
      inputs[slot][0]=take(m::Buffer::create(resource,f.activation_description.required_bytes().value(),256),"CT original A");
      check(runs[slot].bind(f.activation,take(c::ConstTensorView::create(f.activation_description,inputs[slot][0].view()),"CT A view")),"bind A");
      if(grouped) {
        const auto desc=dense(Type::kInt32,{f.tokens,f.slots});inputs[slot][1]=take(m::Buffer::create(resource,desc.required_bytes().value(),256),"CT current IDs");
        check(runs[slot].bind(f.ids,take(c::ConstTensorView::create(desc,inputs[slot][1].view()),"CT ID view")),"bind IDs");
      }
      if(ownership==g::OutputOwnership::kCaller) check(runs[slot].bind(f.output,f.output_description,take(pool.allocate(size),"CT result writer")),"bind result");
      check(prepare(runs[slot],queues),"prepare ordinary CT replay");
    }
    std::array<g::GraphOutput,2> old;std::array<std::vector<std::uint16_t>,2> expected;
    for(unsigned round=0;round<2;++round) for(std::size_t slot=0;slot<2;++slot) {
      const auto d=f.data(round*2+unsigned(slot));
      if(round) {storage::expect_result(old[slot],expected[slot],read);EXPECT_FALSE(runs[slot].submit(queues));old[slot].reset();}
      check(write(inputs[slot][0].view(),std::as_bytes(std::span(d.a))),"update actual A");
      if(grouped) check(write(inputs[slot][1].view(),std::as_bytes(std::span(d.ids))),"update actual IDs");
      auto done=take(runs[slot].submit(queues),"CT submit");check(done.wait(),"CT completion");++submissions;
      old[slot]=take(runs[slot].result(f.output),"CT result");expected[slot]=f.oracle(d);storage::expect_result(old[slot],expected[slot],read);
      if(old[1-slot].valid())storage::expect_result(old[1-slot],expected[1-slot],read);
    }
    runs={};for(std::size_t i=0;i<2;++i)storage::expect_result(old[i],expected[i],read);old={};
    if(ownership==g::OutputOwnership::kCaller)EXPECT_EQ(pool.stats().live_blocks,0u);
  }
  ::testing::Test::RecordProperty("actual_submissions",submissions);
}

template<class Compile,class Write,class Read,class Prepare>
void run_checked_ids(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
    Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={}) {
  Case f(true);std::array<c::ReadLease,7> leases;std::array<g::ConstantBinding,7> constants;
  for(std::size_t i=0;i<constants.size();++i) {
    auto buffer=take(m::Buffer::create(resource,f.payloads[i].size(),256),"checked CT original constant");check(write(buffer.view(),f.payloads[i]),"upload checked CT bytes");
    leases[i]=owner?take(owner.adopt_buffer(std::move(buffer)),"owned checked CT bytes"):take(c::ReadLease::adopt(std::move(buffer)),"checked CT bytes");
    constants[i]={f.constants[i],take(leases[i].tensor(f.descriptions[i]),"checked CT constant view"),g::ConstantPolicy::kMaterialize,owner,owner?leases[i]:c::ReadLease{}};
  }
  auto compiled=compile(f.graph,g::CompileOptions{.constants=constants});ASSERT_TRUE(compiled)<<compiled.error().message();
  auto executable=compiled->executable;const auto ownership=take(executable.output_ownership(),"checked CT output ownership");
  ASSERT_TRUE(ownership==g::OutputOwnership::kCaller || ownership==g::OutputOwnership::kInvocation);
  auto run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"checked CT invocation");
  constants={};leases={};compiled=g::CompiledGraph{};
  auto a=take(m::Buffer::create(resource,f.activation_description.required_bytes().value(),256),"checked CT A");
  const auto id_description=dense(Type::kInt32,{f.tokens,f.slots});
  auto ids=take(m::Buffer::create(resource,id_description.required_bytes().value(),256),"checked CT IDs");
  const auto size=f.output_description.required_bytes().value();c::LeasePool pool;
  if(ownership==g::OutputOwnership::kCaller)pool=take(c::LeasePool::create(resource,(size+255)/256*256,{.max_blocks=1}),"checked CT result pool");
  auto data=f.data();
  const auto upload=[&] {
    check(write(a.view(),std::as_bytes(std::span(data.a))),"update checked A");
    check(write(ids.view(),std::as_bytes(std::span(data.ids))),"update checked IDs");
  };
  const auto bind=[&] {
    check(run.bind(f.activation,take(c::ConstTensorView::create(f.activation_description,a.view()),"checked A view")),"bind checked A");
    check(run.bind(f.ids,take(c::ConstTensorView::create(id_description,ids.view()),"checked ID view")),"bind checked IDs");
    if(ownership==g::OutputOwnership::kCaller)check(run.bind(f.output,f.output_description,take(pool.allocate(size),"checked result writer")),"bind checked result");
    check(prepare(run,queues),"prepare checked CT replay");
  };
  unsigned successes=0,failures=0;
  const auto success=[&] {
    upload();auto done=take(run.submit(queues),"checked CT submit");check(done.wait(),"checked CT valid completion");
    auto result=take(run.result(f.output),"checked CT result");storage::expect_result(result,f.oracle(data),read);++successes;
  };
  upload();bind();success();
  for(const auto invalid:{-1,int(f.experts)}) {
    data.ids.back()=invalid;upload();
    {auto done=run.submit(queues);ASSERT_TRUE(done)<<done.error().message();EXPECT_FALSE(done->wait().ok());EXPECT_FALSE(run.result(f.output));}
    EXPECT_FALSE(run.submit(queues))<<"failed CT invocation must be terminal";++failures;
    run={};run=take(executable.create_invocation({.resources=g::ResourceOptions{}}),"fresh CT recovery");
    data=f.data(failures);upload();bind();success();
  }
  ::testing::Test::RecordProperty("successful_submissions",successes);
  ::testing::Test::RecordProperty("expected_failed_completions",failures);
}
} // namespace ct_block128_qdq_test
