#pragma once

#include "original_readonly_role.h"
#include "model_plane_resident_weights.h"
#include "local_nvfp4_reference_fixture.h"
#include "ct_block128_qdq_fixture.h"
#include "qwen_nvfp4_original_reference.h"

// Ordinary source composition for the explicit CT46196ce BF16 reader. The
// attention/GDN and post-attention-MLP inputs are separate semantic boundaries;
// a128-wide projection is not a hidden-width input. This is not a full layer.
// Caller authenticates original bytes; fixture identity never claims them.
namespace ct18_selected_source_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;
namespace named=original_named_role_source;namespace supply=model_plane_resident_weights;
namespace local=local_nvfp4_reference_test;namespace scalar=nvfp4_reference_test;
namespace ct=ct_block128_qdq_test;namespace math=hyle::inference::ct_block128;
using Type=c::DataType;using scalar::take;using scalar::check;using scalar::dense;
inline constexpr std::string_view repository="RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8";
inline constexpr std::string_view revision="b94577f02dd4bcb722aad2c60f8117032411dfb5";
inline constexpr std::string_view file="model-00001-of-00070.safetensors";
inline constexpr named::Identity original_identity{repository,revision,"CT46196ce-BF16-block128-and-local16",file};
inline constexpr named::Identity fixture_identity{"hyle-fixture/ct18-selected-source",
    "0000000000000000000000000000000000000000","generated-original-shapes","not-publisher-values"};
inline constexpr std::int64_t hidden=8192,columns=128,full_expert_rows=2048,rows=3,domain=512,capacity=2;
struct Spec final {
  std::string_view name;Type type;std::array<std::int64_t,4> shape;std::size_t rank;
  std::uint64_t begin,end;
  c::TensorDesc description()const{return take(c::TensorDesc::dense(type,std::span(shape).first(rank)),"CT18 exact original role");}
};
inline constexpr std::array<Spec,2> ct_specs{{
    {"model.layers.0.linear_attn.in_proj_a.weight",Type::kFloat8E4M3FN,{128,8192,0,0},2,4085750224ULL,4086798800ULL},
    {"model.layers.0.linear_attn.in_proj_a.weight_scale",Type::kBFloat16,{1,64,0,0},2,4068668368ULL,4068668496ULL}
}};
struct ExpertSpec final{std::int32_t global;std::array<Spec,4> roles;};
inline constexpr std::array<ExpertSpec,4> expert_specs{{
    {0, {{{"model.layers.0.mlp.experts.0.gate_proj.weight_packed",Type::kUInt8,{2048,4096,0,0},2,6641129936ULL,6649518544ULL},
      {"model.layers.0.mlp.experts.0.gate_proj.weight_scale",Type::kFloat8E4M3FN,{2048,512,0,0},2,4525103568ULL,4526152144ULL},
      {"model.layers.0.mlp.experts.0.gate_proj.weight_global_scale",Type::kFloat32,{1,0,0,0},1,12ULL,16ULL},
      {"model.layers.0.mlp.experts.0.gate_proj.input_global_scale",Type::kFloat32,{1,0,0,0},1,8ULL,12ULL}}}},
    {1, {{{"model.layers.0.mlp.experts.1.gate_proj.weight_packed",Type::kUInt8,{2048,4096,0,0},2,6666295760ULL,6674684368ULL},
      {"model.layers.0.mlp.experts.1.gate_proj.weight_scale",Type::kFloat8E4M3FN,{2048,512,0,0},2,4528249296ULL,4529297872ULL},
      {"model.layers.0.mlp.experts.1.gate_proj.weight_global_scale",Type::kFloat32,{1,0,0,0},1,36ULL,40ULL},
      {"model.layers.0.mlp.experts.1.gate_proj.input_global_scale",Type::kFloat32,{1,0,0,0},1,32ULL,36ULL}}}},
    {2, {{{"model.layers.0.mlp.experts.2.gate_proj.weight_packed",Type::kUInt8,{2048,4096,0,0},2,9459702224ULL,9468090832ULL},
      {"model.layers.0.mlp.experts.2.gate_proj.weight_scale",Type::kFloat8E4M3FN,{2048,512,0,0},2,4877425104ULL,4878473680ULL},
      {"model.layers.0.mlp.experts.2.gate_proj.weight_global_scale",Type::kFloat32,{1,0,0,0},1,2700ULL,2704ULL},
      {"model.layers.0.mlp.experts.2.gate_proj.input_global_scale",Type::kFloat32,{1,0,0,0},1,2696ULL,2700ULL}}}},
    {3, {{{"model.layers.0.mlp.experts.3.gate_proj.weight_packed",Type::kUInt8,{2048,4096,0,0},2,12253108688ULL,12261497296ULL},
      {"model.layers.0.mlp.experts.3.gate_proj.weight_scale",Type::kFloat8E4M3FN,{2048,512,0,0},2,5226600912ULL,5227649488ULL},
      {"model.layers.0.mlp.experts.3.gate_proj.weight_global_scale",Type::kFloat32,{1,0,0,0},1,5364ULL,5368ULL},
      {"model.layers.0.mlp.experts.3.gate_proj.input_global_scale",Type::kFloat32,{1,0,0,0},1,5360ULL,5364ULL}}}}
}};
struct ExternalRole final{named::ReadonlyRole ready;std::uint64_t begin=0,end=0;};
struct ExternalExpert final{std::int32_t global=-1;std::array<ExternalRole,4> roles;};
struct Selection final{std::vector<std::int32_t> globals,map;};
inline Selection selection(unsigned phase) {
  Selection s;s.globals={std::int32_t(phase*2),std::int32_t(phase*2+1)};
  s.map.assign(std::size_t(domain),-1);
  for(std::size_t i=0;i<s.globals.size();++i)s.map[std::size_t(s.globals[i])]=std::int32_t(i);
  return s;
}
inline std::array<c::TensorDesc,4> plane_descriptions() {
  return {dense(Type::kUInt8,{rows,hidden/2}),dense(Type::kFloat8E4M3FN,{rows,hidden/16}),
      dense(Type::kFloat32,{1}),dense(Type::kFloat32,{1})};
}
inline bool same_identity(const named::Identity *a,const named::Identity &b) {
  return a&&a->repository==b.repository&&a->revision==b.revision&&a->variant==b.variant&&a->file==b.file;
}
inline hyle::Status validate(const ExternalRole &v,const Spec &s,const named::Identity &identity) {
  if(!same_identity(v.ready.identity,identity)||v.ready.name!=s.name||v.ready.description!=s.description()||
      !v.ready.owner||!v.ready.lease.valid()||v.ready.lease.size()!=s.end-s.begin||v.begin!=s.begin||v.end!=s.end)
    return hyle::Status::InvalidArgument("CT18 original role identity/span/type differs");
  HYLE_ASSIGN_OR_RETURN(auto tensor,v.ready.lease.tensor(v.ready.description));
  if(tensor.storage().resource()!=v.ready.owner.ref())
    return hyle::Status::InvalidArgument("CT18 original lease authority differs from its actual owner");
  return {};
}
class Source final {
 public:
  Source()=default;Source(Source&&)=default;Source&operator=(Source&&)=delete;
  Source(const Source&)=delete;Source&operator=(const Source&)=delete;
  static hyle::Result<Source> accept(const std::array<ExternalExpert,2> &experts) {
    const auto &identity=same_identity(experts[0].roles[0].ready.identity,original_identity)?original_identity:fixture_identity;
    for(std::size_t i=0;i<experts.size();++i) {
      const auto global=experts[i].global;
      if(global<0||global>=std::int32_t(expert_specs.size())||(i&&experts[i-1].global>=global))
        return std::unexpected(hyle::Status::InvalidArgument("CT18 finite original experts must be cached/sorted/distinct"));
      for(std::size_t p=0;p<4;++p)HYLE_RETURN_IF_ERROR(validate(experts[i].roles[p],expert_specs[std::size_t(global)].roles[p],identity));
    }
    Source result;result.experts_=experts;
    const auto descriptions=plane_descriptions();
    for(std::size_t i=0;i<experts.size();++i)for(std::size_t p=0;p<4;++p) {
      HYLE_ASSIGN_OR_RETURN(auto view,result.experts_[i].roles[p].ready.lease.view());
      HYLE_ASSIGN_OR_RETURN(auto bytes,descriptions[p].required_bytes());
      HYLE_ASSIGN_OR_RETURN(view,view.subview(0,bytes)); // Explicit first3 original rows, or the complete original F32[1].
      HYLE_ASSIGN_OR_RETURN(result.maps_[i][p],view.map_host());
      HYLE_RETURN_IF_ERROR(result.maps_[i][p].invalidate());
    }
    return result;
  }
  std::span<const std::byte> plane_bytes(std::size_t plane,std::int32_t global)const {
    if(plane>=4)throw std::runtime_error("CT18 original plane is not a quartet member");
    for(std::size_t i=0;i<experts_.size();++i)if(experts_[i].global==global)return maps_[i][plane].bytes();
    throw std::runtime_error("CT18 selected source expert was not supplied");
  }
  hyle::Status close() {
    for(auto &expert:maps_)for(auto &mapping:expert)if(mapping.valid())HYLE_RETURN_IF_ERROR(mapping.close());
    experts_={};return {};
  }
 private:
  std::array<ExternalExpert,2> experts_{};
  std::array<std::array<m::HostMapping,4>,2> maps_{};
};

struct Case final {
  g::Graph graph;std::array<g::ConstantBinding,6> constants;
  g::ValueId attention{},mlp{},ids{},ct_output{},expert_output{};std::array<g::ValueId,5> banks;
  c::TensorDesc attention_desc=dense(Type::kBFloat16,{1,hidden}),mlp_desc=dense(Type::kBFloat16,{1,1,hidden}),
      ids_desc=dense(Type::kInt32,{1,2}),ct_output_desc=dense(Type::kBFloat16,{1,columns}),
      expert_output_desc=dense(Type::kBFloat16,{1,2,rows});
  Case(const std::array<ExternalRole,2> &source,const supply::Layout &layout) {
    const auto &identity=same_identity(source[0].ready.identity,original_identity)?original_identity:fixture_identity;
    // Validate both actual readonly roles before installing either binding.
    for(std::size_t i=0;i<source.size();++i)check(validate(source[i],ct_specs[i],identity),"CT18 paired original-role validation");
    auto b=take(g::GraphBuilder::create(),"CT18 finite ordinary graph");
    attention=take(b.input(attention_desc),"actual GDN input boundary");
    mlp=take(b.input(mlp_desc),"actual separate post-attention MLP input boundary");
    ids=take(b.input(ids_desc),"actual original expert IDs");
    for(std::size_t p=0;p<4;++p)banks[p]=take(b.input(layout.resident[p]),"same-generation original local quartet plane");
    banks[4]=take(b.input(layout.map_description()),"same-generation original remap");
    std::array<math::Operand,2> parameters;
    for(std::size_t i=0;i<source.size();++i) {
      const auto value=take(b.constant(source[i].ready.description),"original E4/BF16 constant");
      constants[i]=take(source[i].ready.binding(value),"real original CT lease binding");parameters[i]={value,source[i].ready.description};
    }
    const std::array desc{dense(Type::kInt32,{columns}),dense(Type::kBFloat16,{}),dense(Type::kBFloat16,{}),dense(Type::kBFloat16,{})};
    const std::array<std::int32_t,std::size_t(columns)> block_rows{};
    const std::array<std::uint16_t,1> limit{0x43e0},epsilon{0x3c00},positive_zero{0};
    const std::array<std::span<const std::byte>,4> bytes{std::as_bytes(std::span(block_rows)),std::as_bytes(std::span(limit)),
        std::as_bytes(std::span(epsilon)),std::as_bytes(std::span(positive_zero))};
    std::array<math::Operand,4> literals;
    for(std::size_t i=0;i<literals.size();++i) {
      const auto value=take(b.constant(desc[i]),"original CT small literal");
      auto buffer=take(m::Buffer::create(m::system_resource(),bytes[i].size(),256),"CT18 literal backing");
      check(buffer.view().write(bytes[i]),"CT18 original literal bytes");auto lease=take(c::ReadLease::adopt(std::move(buffer)),"CT18 original literal lease");
      constants[i+2]={value,take(lease.tensor(desc[i]),"CT18 literal tensor"),g::ConstantPolicy::kMaterialize,{},lease};
      literals[i]={value,desc[i]};
    }
    const auto projected=take(math::append_projection(b,{attention,attention_desc},parameters[0],parameters[1],literals[0],
        {literals[1],literals[2],literals[3]}),"same shared BF16 block128 math");
    // The shared terminal reshape borrows Graph storage. This consumer
    // explicitly materializes its256-byte escaped old-result lease.
    ct_output=take(b.add(c::Copy{},projected.value),"CT18 independent terminal result");
    const auto local_ids=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},banks[4],ids),"checked original global-local map");
    // Original global scales are rank1[1], so bank storage is[E,1]. Only the
    // grouped reader's semantic scalar-vector boundary removes that real axis.
    const auto view=take(g::ViewSpec::reshape(std::array{g::Dimension::fixed(capacity)}),"explicit original global-scale reader view");
    const auto wg=take(b.add(view,banks[2]),"original WG divisors");
    const auto ig=take(b.add(view,banks[3]),"original IG multipliers/divisors");
    expert_output=take(b.add(c::model_ops::grouped_local_nvfp4_linear(c::model_ops::Fp4InputRounding::kBFloat16),
        mlp,banks[0],banks[1],wg,ig,local_ids),"existing local16 BF16-preround selected reader");
    check(b.publish(ct_output),"publish shared CT projection");check(b.publish(expert_output),"publish selected local gate projection");
    graph=take(std::move(b).finish(),"finish CT18 source graph");
  }
};

// Generated values only. Independent scalar enumeration is not a replacement
// executor; actual Graph uses the one shared CT math and existing local op.
inline std::uint16_t ct_sf(std::int64_t group) {
  constexpr std::array<std::uint16_t,4> bits{0x3f81,0x3ef1,0x3f03,0x3fdd};return bits[std::size_t(group)%bits.size()];
}
inline std::uint8_t ct_code(std::int64_t row,std::int64_t term) {
  constexpr std::array<std::uint8_t,4> codes{0x38,0xb0,0x28,0xc0};return codes[std::size_t(row+term)%codes.size()];
}
inline std::array<ExternalRole,2> make_ct(const c::StorageOwner &owner) {
  std::vector<std::uint8_t> weight(std::size_t(columns*hidden),0);
  std::vector<std::uint16_t> scales(std::size_t(hidden/128),0);
  for(std::int64_t row=0;row<columns;++row) {
    const auto group=row%(hidden/128);
    weight[std::size_t(row*hidden+group*128)]=ct_code(row,0);
    weight[std::size_t(row*hidden+group*128+3)]=ct_code(row,1);
  }
  for(std::int64_t group=0;group<hidden/128;++group)scales[std::size_t(group)]=ct_sf(group);
  const std::array<std::span<const std::byte>,2> bytes{std::as_bytes(std::span(weight)),std::as_bytes(std::span(scales))};
  std::array<ExternalRole,2> result;
  for(std::size_t i=0;i<result.size();++i){const auto &s=ct_specs[i];
    result[i]={named::detail::bind(s.name,s.description(),owner,bytes[i],&fixture_identity),s.begin,s.end};}
  return result;
}
inline float wg(std::int32_t global) {
  constexpr std::array<float,4> values{2.F,std::bit_cast<float>(std::uint32_t{0x453f40be}),1.00390625F,.7F};return values[std::size_t(global)];
}
inline float ig(std::int32_t global) {
  constexpr std::array<float,4> values{std::bit_cast<float>(std::uint32_t{0x3f8020c5}),.3F,.7F,1.25F};return values[std::size_t(global)];
}
inline std::int64_t expert_lane(std::int64_t row){return row==0?0:row==1?1:hidden-15;}
inline unsigned expert_code(std::int32_t global,std::int64_t row){return unsigned(2+(global+row)%5)|(row==2?8u:0u);}
inline std::uint8_t expert_sf(std::int32_t global,std::int64_t row,std::int64_t group){return std::uint8_t(0x3c+(global+row+group)%3);}
inline std::array<ExternalExpert,2> make_experts(const Selection &selected,const c::StorageOwner &owner) {
  std::array<ExternalExpert,2> result;
  for(std::size_t slot=0;slot<result.size();++slot) {
    const auto global=selected.globals[slot];result[slot].global=global;
    std::vector<std::uint8_t> w(std::size_t(full_expert_rows*hidden/2),0),sf(std::size_t(full_expert_rows*hidden/16));
    for(std::int64_t row=0;row<full_expert_rows;++row) {
      const auto lane=expert_lane(row%rows);
      w[std::size_t(row*(hidden/2)+lane/2)]=std::uint8_t(expert_code(global,row%rows)<<(4*(lane%2)));
      for(std::int64_t group=0;group<hidden/16;++group)sf[std::size_t(row*(hidden/16)+group)]=expert_sf(global,row,group);
    }
    const std::array<float,1> weight_global{wg(global)},input_global{ig(global)};
    const std::array<std::span<const std::byte>,4> bytes{std::as_bytes(std::span(w)),std::as_bytes(std::span(sf)),
        std::as_bytes(std::span(weight_global)),std::as_bytes(std::span(input_global))};
    for(std::size_t p=0;p<4;++p){const auto &s=expert_specs[std::size_t(global)].roles[p];
      result[slot].roles[p]={named::detail::bind(s.name,s.description(),owner,bytes[p],&fixture_identity),s.begin,s.end};}
  }
  return result;
}
inline std::vector<std::uint16_t> attention_input(unsigned phase) {
  std::vector<std::uint16_t> a(std::size_t(hidden),0);
  for(std::int64_t group=0;group<hidden/128;++group) {
    const auto base=std::size_t(group*128);
    if(group==1){a[base]=0x8000;continue;}
    const auto factor=group==0?std::ldexp(1.F,-20):std::ldexp(1.F,int((group+phase)%3));
    a[base]=scalar::bf16(factor*(.5F+float((group+phase)%3)*.125F));
    a[base+3]=scalar::bf16(factor*(group%2?-.25F:.125F));
  }
  return a;
}
inline std::vector<std::uint16_t> mlp_input(unsigned phase) {
  std::vector<std::uint16_t> a(std::size_t(hidden),0);
  for(std::int64_t group=0;group<hidden/16;++group) {
    const auto base=std::size_t(group*16);
    a[base]=scalar::bf16(phase?.75F:1.25F);a[base+1]=scalar::bf16(6.F);
    a[base+2]=scalar::bf16(-.25F);a[base+3]=0x8000;
  }
  return a;
}
inline std::vector<std::uint16_t> activation_reference(std::span<const std::uint16_t> a,bool wrong_floor=false) {
  std::vector<std::uint16_t> out(a.begin(),a.end());
  for(std::int64_t group=0;group<hidden/128;++group) {
    float maximum=0;
    for(std::int64_t lane=0;lane<128;++lane)maximum=std::max(maximum,std::abs(scalar::wide(a[std::size_t(group*128+lane)])));
    auto scale=ct::bf(double(maximum)/448.);
    scale=wrong_floor?std::max(scale,1.F/128.F):(scale==0.F?1.F/128.F:scale);
    for(std::int64_t lane=0;lane<128;++lane) {
      const auto index=std::size_t(group*128+lane);
      const auto normalized=ct::bf(double(scalar::wide(a[index]))/double(scale));
      const auto shifted=ct::bf(double(normalized)+0.);
      const auto code=ct::encode(std::clamp(shifted,-448.F,448.F));
      const auto centered=ct::bf(double(scalar::e4m3(code))-0.);
      out[index]=scalar::bf16(ct::bf(double(centered)*double(scale)));
    }
  }
  return out;
}
inline bool expect_ct(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> a) {
  if(actual.size()!=std::size_t(columns)){ADD_FAILURE()<<"CT18 actual CT result extent";return false;}
  const auto quantized=activation_reference(a);std::vector<std::uint16_t> weight(std::size_t(hidden),0);
  for(std::int64_t row=0;row<columns;++row) {
    std::fill(weight.begin(),weight.end(),0);const auto group=row%(hidden/128);
    for(std::int64_t term=0;term<2;++term)weight[std::size_t(group*128+(term?3:0))]=scalar::bf16(
        scalar::f32(double(scalar::e4m3(ct_code(row,term)))*scalar::wide(ct_sf(group))));
    const auto dot=qwen_bfloat_original_reference::detail::dot(quantized,std::as_bytes(std::span(weight)));
    if(!qwen_nvfp4_original_reference::interval(actual[std::size_t(row)],dot,"CT18 shared BF16 block projection",0,std::size_t(row)))return false;
  }
  return true;
}
inline std::vector<std::uint16_t> expert_oracle(std::span<const std::uint16_t> a,std::span<const std::int32_t,2> route,
    c::model_ops::Fp4InputRounding rounding=c::model_ops::Fp4InputRounding::kBFloat16,bool share_ig=false) {
  std::vector<std::uint16_t> y(std::size_t(2*rows),0);
  for(std::size_t slot=0;slot<2;++slot)for(std::int64_t row=0;row<rows;++row) {
    const auto global=route[slot];const auto lane=expert_lane(row),group=lane/16;float maximum=0;
    for(std::int64_t i=0;i<16;++i)maximum=std::max(maximum,std::abs(scalar::wide(a[std::size_t(group*16+i)])));
    const auto aq=local::activation(scalar::wide(a[std::size_t(lane)]),maximum,ig(share_ig?0:global),rounding);
    const auto w=local::weight(expert_code(global,row),expert_sf(global,row,group),wg(global));
    y[slot*std::size_t(rows)+std::size_t(row)]=scalar::bf16(scalar::f32(double(aq)*double(w)));
  }
  return y;
}
} // namespace ct18_selected_source_test
