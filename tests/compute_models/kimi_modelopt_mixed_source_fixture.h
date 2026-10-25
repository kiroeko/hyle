#pragma once

#include "original_readonly_role.h"
#include "model_plane_resident_weights.h"
#include "e8m0_weight_import_fixture.h"

// One finite source composition, not a loader or a complete Kimi block.
// Attention f_a and MLP routed-expert inputs are distinct upstream boundaries.
// Caller authenticates source bytes; the explicit fixture identity below never
// claims publisher values. All arithmetic uses existing public operations.
namespace kimi20_mixed_source_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;
namespace named=original_named_role_source;namespace supply=model_plane_resident_weights;
namespace scalar=nvfp4_reference_test;
using Type=c::DataType;using scalar::take;using scalar::check;using scalar::dense;
inline constexpr std::string_view repository="nvidia/Kimi-K3-NVFP4";
inline constexpr std::string_view revision="b2428a0b83a8b712ff2e1a8448a103d4175341f1";
inline constexpr std::string_view file="model-00013-of-000096.safetensors";
inline constexpr named::Identity original_identity{repository,revision,"stored-F32-PB-divide128-and-NVFP4-quartet",file};
inline constexpr named::Identity fixture_identity{"hyle-fixture/kimi20-mixed-source",
    "0000000000000000000000000000000000000000","generated-original-shapes","not-publisher-values"};
inline constexpr std::int64_t hidden=7168,pb_columns=128,expert_depth=3584,full_expert_rows=3072,
    rows=3,domain=896,capacity=2;
struct Spec final {
  std::string_view name;Type type;std::array<std::int64_t,4> shape;std::size_t rank;
  std::uint64_t begin,end;
  c::TensorDesc description()const{return take(c::TensorDesc::dense(type,std::span(shape).first(rank)),"exact Kimi20 source role");}
};
inline constexpr std::array<Spec,2> pb_specs{{
    {"language_model.model.layers.12.self_attn.f_a_proj.weight",Type::kFloat8E4M3FN,{128,7168,0,0},2,2231089472ULL,2232006976ULL},
    {"language_model.model.layers.12.self_attn.f_a_proj.weight_scale",Type::kFloat32,{1,1,56,1},4,74976ULL,75200ULL}
}};
struct ExpertSpec final{std::int32_t global;std::array<Spec,4> roles;};
inline constexpr std::array<ExpertSpec,4> expert_specs{{
    {0, {{{"language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight",Type::kUInt8,{3072,1792,0,0},2,2673981760ULL,2679486784ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_scale",Type::kFloat8E4M3FN,{3072,224,0,0},2,380713280ULL,381401408ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_scale_2",Type::kFloat32,{0,0,0,0},0,4ULL,8ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.0.w1.input_scale",Type::kFloat32,{0,0,0,0},0,0ULL,4ULL}}}},
    {1, {{{"language_model.model.layers.12.block_sparse_moe.experts.1.w1.weight",Type::kUInt8,{3072,1792,0,0},2,2690496832ULL,2696001856ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.1.w1.weight_scale",Type::kFloat8E4M3FN,{3072,224,0,0},2,382777664ULL,383465792ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.1.w1.weight_scale_2",Type::kFloat32,{0,0,0,0},0,28ULL,32ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.1.w1.input_scale",Type::kFloat32,{0,0,0,0},0,24ULL,28ULL}}}},
    {2, {{{"language_model.model.layers.12.block_sparse_moe.experts.2.w1.weight",Type::kUInt8,{3072,1792,0,0},2,4523669824ULL,4529174848ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.2.w1.weight_scale",Type::kFloat8E4M3FN,{3072,224,0,0},2,611924288ULL,612612416ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.2.w1.weight_scale_2",Type::kFloat32,{0,0,0,0},0,2692ULL,2696ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.2.w1.input_scale",Type::kFloat32,{0,0,0,0},0,2688ULL,2692ULL}}}},
    {3, {{{"language_model.model.layers.12.block_sparse_moe.experts.3.w1.weight",Type::kUInt8,{3072,1792,0,0},2,6356842816ULL,6362347840ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.3.w1.weight_scale",Type::kFloat8E4M3FN,{3072,224,0,0},2,841070912ULL,841759040ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.3.w1.weight_scale_2",Type::kFloat32,{0,0,0,0},0,5356ULL,5360ULL},
      {"language_model.model.layers.12.block_sparse_moe.experts.3.w1.input_scale",Type::kFloat32,{0,0,0,0},0,5352ULL,5356ULL}}}}
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
  return {dense(Type::kUInt8,{rows,expert_depth/2}),dense(Type::kFloat8E4M3FN,{rows,expert_depth/16}),
      dense(Type::kFloat32,{}),dense(Type::kFloat32,{})};
}
inline c::TensorDesc pb_description() {
  return take(c::float8_weight_description(hidden,pb_columns,c::Float8ScaleMode::kFloat32Block128),"existing F32-SF PB carrier");
}
inline bool same_identity(const named::Identity *a,const named::Identity &b) {
  return a&&a->repository==b.repository&&a->revision==b.revision&&a->variant==b.variant&&a->file==b.file;
}
inline hyle::Status validate(const ExternalRole &v,const Spec &spec,const named::Identity &identity) {
  if(!same_identity(v.ready.identity,identity)||v.ready.name!=spec.name||v.ready.description!=spec.description()||
      !v.ready.owner||!v.ready.lease.valid()||v.ready.lease.size()!=spec.end-spec.begin||v.begin!=spec.begin||v.end!=spec.end)
    return hyle::Status::InvalidArgument("Kimi20 original role identity/span/typed extent differs");
  HYLE_ASSIGN_OR_RETURN(auto tensor,v.ready.lease.tensor(v.ready.description));
  if(tensor.storage().resource()!=v.ready.owner.ref())
    return hyle::Status::InvalidArgument("Kimi20 source owner differs from its actual readonly storage");
  return {};
}
inline hyle::Result<named::ReadonlyRole> prepare_pb(const std::array<ExternalRole,2> &pb) {
  const auto &identity=same_identity(pb[0].ready.identity,original_identity)?original_identity:fixture_identity;
  for(std::size_t p=0;p<pb.size();++p)HYLE_RETURN_IF_ERROR(validate(pb[p],pb_specs[p],identity));
  HYLE_ASSIGN_OR_RETURN(auto codes,pb[0].ready.lease.tensor(pb[0].ready.description));
  HYLE_ASSIGN_OR_RETURN(auto raw_sf,pb[1].ready.lease.view());
  // Only original singleton axes are removed; F32 bits never pass BF16.
  HYLE_ASSIGN_OR_RETURN(auto scales,c::ConstTensorView::create(dense(Type::kFloat32,{1,hidden/128}),raw_sf));
  const auto desc=pb_description();HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto buffer,m::Buffer::create(pb[0].ready.owner.ref(),bytes,256));
  HYLE_ASSIGN_OR_RETURN(auto output,c::TensorView::create(desc,buffer.view()));
  HYLE_RETURN_IF_ERROR(c::import_float8_block128_weight(codes,scales,output));
  HYLE_ASSIGN_OR_RETURN(auto ready,pb[0].ready.owner.adopt_buffer(std::move(buffer)));
  return named::ReadonlyRole{pb_specs[0].name,desc,pb[0].ready.owner,std::move(ready),&identity};
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
        return std::unexpected(hyle::Status::InvalidArgument("Kimi20 finite source IDs must be cached/sorted/distinct"));
      for(std::size_t p=0;p<4;++p)HYLE_RETURN_IF_ERROR(validate(experts[i].roles[p],expert_specs[std::size_t(global)].roles[p],identity));
    }
    Source result;result.experts_=experts;
    const auto planes=plane_descriptions();
    for(std::size_t i=0;i<experts.size();++i)for(std::size_t p=0;p<4;++p) {
      HYLE_ASSIGN_OR_RETURN(auto view,result.experts_[i].roles[p].ready.lease.view());
      HYLE_ASSIGN_OR_RETURN(auto bytes,planes[p].required_bytes());
      HYLE_ASSIGN_OR_RETURN(view,view.subview(0,bytes)); // First N3 original rows; scalar planes keep their full4 bytes.
      HYLE_ASSIGN_OR_RETURN(result.maps_[i][p],view.map_host());
      HYLE_RETURN_IF_ERROR(result.maps_[i][p].invalidate());
    }
    return result;
  }
  std::span<const std::byte> plane_bytes(std::size_t plane,std::int32_t global)const {
    if(plane>=4)throw std::runtime_error("Kimi20 source plane is not a quartet member");
    for(std::size_t i=0;i<experts_.size();++i)if(experts_[i].global==global)return maps_[i][plane].bytes();
    throw std::runtime_error("Kimi20 selected expert was not supplied");
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
  g::Graph graph;g::ConstantBinding constant;g::ValueId attention{},routed{},ids{},pb_output{},expert_output{};
  std::array<g::ValueId,5> banks;
  c::TensorDesc attention_desc=dense(Type::kBFloat16,{1,hidden}),routed_desc=dense(Type::kBFloat16,{1,1,expert_depth}),
      ids_desc=dense(Type::kInt32,{1,2}),pb_output_desc=dense(Type::kBFloat16,{1,pb_columns}),
      expert_output_desc=dense(Type::kBFloat16,{1,2,rows});
  Case(const named::ReadonlyRole &pb,const supply::Layout &layout) {
    if(pb.name!=pb_specs[0].name||pb.description!=pb_description()||!pb.owner||!pb.lease.valid())
      throw std::runtime_error("Kimi20 PB binding must be the real prepared encoded carrier");
    auto b=take(g::GraphBuilder::create(),"Kimi20 finite mixed source graph");
    attention=take(b.input(attention_desc),"actual attention f_a boundary");
    routed=take(b.input(routed_desc),"actual independent post-routed-down/norm boundary");
    ids=take(b.input(ids_desc),"actual selected global expert IDs");
    for(std::size_t p=0;p<4;++p)banks[p]=take(b.input(layout.resident[p]),"same-generation original quartet plane");
    banks[4]=take(b.input(layout.map_description()),"same-generation global-local map");
    const auto weight=take(b.constant(pb_description()),"original F32-SF PB constant");
    constant=take(pb.binding(weight),"same-owner encoded PB constant lease");
    // This stored-F32 reader uses divide, not the different E8/power2 policy.
    pb_output=take(b.add(c::model_ops::float8_block_linear(c::model_ops::Float8ScalePolicy::kDivide,1.e-4F),attention,weight),"existing dynamic128 PB reader");
    const auto local=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},banks[4],ids),"checked original selected map");
    expert_output=take(b.add(c::model_ops::grouped_nvfp4_linear(),routed,banks[0],banks[1],banks[2],banks[3],local),"existing selected NVFP4 BF16-QDQ reader");
    check(b.publish(pb_output),"publish actual PB result");check(b.publish(expert_output),"publish actual selected W1 result");
    graph=take(std::move(b).finish(),"finish Kimi20 source composition");
  }
};

// Explicit generated bodies, independent scalar reference and original header
// geometry. They are not downloaded publisher values or a complete Kimi block.
inline float pb_sf(std::size_t group) {
  constexpr std::array<float,4> values{1.0039F,.7F,.3333F,1.997F};return values[group%values.size()];
}
inline std::uint8_t pb_code(std::int64_t row,std::int64_t group) {
  constexpr std::array<std::uint8_t,4> codes{0x38,0xb0,0x28,0xc0};return codes[std::size_t(row+group)%codes.size()];
}
inline float wg(std::int32_t global){return 1.00390625F+float(global)*.0078125F;}
inline float ig(std::int32_t global){constexpr std::array values{.1F,.3F,.2F,.7F};return values[std::size_t(global)];}
inline std::int64_t expert_lane(std::int64_t row){return row==0?0:row==1?17:expert_depth-1;}
inline unsigned expert_code(std::int32_t global,std::int64_t row){return unsigned(1+(global+row*2)%7)|(row==2?8u:0u);}
inline std::uint8_t expert_sf(std::int32_t global,std::int64_t row,std::int64_t group){return std::uint8_t(0x39+(global+row+group)%5);}
inline std::array<ExternalRole,2> make_pb(const c::StorageOwner &owner) {
  std::vector<std::uint8_t> w(std::size_t(pb_columns*hidden),0);
  std::vector<float> sf(std::size_t(hidden/128));
  for(std::int64_t n=0;n<pb_columns;++n)for(const auto group:std::array<std::int64_t,2>{0,hidden/128-1})
    w[std::size_t(n*hidden+group*128+1)]=pb_code(n,group);
  for(std::size_t g=0;g<sf.size();++g)sf[g]=pb_sf(g);
  const std::array<std::span<const std::byte>,2> bytes{std::as_bytes(std::span(w)),std::as_bytes(std::span(sf))};
  std::array<ExternalRole,2> result;
  for(std::size_t p=0;p<result.size();++p)result[p]={named::detail::bind(pb_specs[p].name,pb_specs[p].description(),owner,bytes[p],&fixture_identity),pb_specs[p].begin,pb_specs[p].end};
  return result;
}
inline std::array<ExternalExpert,2> make_experts(const Selection &selected,const c::StorageOwner &owner) {
  std::array<ExternalExpert,2> result;
  for(std::size_t slot=0;slot<result.size();++slot) {
    const auto global=selected.globals[slot];result[slot].global=global;
    std::vector<std::uint8_t> w(std::size_t(full_expert_rows*expert_depth/2),0),sf(std::size_t(full_expert_rows*expert_depth/16));
    for(std::int64_t row=0;row<full_expert_rows;++row) {
      const auto lane=expert_lane(row%rows);
      w[std::size_t(row*(expert_depth/2)+lane/2)]=std::uint8_t(expert_code(global,row%rows)<<(4*(lane%2)));
      for(std::int64_t group=0;group<expert_depth/16;++group)sf[std::size_t(row*(expert_depth/16)+group)]=expert_sf(global,row,group);
    }
    const std::array<float,1> weight_global{wg(global)},input_global{ig(global)};
    const std::array<std::span<const std::byte>,4> bytes{std::as_bytes(std::span(w)),std::as_bytes(std::span(sf)),
        std::as_bytes(std::span(weight_global)),std::as_bytes(std::span(input_global))};
    for(std::size_t p=0;p<4;++p){const auto &spec=expert_specs[std::size_t(global)].roles[p];
      result[slot].roles[p]={named::detail::bind(spec.name,spec.description(),owner,bytes[p],&fixture_identity),spec.begin,spec.end};}
  }
  return result;
}
inline std::vector<std::uint16_t> attention_input(unsigned phase) {
  std::vector<std::uint16_t> a(std::size_t(hidden),0);
  for(const auto group:std::array<std::int64_t,2>{0,hidden/128-1}) {
    const auto base=std::size_t(group*128);a[base]=scalar::bf16(phase?448.F:450.F);
    a[base+1]=scalar::bf16((group?.50390625F:.40625F)*(phase?-1.F:1.F));
  }
  return a;
}
inline std::vector<std::uint16_t> routed_input(unsigned phase) {
  constexpr std::array values{.25F,.75F,1.25F,1.75F,2.5F,3.5F,5.F,6.F,-.25F,-.75F,-1.25F,-1.75F,-2.5F,-3.5F,-5.F,-6.F};
  std::vector<std::uint16_t> a(std::size_t(expert_depth),0);
  for(std::size_t k=0;k<a.size();++k)a[k]=scalar::bf16(values[(k+phase*3)%values.size()]);return a;
}
inline std::vector<std::uint16_t> pb_oracle(std::span<const std::uint16_t> a,bool omit_qdq=false,bool power2=false) {
  std::vector<std::uint16_t> y(std::size_t(pb_columns),0);
  for(std::int64_t n=0;n<pb_columns;++n) {
    float sum=0;
    for(const auto group:std::array<std::int64_t,2>{0,hidden/128-1}) {
      const auto base=std::size_t(group*128);float maximum=0;
      for(std::size_t lane=0;lane<128;++lane)maximum=std::max(maximum,std::abs(scalar::wide(a[base+lane])));
      auto scale=scalar::f32(double(std::max(maximum,1.e-4F))/448.);
      if(power2){int exponent=0;const auto fraction=std::frexp(scale,&exponent);scale=std::ldexp(1.F,fraction==.5F?exponent-1:exponent);}
      const auto input=scalar::wide(a[base+1]);
      const auto normalized=scalar::f32(double(input)/double(scale));
      const auto code=scalar::encode_scale(std::abs(normalized));
      const auto aq=omit_qdq?input:std::copysign(scalar::e4m3(code),normalized);
      auto term=scalar::multiply(aq,scalar::e4m3(pb_code(n,group)));
      if(!omit_qdq)term=scalar::multiply(term,scale);
      term=scalar::multiply(term,pb_sf(std::size_t(group)));sum=scalar::f32(double(sum)+double(term));
    }
    y[std::size_t(n)]=scalar::bf16(sum);
  }
  return y;
}
inline std::vector<std::uint16_t> expert_oracle(std::span<const std::uint16_t> a,
    std::span<const std::int32_t,2> route,bool share_ig=false,bool omit_a_round=false,bool omit_w_round=false) {
  std::vector<std::uint16_t> result(std::size_t(2*rows));
  for(std::size_t slot=0;slot<2;++slot)for(std::int64_t row=0;row<rows;++row) {
    const auto global=route[slot];const auto lane=expert_lane(row),group=lane/16;float maximum=0;
    for(std::int64_t i=0;i<16;++i)maximum=std::max(maximum,std::abs(scalar::wide(a[std::size_t(group*16+i)])));
    const auto aq=scalar::activation_value(scalar::wide(a[std::size_t(lane)]),maximum,ig(share_ig?0:global),omit_a_round);
    const auto w=scalar::weight_value(expert_code(global,row),expert_sf(global,row,group),wg(global),omit_w_round);
    result[slot*std::size_t(rows)+std::size_t(row)]=scalar::bf16(scalar::f32(double(aq)*double(w)));
  }
  return result;
}
} // namespace kimi20_mixed_source_test
