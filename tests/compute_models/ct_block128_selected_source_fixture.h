#pragma once

#include "original_readonly_role.h"
#include "model_plane_resident_weights.h"
#include "ct_block128_qdq_fixture.h"
#include "qwen_nvfp4_original_reference.h"

// Fixed CT21 layer1/W1 source metadata and one finite generated consumer.
// Only ordinary Graph/leases and existing CpuSupply are used. The shared CT
// append remains the sole execution math; no publisher payload is implied.
namespace ct21_selected_source_test {
namespace c=hyle::compute;namespace g=hyle::graph;namespace m=hyle::mem;
namespace named=original_named_role_source;namespace supply=model_plane_resident_weights;
namespace f=ct_block128_qdq_test;namespace scalar=nvfp4_reference_test;
namespace reference=qwen_nvfp4_original_reference;
namespace math=hyle::inference::ct_block128;
using Type=c::DataType;using f::take;using f::check;
inline constexpr std::string_view repository="RedHatAI/Kimi-K3-FP8-BLOCK";
inline constexpr std::string_view revision="6409c37cf440abeb76f0b44ab4fb7db8c8bccb4b";
inline constexpr std::string_view file="model-00002-of-000096.safetensors";
inline constexpr std::int64_t n=3072,k=3584,groups=k/128,blocks=n/128,domain=896,capacity=2;
inline constexpr named::Identity original_identity{repository,revision,"CT-E4-BF16-block128",file};
inline constexpr named::Identity fixture_identity{"hyle-fixture/ct21-selected-source",
    "0000000000000000000000000000000000000000","generated-original-geometry","not-original-publisher-values"};
struct Spec final {
  std::string_view name;Type type;std::array<std::int64_t,2> shape;
  std::uint64_t begin,end;
  c::TensorDesc description()const{return take(c::TensorDesc::dense(type,shape),"exact CT21 source description");}
};
struct ExpertSpec final{std::int32_t global;std::array<Spec,2> roles;};
inline constexpr std::array<ExpertSpec,4> specs{{
    {0, {{{"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",Type::kFloat8E4M3FN,{3072,3584},904377728ULL,915387776ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale",Type::kBFloat16,{24,28},643584ULL,644928ULL}}}},
    {1, {{{"language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight",Type::kFloat8E4M3FN,{3072,3584},937407872ULL,948417920ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight_scale",Type::kBFloat16,{24,28},647616ULL,648960ULL}}}},
    {2, {{{"language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight",Type::kFloat8E4M3FN,{3072,3584},4603753856ULL,4614763904ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.2.w1.weight_scale",Type::kBFloat16,{24,28},1095168ULL,1096512ULL}}}},
    {3, {{{"language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight",Type::kFloat8E4M3FN,{3072,3584},8270099840ULL,8281109888ULL},
      {"language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_scale",Type::kBFloat16,{24,28},1542720ULL,1544064ULL}}}}
}};
struct ExternalRole final{named::ReadonlyRole ready;std::uint64_t begin=0,end=0;};
struct ExternalExpert final{std::int32_t global=-1;std::array<ExternalRole,2> roles;};
struct Selection final{std::vector<std::int32_t> globals,map;};
inline Selection selection(unsigned generation) {
  Selection result;result.globals={std::int32_t(2*generation),std::int32_t(2*generation+1)};
  result.map.assign(std::size_t(domain),-1);
  for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
  return result;
}
inline std::array<c::TensorDesc,2> descriptions(){return {specs[0].roles[0].description(),specs[0].roles[1].description()};}
class Source final {
 public:
  Source()=default;Source(Source&&)=default;Source&operator=(Source&&)=delete;
  Source(const Source&)=delete;Source&operator=(const Source&)=delete;
  static hyle::Result<Source> accept(const std::array<ExternalExpert,2> &input) {
    for(std::size_t i=0;i<input.size();++i) {
      const auto &expert=input[i];
      if(expert.global<0||expert.global>=std::int32_t(specs.size())||(i&&input[i-1].global>=expert.global))
        return std::unexpected(hyle::Status::InvalidArgument("finite CT21 source IDs must be cached/sorted/distinct"));
      for(std::size_t p=0;p<2;++p) {
        const auto &role=expert.roles[p];const auto &spec=specs[std::size_t(expert.global)].roles[p];
        const auto matches=[&](const named::Identity &expected) {
          const auto *id=role.ready.identity;
          return id&&id->repository==expected.repository&&id->revision==expected.revision&&
              id->variant==expected.variant&&id->file==expected.file;
        };
        if(!matches(original_identity)&&!matches(fixture_identity))
          return std::unexpected(hyle::Status::InvalidArgument("CT21 source needs its exact artifact or explicit fixture identity"));
        if(role.ready.name!=spec.name||role.ready.description!=spec.description()||!role.ready.owner||
            !role.ready.lease.valid()||role.ready.lease.size()!=spec.end-spec.begin||role.begin!=spec.begin||role.end!=spec.end)
          return std::unexpected(hyle::Status::InvalidArgument("CT21 independent E4/BF16 source span/type differs"));
        HYLE_ASSIGN_OR_RETURN(auto tensor,role.ready.lease.tensor(role.ready.description));
        if(tensor.storage().resource()!=role.ready.owner.ref())
          return std::unexpected(hyle::Status::InvalidArgument("CT21 ready source owner/resource differs"));
      }
    }
    Source result;result.original_=input;
    for(std::size_t i=0;i<input.size();++i)for(std::size_t p=0;p<2;++p) {
      HYLE_ASSIGN_OR_RETURN(auto view,result.original_[i].roles[p].ready.lease.view());
      HYLE_ASSIGN_OR_RETURN(result.maps_[i][p],view.map_host());
      HYLE_RETURN_IF_ERROR(result.maps_[i][p].invalidate());
    }
    return result;
  }
  std::span<const std::byte> plane_bytes(std::size_t p,std::int32_t global)const {
    if(p>=2)throw std::runtime_error("CT21 source plane is not W/SF");
    for(std::size_t i=0;i<original_.size();++i)if(original_[i].global==global)return maps_[i][p].bytes();
    throw std::runtime_error("CT21 selected original role was not supplied");
  }
  hyle::Status close() {
    for(auto &expert:maps_)for(auto &mapping:expert)if(mapping.valid())HYLE_RETURN_IF_ERROR(mapping.close());
    original_={};return {};
  }
 private:
  std::array<ExternalExpert,2> original_{};
  std::array<std::array<m::HostMapping,2>,2> maps_{};
};

struct Case final {
  g::Graph graph;g::ValueId activation{},ids{},output{};std::array<g::ValueId,3> banks;
  std::array<g::ConstantBinding,4> constants;
  c::TensorDesc a_desc=f::dense(Type::kBFloat16,{1,k}),ids_desc=f::dense(Type::kInt32,{1,2}),output_desc;
  explicit Case(const supply::Layout &layout) {
    auto b=take(g::GraphBuilder::create(),"CT21 selected original graph");
    activation=take(b.input(a_desc),"CT21 actual BF16 A");ids=take(b.input(ids_desc),"CT21 actual global IDs");
    for(std::size_t p=0;p<2;++p)banks[p]=take(b.input(layout.resident[p]),"CT21 compact original W/SF bank");
    banks[2]=take(b.input(layout.map_description()),"CT21 same-generation global-local map");
    const auto local=take(b.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},banks[2],ids),"actual mapped expert IDs");
    const std::array literal_desc{f::dense(Type::kInt32,{n}),f::dense(Type::kBFloat16,{}),
        f::dense(Type::kBFloat16,{}),f::dense(Type::kBFloat16,{})};
    std::vector<std::int32_t> row_blocks(std::size_t(n),0);
    for(std::int64_t row=0;row<n;++row)row_blocks[std::size_t(row)]=std::int32_t(row/128);
    const std::array<std::uint16_t,1> limit{0x43e0},epsilon{0x3c00},zero{0};
    const std::array<std::span<const std::byte>,4> bytes{
        std::as_bytes(std::span(row_blocks)),std::as_bytes(std::span(limit)),std::as_bytes(std::span(epsilon)),std::as_bytes(std::span(zero))};
    std::array<math::Operand,4> literals;
    for(std::size_t i=0;i<literals.size();++i) {
      const auto value=take(b.constant(literal_desc[i]),"original BF16/row block literal");
      auto memory=take(m::Buffer::create(m::system_resource(),bytes[i].size(),256),"CT21 small literal bytes");
      check(memory.view().write(bytes[i]),"CT21 literal write");auto lease=take(c::ReadLease::adopt(std::move(memory)),"CT21 ready literal");
      constants[i]={value,take(lease.tensor(literal_desc[i]),"CT21 literal view"),g::ConstantPolicy::kMaterialize,{},lease};
      literals[i]={value,literal_desc[i]};
    }
    const auto result=take(math::append_projection(b,{activation,a_desc},{banks[0],layout.resident[0]},
        {banks[1],layout.resident[1]},literals[0],{literals[1],literals[2],literals[3]},math::Operand{local,ids_desc}),"shared selected CT21 projection");
    // Shared math ends in a Graph-owned reshape. Materialize only the 12KiB
    // published result so its lease can outlive the original selected banks.
    output=take(b.add(c::Copy{},result.value),"CT21 independent terminal result");
    output_desc=take(result.description.concrete_description(),"CT21 actual output");
    check(b.publish(output),"publish CT21 BF16 projection");graph=take(std::move(b).finish(),"finish CT21 composition");
  }
};

// All values below are fixture-only; original payload remains unqualified.
inline std::uint16_t sf(std::int32_t global,std::int64_t block,std::int64_t group) {
  constexpr std::array<std::uint16_t,4> values{0x3f81,0x3ef1,0x3f03,0x3fdd};
  return values[std::size_t((global*3+block+group)%4)];
}
inline std::uint8_t code(std::int32_t global,std::int64_t row,std::int64_t group,std::int64_t term) {
  return std::uint8_t(0x38+(global*7+row*3+group*5+term)%8+((row+group+term+global)%5==0?0x80:0));
}
inline std::array<ExternalExpert,2> make_source(const Selection &selected,const c::StorageOwner &owner) {
  std::array<ExternalExpert,2> result;
  for(std::size_t i=0;i<result.size();++i) {
    const auto global=selected.globals[i];result[i].global=global;
    std::vector<std::uint8_t> w(std::size_t(n*k),0);
    std::vector<std::uint16_t> scales(std::size_t(blocks*groups));
    for(std::int64_t row=0;row<n;++row)for(std::int64_t group=0;group<groups;++group) {
      w[std::size_t(row*k+group*128)]=code(global,row,group,0);
      w[std::size_t(row*k+group*128+3)]=code(global,row,group,1);
    }
    for(std::int64_t block=0;block<blocks;++block)for(std::int64_t group=0;group<groups;++group)
      scales[std::size_t(block*groups+group)]=sf(global,block,group);
    const std::array<std::span<const std::byte>,2> bytes{std::as_bytes(std::span(w)),std::as_bytes(std::span(scales))};
    for(std::size_t p=0;p<2;++p) {
      const auto &spec=specs[std::size_t(global)].roles[p];
      result[i].roles[p]={named::detail::bind(spec.name,spec.description(),owner,bytes[p],&fixture_identity),spec.begin,spec.end};
    }
  }
  return result;
}
inline std::vector<std::uint16_t> activation(unsigned phase) {
  std::vector<std::uint16_t> a(std::size_t(k),0);
  for(std::int64_t group=0;group<groups;++group) {
    const auto factor=group==0?std::ldexp(1.F,-20):1.F;
    a[std::size_t(group*128)]=scalar::bf16(factor*(.5F+float((group+phase)%3)*.125F));
    a[std::size_t(group*128+3)]=scalar::bf16(factor*(group%2?-.25F:.125F));
  }
  return a;
}
inline std::vector<std::uint16_t> activation_reference(std::span<const std::uint16_t> input) {
  std::vector<std::uint16_t> output(input.begin(),input.end());
  for(std::int64_t group=0;group<groups;++group) {
    float maximum=0;
    for(std::int64_t lane=0;lane<128;++lane)maximum=std::max(maximum,std::abs(scalar::wide(input[std::size_t(group*128+lane)])));
    auto scale=f::bf(double(maximum)/448.);if(scale==0.F)scale=1.F/128.F;
    for(std::int64_t lane=0;lane<128;++lane) {
      const auto index=std::size_t(group*128+lane);
      const auto normalized=f::bf(double(scalar::wide(input[index]))/double(scale));
      const auto shifted=f::bf(double(normalized)+0.);
      const auto encoded=f::encode(std::clamp(shifted,-448.F,448.F));
      output[index]=scalar::bf16(f::bf(double(f::bf(double(scalar::e4m3(encoded))-0.))*double(scale)));
    }
  }
  return output;
}
inline bool expect(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> a,
    std::span<const std::int32_t,2> global) {
  if(actual.size()!=std::size_t(2*n)){ADD_FAILURE()<<"CT21 projection extent";return false;}
  const auto quantized=activation_reference(a);std::vector<std::uint16_t> weight(std::size_t(k),0);
  for(std::size_t slot=0;slot<2;++slot)for(std::int64_t row=0;row<n;++row) {
    std::fill(weight.begin(),weight.end(),0);
    for(std::int64_t group=0;group<groups;++group)for(std::int64_t term=0;term<2;++term)
      weight[std::size_t(group*128+(term?3:0))]=scalar::bf16(scalar::f32(
          double(scalar::e4m3(code(global[slot],row,group,term)))*scalar::wide(sf(global[slot],row/128,group))));
    const auto expected=qwen_bfloat_original_reference::detail::dot(quantized,std::as_bytes(std::span(weight)));
    if(!reference::interval(actual[slot*std::size_t(n)+std::size_t(row)],expected,"CT21 generated original-shaped selected projection",slot,std::size_t(row)))return false;
  }
  return true;
}
} // namespace ct21_selected_source_test
