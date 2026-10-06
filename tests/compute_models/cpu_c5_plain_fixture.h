#pragma once
#include "cpu_c5_plain_inputs.h"
#include "cpu_c5_non_gguf_fixture.h"

namespace cpu_c5_plain_fixture {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace mixed = cpu_c5_non_gguf_fixture;
namespace core = d5_moe_semantic_fragments;
using sem::Fragment; using sem::ValueId; using sem::dense; using sem::bytes;
using cpu_c5_plain_inputs::Case;
inline c::ModelArtifactProfileDescription profile(const Case &item) {
  cpu_c5_non_gguf_inputs::Case packed{}; packed.artifact=item.artifact;
  return mixed::profile(packed);
}
inline ValueId publish_float(Fragment &f, ValueId output) {
  if(f.description(output).data_type()!=c::DataType::kFloat32 && f.description(output).data_type()!=c::DataType::kInt32)
    return f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{output});
  return output;
}
inline std::vector<float> values(std::int64_t n,int seed) {
  std::vector<float> result(n); for(std::int64_t i=0;i<n;++i)result[i]=float((i*7+seed)%13-6)/16;
  return result;
}
inline Fragment projection(const Case &item,const c::ModelArtifactProfileDescription *identity,
                           std::int64_t n,std::int64_t first=0) {
  const auto k=item.shape[1];Fragment f{.profile=identity,.region=0x0c052001};
  constexpr std::int64_t period=17;
  auto input=f.input(dense({1,k},item.type));std::vector<float> weight(period*k);
  for(std::int64_t row=0;row<period;++row)for(std::int64_t col=0;col<k;++col)weight[row*k+col]=float(((row+first)*3+col*7)%17-8)/32;
  auto prototypes=mixed::floating_bytes(item.type,weight);const auto stride=prototypes.size()/period;
  std::vector<std::byte> storage(n*stride);
  for(std::int64_t row=0;row<n;++row)std::memcpy(storage.data()+row*stride,prototypes.data()+(row%period)*stride,stride);
  auto stored=f.constant(dense({n,k},item.type),std::move(storage));
  auto rhs=f.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{stored});
  f.outputs={publish_float(f,f.one(c::Operation(c::MatMul{}),{input,rhs}))};
  for(int pass:{0,1}) {
    auto activation=mixed::activation(1,k,pass);std::vector<float> output(n);
    std::array<float,period> dots{};
    for(std::int64_t row=0;row<period;++row) {
      double sum=0;for(std::int64_t col=0;col<k;++col)sum+=double(activation[col])*weight[row*k+col];
      dots[row]=mixed::rounded(item.type,float(sum));
    }
    for(std::int64_t row=0;row<n;++row)output[row]=dots[row%period];
    f.replays.push_back({{mixed::floating_bytes(item.type,activation)},{std::move(output)}});
  }
  return f;
}
inline Fragment lookup(const Case &item,const c::ModelArtifactProfileDescription *identity) {
  const auto n=item.shape[0],k=item.shape[1];Fragment f{.profile=identity,.region=0x0c052002};
  const std::int64_t period=item.type==c::DataType::kInt32?256:17;
  auto indices=f.input(dense({4},c::DataType::kInt32));std::vector<float> table(period*k);
  for(std::int64_t row=0;row<period;++row)for(std::int64_t col=0;col<k;++col)
    table[row*k+col]=item.type==c::DataType::kInt32?float((row*3+col)%256):float((row*3+col*7)%17-8)/32;
  std::vector<std::byte> prototypes;
  if(item.type==c::DataType::kInt32) {std::vector<std::int32_t> data;data.reserve(table.size());for(auto v:table)data.push_back(std::int32_t(v));prototypes=bytes(data);}
  else prototypes=mixed::floating_bytes(item.type,table);
  const auto stride=prototypes.size()/period;std::vector<std::byte> storage(n*stride);
  for(std::int64_t row=0;row<n;++row)std::memcpy(storage.data()+row*stride,prototypes.data()+(row%period)*stride,stride);
  auto weight=f.constant(dense({n,k},item.type),std::move(storage));
  f.outputs={publish_float(f,f.one(c::Operation(c::Gather{}),{weight,indices}))};
  f.exact_outputs={0};f.invalid_index_input=0;
  for(int pass:{0,1}) {
    const std::array<std::int32_t,4> ids{pass?1:0,std::int32_t(n-1),std::int32_t(n/2),std::int32_t(n/2)};
    std::vector<float> output(4*k);for(int row=0;row<4;++row)std::copy_n(table.data()+(ids[row]%period)*k,k,output.data()+row*k);
    f.replays.push_back({{bytes<std::int32_t>(ids)},{std::move(output)}});
  }
  return f;
}
inline Fragment norm(const Case &item,const c::ModelArtifactProfileDescription *identity) {
  const auto n=item.shape[0];Fragment f{.profile=identity,.region=0x0c052003};
  auto input=f.input(dense({1,n},item.type));auto scale=sem::oracle::scale(n);
  const bool zero_centered=std::string_view(item.kind)=="norm_zero_centered";
  if(zero_centered)for(auto &value:scale)value-=1.0F;
  for(auto &value:scale)value=mixed::rounded(item.type,value);
  auto gamma=f.constant(dense({n},item.type),mixed::floating_bytes(item.type,scale));
  auto wide_input=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{input});
  auto wide_gamma=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{gamma});
  if(zero_centered) {
    const std::array<float,1> one{1};auto offset=f.constant(dense({1}),bytes<float>(one));
    wide_gamma=core::binary(f,c::BinaryKind::kAdd,wide_gamma,offset);
    for(auto &value:scale)value+=1.0F;
  }
  auto normalized=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-6F}),{wide_input,wide_gamma});
  auto stored=f.one(c::Operation(c::Cast{.output_type=item.type}),{normalized});
  f.outputs={publish_float(f,stored)};
  for(int seed:{3,19}) {
    auto v=values(n,seed);auto output=sem::oracle::rms_norm_rows(v,scale,n,1.0e-6F);
    for(auto &value:output)value=mixed::rounded(item.type,value);
    f.replays.push_back({{mixed::floating_bytes(item.type,v)},{std::move(output)}});
  }
  return f;
}
inline Fragment convolution(const Case &item,const c::ModelArtifactProfileDescription *identity) {
  const auto channels=item.shape[0],taps=item.shape[2];Fragment f{.profile=identity,.region=0x0c052004};
  auto samples=f.input(dense({1,taps,channels},item.type));
  auto weights=values(channels*taps,7);
  auto physical=c::TensorDesc::dense(item.type,std::span(item.shape).first(item.rank)).value();
  auto stored=f.constant(physical,mixed::floating_bytes(item.type,weights));
  auto reshaped=f.one(c::Operation(c::View::reshape(c::TensorShape::create(std::array{channels,taps}).value()).value()),{stored});
  f.outputs={publish_float(f,f.one(c::model_ops::causal_conv1d(),{samples,reshaped}))};
  for(int seed:{3,19}) {
    auto input=values(taps*channels,seed);auto output=sem::oracle::causal_conv(input,weights,channels,taps);
    for(auto &value:output)value=mixed::rounded(item.type,value);
    f.replays.push_back({{mixed::floating_bytes(item.type,input)},{std::move(output)}});
  }
  return f;
}
inline Fragment vector_parameter(const Case &item,const c::ModelArtifactProfileDescription *identity) {
  const auto n=item.shape[0];Fragment f{.profile=identity,.region=0x0c052005};
  auto input=f.input(dense({1,n}));auto parameter=values(n,7);
  auto source=f.constant(dense({n},item.type),mixed::floating_bytes(item.type,parameter));
  if(item.type!=c::DataType::kFloat32)source=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{source});
  const bool decay=std::string_view(item.kind)=="log_decay";
  if(decay)source=core::unary(f,c::UnaryKind::kNeg,core::unary(f,c::UnaryKind::kExp,source));
  auto output=core::binary(f,decay?c::BinaryKind::kMultiply:c::BinaryKind::kAdd,input,source);
  f.outputs={output};
  for(int seed:{3,19}) {
    auto v=values(n,seed),expected=v;
    for(std::int64_t i=0;i<n;++i)expected[i]=decay?float(-double(v[i])*std::exp(double(parameter[i]))):v[i]+parameter[i];
    f.replays.push_back({{bytes(v)},{std::move(expected)}});
  }
  return f;
}
} // namespace cpu_c5_plain_fixture
