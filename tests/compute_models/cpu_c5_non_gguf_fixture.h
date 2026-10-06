#pragma once
#include "cpu_c5_gguf_fixture.h"
#include "cpu_c5_non_gguf_inputs.h"

namespace cpu_c5_non_gguf_fixture {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace in=cpu_c5_non_gguf_inputs;
using sem::Fragment;using sem::ValueId;using sem::bytes;using sem::dense;
using Half=hyle::simd::float16_t;using Bfloat=hyle::simd::bfloat16_t;

inline c::ModelArtifactProfileDescription profile(const in::Case &item) {
  const auto &artifact=in::artifacts[item.artifact];
  auto result=item.artifact==0?d5_qwen35_fixture::kPrimaryGgufProfile:
      (item.artifact==3?cpu_c4_model_branches::flash_profile:d5_qwen38_gdn_fixture::kOfficialProfile);
  result.artifact_repository=artifact.repository;result.artifact_revision=artifact.revision;
  result.variant=item.artifact==0?"bnb-4bit-double":(item.artifact==1?"awq-gemm-int4":
      (item.artifact==2?"compressed-tensors-symmetric-int4":"mlx-mixed-4bit"));
  result.manifest_digest=artifact.digest;result.format_family="safetensors";return result;
}
inline float rounded(c::DataType type,float value) {
  return type==c::DataType::kFloat16?float(half_test::decode(half_test::encode<Half>(value))):
      (type==c::DataType::kBFloat16?float(half_test::decode(half_test::encode<Bfloat>(value))):value);
}
inline std::vector<std::byte> floating_bytes(c::DataType type,std::span<const float> values) {
  if(type==c::DataType::kFloat16) {
    std::vector<Half> converted;converted.reserve(values.size());
    for(auto value:values)converted.push_back(half_test::encode<Half>(value));
    return bytes(converted);
  }
  if(type==c::DataType::kBFloat16) {
    std::vector<Bfloat> converted;converted.reserve(values.size());
    for(auto value:values)converted.push_back(half_test::encode<Bfloat>(value));
    return bytes(converted);
  }
  return bytes<float>(values);
}
inline std::vector<float> activation(std::int64_t rows,std::int64_t k,int pass) {
  std::vector<float> result(rows*k);
  for(std::int64_t row=0;row<rows;++row) {
    result[row*k]=pass?-.5f:.25f;result[row*k+1]=pass?.125f:-.125f;
    result[(row+1)*k-1]=pass?.25f:.5f;
  }
  return result;
}

inline Fragment artifact(const in::Case &item,const c::ModelArtifactProfileDescription *identity) {
  using Format=c::model_ops::WeightArtifactFormat;
  const std::int64_t n=item.n,k=item.k,group=item.group,groups=n*k/group;
  const auto block=std::uint32_t(item.group);
  const bool nf4=item.format==in::Format::kNf4,awq=item.format==in::Format::kAwq;
  const auto type=awq?c::DataType::kFloat16:c::DataType::kBFloat16;
  auto cold=std::make_shared<Fragment>(Fragment{.profile=identity,.region=0x0c050101});
  std::vector<ValueId> inputs;std::vector<float> expected(n*k);
  const auto constant=[&](c::TensorDesc desc,std::vector<std::byte> payload) {
    auto value=cold->constant(desc,std::move(payload));inputs.push_back(value);
  };
  if(nf4) {
    std::vector<std::uint8_t> codes(n*k/2,0x77),indices(groups);
    for(std::int64_t row=0;row<n;++row) {codes[row*k/2]=0xf0;codes[(row+1)*k/2-1]=0x7f;}
    std::vector<float> nested_absmax((groups+255)/256,.5f),nested(256),book(16);
    // Canonical NF4 endpoint/zero codes are exercised at every actual row.
    book={-1.f,-.6961928f,-.52507305f,-.3949175f,-.28444138f,-.18477343f,-.09105004f,0.f,
          .0795803f,.1609302f,.2461123f,.33791524f,.44070983f,.562617f,.72295684f,1.f};
    for(int i=0;i<256;++i)nested[i]=float(i)/256;
    for(std::int64_t g=0;g<groups;++g)indices[g]=std::uint8_t(128+g%8);
    constexpr float offset=.125f;
    for(std::int64_t row=0;row<n;++row)for(std::int64_t col:{std::int64_t{0},std::int64_t{1},k-1}) {
      const auto g=(row*k+col)/group;
      expected[row*k+col]=rounded(type,(col==1?-1.f:1.f)*(nested[indices[g]]*.5f+offset));
    }
    constant(dense({n,k/2},c::DataType::kUInt8),bytes(codes));
    constant(dense({groups},c::DataType::kUInt8),bytes(indices));
    constant(dense({(groups+255)/256}),bytes(nested_absmax));
    constant(dense({256}),bytes(nested));constant(dense({}),bytes(std::vector<float>{offset}));
    constant(dense({16}),bytes(book));
  } else {
    constexpr std::array<int,8> reverse{0,4,1,5,2,6,3,7};
    std::vector<std::int32_t> words(n*k/8,0x76543210);
    std::vector<float> scales(n*k/group);
    for(std::int64_t row=0;row<n;++row)for(std::int64_t g=0;g<k/group;++g) {
      const auto value=float(1+(row+g)%8)/8;
      scales[awq?g*n+row:row*(k/group)+g]=value;
      for(std::int64_t col=g*group;col<(g+1)*group;++col)
        expected[row*k+col]=rounded(type,float(awq?reverse[row%8]-3:int(col%8)-8)*value);
    }
    constant(dense(awq?std::initializer_list<std::int64_t>{k,n/8}:
        std::initializer_list<std::int64_t>{n,k/8},c::DataType::kInt32),bytes(words));
    if(awq)constant(dense({k/group,n/8},c::DataType::kInt32),bytes(std::vector<std::int32_t>(n*k/group/8,0x33333333)));
    constant(dense(awq?std::initializer_list<std::int64_t>{k/group,n}:
        std::initializer_list<std::int64_t>{n,k/group},type),floating_bytes(type,scales));
  }
  const auto op=c::model_ops::artifact_weight_decode(nf4?Format::kNf4DoubleQuant:
      (awq?Format::kAwqGemm4:Format::kCompressedTensorsSymmetric4),block);
  std::vector<c::TensorDesc> descriptions;for(auto value:inputs)descriptions.push_back(cold->description(value));
  std::array<c::TensorDesc,1> outputs;auto status=c::infer_output_descriptions(op,descriptions,outputs);
  if(!status.ok())throw std::runtime_error(std::string(status.message()));
  if(outputs[0].data_type()!=type)throw std::runtime_error("actual artifact default precision changed");
  auto output=cold->value(outputs[0],d5_qwen38_24t_full_router::ValueKind::kNodeOutput);
  cold->nodes.push_back({op,inputs,{output}});cold->outputs={output};cold->replays.push_back({{}, {expected}});
  Fragment f{.profile=identity,.region=0x0c050102};f.preparation=cold;
  auto a=f.input(dense({1,k}));auto w=f.constant(outputs[0],{});f.prepared_constants={0};
  auto transposed=f.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{w});
  auto cast=f.one(c::Operation(c::Cast{.output_type=type}),{a});
  auto y=f.one(c::Operation(c::MatMul{}),{cast,transposed});
  f.outputs={f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{y})};
  for(int pass:{0,1}) {
    auto values=activation(1,k,pass);std::vector<float> answer(n);
    for(std::int64_t row=0;row<n;++row) {
      const float sum=expected[row*k]*values[0]+expected[row*k+1]*values[1]+expected[(row+1)*k-1]*values[k-1];
      answer[row]=rounded(type,sum);
    }
    f.replays.push_back({{bytes(values)},{std::move(answer)}});
  }
  return f;
}

inline Fragment mlx(const in::Case &item,const c::ModelArtifactProfileDescription *identity,
                    std::int64_t n,std::int64_t first=0) {
  using Format=c::model_ops::MlxWeightFormat;
  const bool affine=item.format==in::Format::kAffine,mx8=item.format==in::Format::kMx8;
  const auto format=affine?Format::kAffine4:(mx8?Format::kMxfp8:Format::kMxfp4);
  const std::int64_t k=item.k,group=item.group,pack=mx8?4:8;
  const auto block=std::uint32_t(item.group);
  const std::int64_t e=item.grouped?std::min<std::int64_t>(item.e,8):1,slots=item.grouped?3:1;
  constexpr auto type=c::DataType::kBFloat16;
  Fragment f{.profile=identity,.region=0x0c050103};
  std::vector<std::uint32_t> words(e*n*k/pack,mx8?0x4038b8c0u:0x76543210u);
  std::vector<float> scales(e*n*k/group),bias(scales.size());
  std::vector<std::uint8_t> exponents(scales.size());
  for(std::int64_t expert=0;expert<e;++expert)for(std::int64_t row=0;row<n;++row)for(std::int64_t b=0;b<k/group;++b) {
    const auto index=(expert*n+row)*(k/group)+b;
    scales[index]=float(1+(first+row+expert+b)%8)/8;
    bias[index]=float((first+row)%3-1)/16;
    exponents[index]=std::uint8_t(126+(first+row+expert+b)%3);
  }
  const auto weight_desc=item.grouped?dense({e,n,k/pack},c::DataType::kUInt32):dense({n,k/pack},c::DataType::kUInt32);
  const auto scale_desc=item.grouped?dense({e,n,k/group},affine?type:c::DataType::kUInt8):dense({n,k/group},affine?type:c::DataType::kUInt8);
  auto w=f.constant(weight_desc,bytes(words));
  auto s=f.constant(scale_desc,affine?floating_bytes(type,scales):bytes(exponents));
  ValueId z;if(affine)z=f.constant(scale_desc,floating_bytes(type,bias));
  auto input=f.input(item.embedding?dense({4},c::DataType::kInt32):
      (item.grouped?dense({1,slots,k}):dense({1,k})));
  ValueId ids;if(item.grouped)ids=f.input(dense({1,slots},c::DataType::kInt32));
  auto a=item.embedding?input:f.one(c::Operation(c::Cast{.output_type=type}),{input});
  ValueId y;
  if(item.embedding)y=f.one(c::model_ops::mlx_quantized_embedding(block),{w,s,z,input});
  else if(item.grouped)y=affine?f.one(c::model_ops::mlx_grouped_matmul(format,block),{a,w,s,z,ids}):
      f.one(c::model_ops::mlx_grouped_matmul(format,block),{a,w,s,ids});
  else y=affine?f.one(c::model_ops::mlx_quantized_matmul(format,block),{a,w,s,z}):
      f.one(c::model_ops::mlx_quantized_matmul(format,block),{a,w,s});
  f.outputs={f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{y})};
  if(item.embedding)f.invalid_index_input=0;else if(item.grouped)f.invalid_index_input=1;
  constexpr std::array<float,8> fp4{0,.5f,1,1.5f,2,3,4,6};
  constexpr std::array<float,4> fp8{-2,-1,1,2};
  const auto decoded=[&](std::int64_t expert,std::int64_t row,std::int64_t col) {
    const auto index=(expert*n+row)*(k/group)+col/group;
    return affine?float(col%8)*scales[index]+bias[index]:
        (mx8?fp8[col%4]:fp4[col%8])*std::ldexp(1.f,int(exponents[index])-127);
  };
  for(int pass:{0,1}) {
    if(item.embedding) {
      const std::array<std::int32_t,4> selected{pass?1:0,std::int32_t(n-1),std::int32_t(n/2),std::int32_t(n/2)};
      std::vector<float> answer(4*k);
      for(int row=0;row<4;++row)for(std::int64_t col=0;col<k;++col)answer[row*k+col]=rounded(type,decoded(0,selected[row],col));
      f.replays.push_back({{bytes<std::int32_t>(selected)},{std::move(answer)}});
    } else {
      auto values=activation(slots,k,pass);std::vector<std::int32_t> selected(slots);
      std::vector<float> answer(slots*n);
      for(std::int64_t slot=0;slot<slots;++slot) {
        selected[slot]=std::int32_t(item.grouped?(slot?e-1:pass):0);
        for(std::int64_t row=0;row<n;++row) {
          const float sum=decoded(selected[slot],row,0)*values[slot*k]+decoded(selected[slot],row,1)*values[slot*k+1]+
              decoded(selected[slot],row,k-1)*values[(slot+1)*k-1];
          answer[slot*n+row]=rounded(type,sum);
        }
      }
      std::vector<std::vector<std::byte>> inputs{bytes(values)};if(item.grouped)inputs.push_back(bytes(selected));
      f.replays.push_back({std::move(inputs),{std::move(answer)}});
    }
  }
  return f;
}
} // namespace cpu_c5_non_gguf_fixture
