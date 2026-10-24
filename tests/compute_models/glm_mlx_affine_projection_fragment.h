#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"

// Original affine planes, without a loader, executor or storage conversion.
// In the Q32 domain BF16/F32 A plus original F16 SF/bias promotes to F32.
// Keep code*scale+bias in F32; narrowing decoded W to BF16 changes the model.
namespace glm_mlx_affine_projection_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
struct Parameters final {Value codes,scales,biases;std::uint32_t group=64;};
struct Projection final {Value input,output;std::uint32_t bits,group;};

inline std::uint32_t bits(const sem::Fragment &f,Value input,const Parameters &p,bool grouped) {
  const auto &a=f.description(input),&w=f.description(p.codes),&s=f.description(p.scales),&b=f.description(p.biases);
  const auto rank=std::size_t(grouped?3:2);
  if(!a.is_plain()||!a.is_dense()||a.rank()!=rank||
      (a.data_type()!=c::DataType::kBFloat16&&a.data_type()!=c::DataType::kFloat32)||
      !w.is_plain()||!w.is_dense()||w.rank()!=rank||w.data_type()!=c::DataType::kUInt32||
      !s.is_plain()||!s.is_dense()||s.rank()!=rank||s.data_type()!=c::DataType::kFloat16||s!=b||
      (p.group!=32&&p.group!=64&&p.group!=128))
    throw std::runtime_error("MLX original affine requires dense BF16/F32 A, U32 code words and unchanged F16 scale/bias planes");
  const auto k=a.dimensions()[rank-1],words=w.dimensions()[rank-1];
  if(k<=0||words<=0||words>std::numeric_limits<std::int64_t>::max()/32||k%p.group||
      words*32%k||s.dimensions()[rank-1]!=k/p.group)
    throw std::runtime_error("MLX original affine packed width/group extent differs from actual activation");
  for(std::size_t axis=0;axis+1<rank;++axis)
    if(w.dimensions()[axis]<=0||s.dimensions()[axis]!=w.dimensions()[axis])
      throw std::runtime_error("MLX original affine scale/bias rows or expert domain differ from code words");
  const auto width=words*32/k;
  if(width!=4&&width!=5&&width!=6&&width!=8)
    throw std::runtime_error("MLX original affine code width is not an existing supported continuous-row bitstream");
  return static_cast<std::uint32_t>(width);
}

inline Projection linear(sem::Fragment &f,Value input,const Parameters &p) {
  const auto width=bits(f,input,p,false);
  const auto output=f.one(c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine,width,p.group),
      {input,p.codes,p.scales,p.biases});
  if(f.description(output).data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("MLX mixed affine promotion lost its actual F32 result");
  return {input,output,width,p.group};
}
inline Projection grouped(sem::Fragment &f,Value input,const Parameters &p,Value actual_local_ids) {
  const auto width=bits(f,input,p,true);
  const auto output=f.one(c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine,width,p.group),
      {input,p.codes,p.scales,p.biases,actual_local_ids});
  if(f.description(output).data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("MLX mixed grouped affine promotion lost its actual F32 result");
  return {input,output,width,p.group};
}
} // namespace glm_mlx_affine_projection_fragment
