#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "glm_gguf_model_types.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Converted GGUF projection leaves only. Original packed bytes and native
// activation quantization remain provider operands. No weight decoding,
// BF16 activation/store insertion, loading, state or execution lives here.
// In particular, the per-head K/V banks are ALREADY split and transposed by
// the converter; do not pass them through the original FP8 KV_b preparation.
namespace glm_gguf_projection_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
using Projection=glm_gguf_model_types::Projection;
inline Projection linear(sem::Fragment &f,Value input,Value weight) {
  const auto &a=f.description(input),&w=f.description(weight);
  const auto codec=c::gguf_quantized_type(w.storage_format());
  if(!codec||w.rank()!=2||w.data_type()!=c::DataType::kFloat32||
      a.rank()!=2||!a.is_plain()||!a.is_dense()||a.data_type()!=c::DataType::kFloat32||
      a.dimensions()[0]<=0||a.dimensions()[1]!=w.dimensions()[0])
    throw std::runtime_error("converted GLM projection needs actual F32 rows and original packed [K,N]");
  const auto canonical=c::gguf_weight_description(*codec,c::DataType::kFloat32,
      w.dimensions()[0],w.dimensions()[1]);
  if(!canonical||w!=*canonical)
    throw std::runtime_error("converted GLM projection changed the original packed row layout");
  const auto output=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,weight});
  return {input,weight,output};
}

// Router/index-weight projections are original F32[N,K], not packed and
// not BF16 metadata. A View gives MatMul its logical [K,N] without a copy.
inline Projection f32_linear(sem::Fragment &f,Value input,Value weight) {
  const auto &a=f.description(input),&w=f.description(weight);
  if(a.rank()!=2||!a.is_plain()||!a.is_dense()||a.data_type()!=c::DataType::kFloat32||
      w.rank()!=2||!w.is_plain()||!w.is_dense()||w.data_type()!=c::DataType::kFloat32||
      a.dimensions()[0]<=0||a.dimensions()[1]!=w.dimensions()[1])
    throw std::runtime_error("converted GLM F32 projection requires original [N,K] and actual F32 rows");
  constexpr std::array<std::int32_t,2> axes{1,0};
  const auto transposed=c::View::permute(axes);
  if(!transposed)throw std::runtime_error("converted GLM F32 projection transpose failed");
  const auto rhs=f.one(c::Operation(*transposed),{weight});
  return {input,weight,f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,rhs})};
}

// A[T,H,K] x original packed bank[H,N,K] -> F32[T,H,N]. Each slot h
// addresses bank h; there is no cross-head dense multiply or 64-Graph loop.
// Ordinary MatMul's current packed RHS admission is rank2 only. The existing
// GroupedMatMul is the natural public operation for these native Q8 head rows.
inline Projection per_head(sem::Fragment &f,Value input,Value weight) {
  const auto &a=f.description(input),&w=f.description(weight);
  const auto codec=c::gguf_quantized_type(w.storage_format());
  if(!codec||w.rank()!=3||w.data_type()!=c::DataType::kFloat32||
      a.rank()!=3||!a.is_plain()||!a.is_dense()||a.data_type()!=c::DataType::kFloat32||
      a.dimensions()[0]<=0||a.dimensions()[1]!=w.dimensions()[0]||
      a.dimensions()[2]!=w.dimensions()[2]||w.dimensions()[0]<=0||
      w.dimensions()[0]>std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("converted GLM per-head projection requires exact original head-bank geometry");
  const auto canonical=c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,
      w.dimensions()[0],w.dimensions()[1],w.dimensions()[2]);
  if(!canonical||w!=*canonical)
    throw std::runtime_error("converted GLM per-head bank is not its canonical native GGUF layout");
  const auto rows=std::uint64_t(a.dimensions()[0]),heads=std::uint64_t(w.dimensions()[0]);
  if(rows>std::numeric_limits<std::size_t>::max()/sizeof(std::int32_t)/heads)
    throw std::runtime_error("converted GLM head-order extent overflows");
  std::vector<std::int32_t> order(std::size_t(rows*heads),0);
  for(std::uint64_t row=0;row<rows;++row)for(std::uint64_t head=0;head<heads;++head)
    order[std::size_t(row*heads+head)]=std::int32_t(head);
  const auto ids=f.constant(sem::dense({a.dimensions()[0],w.dimensions()[0]},c::DataType::kInt32),sem::bytes(order));
  const auto output=f.one(c::model_ops::grouped_matmul(),{weight,input,ids});
  return {input,weight,output};
}

inline Value rms(sem::Fragment &f,Value input,Value gamma,float epsilon) {
  const auto &a=f.description(input),&g=f.description(gamma);
  if(a.rank()!=2||!a.is_plain()||!a.is_dense()||a.data_type()!=c::DataType::kFloat32||
      g!=sem::dense({a.dimensions()[1]})||!(epsilon>0.F)||!std::isfinite(epsilon))
    throw std::runtime_error("converted GLM RMS needs original absolute F32 gamma and actual F32 rows");
  return f.one(c::Operation(c::RmsNorm{.epsilon=epsilon}),{input,gamma});
}
struct PackedProjections final {
  Projection linear(sem::Fragment &f,Value input,Value weight) const {
    return f.description(weight).is_plain()?glm_gguf_projection_fragment::f32_linear(f,input,weight):
        glm_gguf_projection_fragment::linear(f,input,weight);
  }
  Projection per_head(sem::Fragment &f,Value input,Value weight) const {
    return glm_gguf_projection_fragment::per_head(f,input,weight);
  }
};
} // namespace glm_gguf_projection_fragment
