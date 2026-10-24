#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/graph.h"

// Cold typed projection leaves only. Plain BF16 weights explicitly consume
// BF16 activations and return F32 accumulation through DenseLinearF32. This
// is a Hyle precision policy, not equality to every peer's F32-A dispatch.
namespace dspark_stored_projection_fragment {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace sem=d5_qwen38_24t_semantic_fragments;

inline c::TensorDesc bfloat_matrix(std::int64_t outputs,std::int64_t width) {
  return c::TensorDesc::dense(c::DataType::kBFloat16,std::array{outputs,width}).value();
}
inline bool bfloat_matrix(const c::TensorDesc &weight,std::int64_t outputs,std::int64_t width) {
  return weight==bfloat_matrix(outputs,width);
}
inline sem::ValueId linear(sem::Fragment &f,sem::ValueId input,sem::ValueId weight) {
  const auto &description=f.description(weight);
  if(description.is_plain()) {
    const auto &activation=f.description(input);
    if(description.rank()!=2||description.data_type()!=c::DataType::kBFloat16||
        !description.is_dense()||activation.rank()!=2||
        (activation.data_type()!=c::DataType::kBFloat16&&activation.data_type()!=c::DataType::kFloat32)||
        activation.dimensions()[1]!=description.dimensions()[1])
      throw std::runtime_error("stored projection requires dense BF16 [N,K] and [T,K]");
    if(activation.data_type()!=c::DataType::kBFloat16)
      input=f.one(c::Operation(c::Cast{c::DataType::kBFloat16}),{input});
    return f.one(c::model_ops::dense_linear_f32(),{input,weight});
  }
  return f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,weight});
}
inline hyle::Result<g::ValueId> linear(g::GraphBuilder &f,g::ValueId input,
    g::ValueId weight,const c::TensorDesc &description) {
  if(description.is_plain()) {
    if(description.rank()!=2||description.data_type()!=c::DataType::kBFloat16||!description.is_dense())
      return std::unexpected(hyle::Status::InvalidArgument("stored projection requires dense BF16 [N,K]"));
    HYLE_ASSIGN_OR_RETURN(auto stored,f.add(c::Cast{c::DataType::kBFloat16},input));
    return f.add(c::model_ops::dense_linear_f32(),stored,weight);
  }
  return f.add(c::MatMul{.accumulation_type=c::DataType::kFloat32},input,weight);
}
inline sem::ValueId dense_permute(sem::Fragment &f,sem::ValueId value) {
  constexpr std::array<std::int32_t,3> order{1,0,2};
  const auto view=f.one(c::Operation(c::View::permute(order).value()),{value});
  return f.one(c::Operation(c::Copy{}),{view});
}
// Once-only preparation from the original grouped [G,N,K] BF16 rows to an
// ordinary batch-MatMul RHS [G,K,N]. The caller shares the actual output lease.
inline sem::ValueId prepare_output_a(sem::Fragment &f,sem::ValueId original,
    std::int64_t groups,std::int64_t outputs,std::int64_t width) {
  if(f.description(original)!=bfloat_matrix(groups*outputs,width))
    throw std::runtime_error("output-a original BF16 row bank differs");
  const auto shape=c::TensorShape::create(std::array{groups,outputs,width}).value();
  const auto grouped=f.one(c::Operation(c::View::reshape(shape).value()),{original});
  constexpr std::array<std::int32_t,3> order{0,2,1};
  const auto transposed=f.one(c::Operation(c::View::permute(order).value()),{grouped});
  return f.one(c::Operation(c::Copy{}),{transposed});
}
// The result is a REAL BF16 BMM result [T,G,N]. There is no observable raw
// F32 accumulator. A later F32 Cast is only an exact stored-result widening.
inline sem::ValueId output_a(sem::Fragment &f,sem::ValueId input,sem::ValueId rhs,
    std::int64_t groups,std::int64_t outputs,std::int64_t width) {
  const auto &activation=f.description(input);
  const auto expected=c::TensorDesc::dense(c::DataType::kBFloat16,std::array{groups,width,outputs}).value();
  if(f.description(rhs)!=expected||activation.rank()!=3||
      (activation.data_type()!=c::DataType::kBFloat16&&activation.data_type()!=c::DataType::kFloat32)||
      activation.dimensions()[1]!=groups||activation.dimensions()[2]!=width)
    throw std::runtime_error("output-a BF16 grouped input/RHS geometry differs");
  auto stored=f.one(c::Operation(c::Cast{c::DataType::kBFloat16}),{input});
  stored=dense_permute(f,stored);
  const auto projected=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{stored,rhs});
  return dense_permute(f,projected);
}
} // namespace dspark_stored_projection_fragment
