#pragma once

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Model-local vocabulary math. Actual descriptors determine the storage
// operation; no provider, artifact name, weight decoding or execution is here.
namespace dspark_vocabulary_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
inline bool bfloat_table(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden) {
  return table==sem::dense({vocabulary,hidden},c::DataType::kBFloat16);
}
inline bool packed_head(const c::TensorDesc &head,std::int64_t vocabulary,std::int64_t hidden) {
  const auto codec=c::gguf_quantized_type(head.storage_format());
  if(!codec)return false;
  const auto canonical=c::gguf_weight_description(*codec,c::DataType::kFloat32,hidden,vocabulary);
  return canonical&&head==*canonical;
}
inline bool embedding_table(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden) {
  if(bfloat_table(table,vocabulary,hidden))return true;
  const auto head=table.transposed();return head&&packed_head(*head,vocabulary,hidden);
}
inline void require_embedding(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden) {
  if(vocabulary<=0||hidden<=0||!embedding_table(table,vocabulary,hidden))
    throw std::runtime_error("DSpark embedding needs a full BF16 table or canonical packed row table");
}
inline void require_head(const c::TensorDesc &head,std::int64_t vocabulary,std::int64_t hidden) {
  if(vocabulary<=0||hidden<=0||(!bfloat_table(head,vocabulary,hidden)&&!packed_head(head,vocabulary,hidden)))
    throw std::runtime_error("DSpark head needs a full BF16 table or canonical packed RHS");
}
struct Embedding final {Value decoded,stored;};
inline Embedding append_embedding(sem::Fragment &f,Value table,Value ids,
    std::int64_t vocabulary,std::int64_t hidden) {
  require_embedding(f.description(table),vocabulary,hidden);
  const auto decoded=f.one(c::Operation(c::Gather{}),{table,ids});
  // A packed row decode returns F32. The DSpark four-stream input is BF16;
  // retain the actual decoded value and its explicit model storage boundary.
  const auto stored=f.description(decoded).data_type()==c::DataType::kBFloat16?decoded:
      f.one(c::Operation(c::Cast{.output_type=c::DataType::kBFloat16}),{decoded});
  return {decoded,stored};
}
inline Value append_head(sem::Fragment &f,Value actual_bfloat,Value weight,
    std::int64_t vocabulary,std::int64_t hidden) {
  require_head(f.description(weight),vocabulary,hidden);
  const auto &input=f.description(actual_bfloat);
  if(input.rank()!=2||!input.is_plain()||!input.is_dense()||input.data_type()!=c::DataType::kBFloat16||
      input.dimensions()[0]<=0||input.dimensions()[1]!=hidden)
    throw std::runtime_error("DSpark head needs its actual BF16 normalized producer");
  if(bfloat_table(f.description(weight),vocabulary,hidden))
    return f.one(c::model_ops::dense_linear_f32(),{actual_bfloat,weight});
  const auto visible=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{actual_bfloat});
  return f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{visible,weight});
}
} // namespace dspark_vocabulary_fragment
