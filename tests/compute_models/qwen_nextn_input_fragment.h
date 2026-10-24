#pragma once

#include "d5_moe_semantic_fragments.h"

// Shared cold input equations for converted F32 and original BF16 Qwen NextN
// parameters. No model profile, payload loader, native fixture or executor.
namespace qwen_nextn_input_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
using sem::Fragment;
using sem::ValueId;
using sem::dense;
struct InputBoundary final {ValueId hidden,embedding,joined;};
inline InputBoundary append_input(Fragment &f,ValueId hidden_input,ValueId embedding_input,
    ValueId hidden_scale,ValueId embedding_scale,c::DataType storage=c::DataType::kFloat32) {
  if(storage!=c::DataType::kFloat32&&storage!=c::DataType::kBFloat16)
    throw std::runtime_error("Qwen NextN input storage must be explicit F32 or BF16");
  const auto normalize=[&](ValueId actual,ValueId scale) {
    if(storage==c::DataType::kBFloat16) {
      const auto &description=f.description(actual);
      if(description.rank()!=2||!description.is_plain()||!description.is_dense()||
          description.data_type()!=c::DataType::kBFloat16||
          f.description(scale)!=dense({description.dimensions()[1]}))
        throw std::runtime_error("original Qwen input needs BF16 rows and explicit effective F32 gamma");
      actual=f.one(c::Operation(c::Cast{.output_type=c::DataType::kFloat32}),{actual});
    }
    const auto raw=f.one(c::Operation(c::RmsNorm{.epsilon=1e-6F}),{actual,scale});
    return storage==c::DataType::kBFloat16?
        f.one(c::Operation(c::Cast{.output_type=c::DataType::kBFloat16}),{raw}):raw;
  };
  const auto hn=normalize(hidden_input,hidden_scale);
  const auto en=normalize(embedding_input,embedding_scale);
  return {hn,en,f.one(c::Operation(c::Concat{.axis=1}),{en,hn})};
}
} // namespace qwen_nextn_input_fragment
