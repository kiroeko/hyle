#include "glm_gguf_compute_fragment.h"

namespace glm_gguf_compute_fragment {
Norm norm(sem::Fragment &f,Value actual,Value gamma,float epsilon) {
  return {actual,leaf::rms(f,actual,gamma,epsilon)};
}
}

namespace glm_gguf_compute_fragment {
Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> shape) {
  const auto dimensions=c::TensorShape::create({shape.begin(),shape.size()});
  if(!dimensions)throw std::runtime_error("converted GLM reshape extent differs");
  const auto view=c::View::reshape(*dimensions);
  if(!view)throw std::runtime_error("converted GLM reshape cannot retain this operand");
  return f.one(c::Operation(*view),{value});
}
}

namespace glm_gguf_compute_fragment {
Value slice(sem::Fragment &f,Value value,std::int32_t axis,std::int64_t first,std::int64_t count) {
  const auto view=c::View::slice(axis,first,count);
  if(!view)throw std::runtime_error("converted GLM slice extent differs");
  return f.one(c::Operation(*view),{value});
}
}

namespace glm_gguf_compute_fragment {
Value copy(sem::Fragment &f,Value value){return f.one(c::Operation(c::Copy{}),{value});}
}

namespace glm_gguf_compute_fragment {
Value embedding_q8(sem::Fragment &f,Value actual_ids,Value original_rows,const model::Geometry &g) {
  model::require_geometry(g);
  const auto &ids=f.description(actual_ids),&rows=f.description(original_rows);
  auto head=c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,
      g.hidden,g.vocabulary);
  if(!head)throw std::runtime_error("converted GLM vocabulary descriptor failed");
  const auto expected=head->transposed();
  if(!expected||rows!=*expected||ids.rank()!=1||!ids.is_plain()||!ids.is_dense()||
      ids.data_type()!=c::DataType::kInt32||ids.dimensions()[0]<=0)
    throw std::runtime_error("converted GLM embedding requires original Q8 rows and actual I32 token leases");
  return f.one(c::Operation(c::Gather{}),{original_rows,actual_ids});
}
}

namespace glm_gguf_compute_fragment {
glm_target_verifier_fragment::Ports append_q8_f32_verifier(sem::Fragment &f,
    Value actual_final_normalized_taps,Value original_head,const model::Geometry &g) {
  model::require_geometry(g);
  if(f.description(actual_final_normalized_taps)!=sem::dense({2,g.hidden}))
    throw std::runtime_error("converted GLM K1 head needs actual final-normalized F32[2,H]");
  auto expected=c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,
      g.hidden,g.vocabulary);
  if(!expected||f.description(original_head)!=*expected)
    throw std::runtime_error("converted GLM K1 head must retain its distinct original Q8 full vocabulary");
  glm_target_verifier_fragment::Ports result;
  // Exactly ONE native fullV M2 matrix multiply. Do not apply NextN shared
  // norm over target-normalized taps or reconstruct pending from host IDs.
  result.logits=leaf::linear(f,actual_final_normalized_taps,original_head).output;
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{result.logits},2);
  result.scores=top[0];result.ids=top[1];
  for(std::size_t row=0;row<result.row_ids.size();++row)
    result.row_ids[row]=copy(f,reshape(f,slice(f,result.ids,0,std::int64_t(row),1),{1}));
  return result;
}
}
