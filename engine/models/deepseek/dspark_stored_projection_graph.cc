#include "dspark_stored_projection_fragment.h"

namespace dspark_stored_projection_fragment {
hyle::Result<g::ValueId> linear(g::GraphBuilder &f,g::ValueId input,
    g::ValueId weight,const c::TensorDesc &description) {
  if(description.is_plain()) {
    if(description.rank()!=2||description.data_type()!=c::DataType::kBFloat16||!description.is_dense())
      return std::unexpected(hyle::Status::InvalidArgument("stored projection requires dense BF16 [N,K]"));
    HYLE_ASSIGN_OR_RETURN(auto stored,f.add(c::Cast{c::DataType::kBFloat16},input));
    return f.add(c::model_ops::dense_linear_f32(),stored,weight);
  }
  return f.add(c::MatMul{.accumulation_type=c::DataType::kFloat32},input,weight);
}
}
