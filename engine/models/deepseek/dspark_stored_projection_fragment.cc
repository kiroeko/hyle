#include "dspark_stored_projection_fragment.h"

namespace dspark_stored_projection_fragment {
c::TensorDesc bfloat_matrix(std::int64_t outputs,std::int64_t width) {
  return c::TensorDesc::dense(c::DataType::kBFloat16,std::array{outputs,width}).value();
}
}

namespace dspark_stored_projection_fragment {
bool bfloat_matrix(const c::TensorDesc &weight,std::int64_t outputs,std::int64_t width) {
  return weight==bfloat_matrix(outputs,width);
}
}

namespace dspark_stored_projection_fragment {
sem::ValueId linear(sem::Fragment &f,sem::ValueId input,sem::ValueId weight) {
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
}

namespace dspark_stored_projection_fragment {
sem::ValueId dense_permute(sem::Fragment &f,sem::ValueId value) {
  constexpr std::array<std::int32_t,3> order{1,0,2};
  const auto view=f.one(c::Operation(c::View::permute(order).value()),{value});
  return f.one(c::Operation(c::Copy{}),{view});
}
}

namespace dspark_stored_projection_fragment {
sem::ValueId prepare_output_a(sem::Fragment &f,sem::ValueId original,
    std::int64_t groups,std::int64_t outputs,std::int64_t width) {
  if(f.description(original)!=bfloat_matrix(groups*outputs,width))
    throw std::runtime_error("output-a original BF16 row bank differs");
  const auto shape=c::TensorShape::create(std::array{groups,outputs,width}).value();
  const auto grouped=f.one(c::Operation(c::View::reshape(shape).value()),{original});
  constexpr std::array<std::int32_t,3> order{0,2,1};
  const auto transposed=f.one(c::Operation(c::View::permute(order).value()),{grouped});
  return f.one(c::Operation(c::Copy{}),{transposed});
}
}

namespace dspark_stored_projection_fragment {
sem::ValueId output_a(sem::Fragment &f,sem::ValueId input,sem::ValueId rhs,
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
}
