#include "math_helpers.h"

namespace hyle::inference::model_definition::ops {
ValueId reshape(Fragment &f, ValueId input,
                       std::initializer_list<std::int64_t> shape) {
  return f.one(c::Operation(c::View::reshape(
                   c::TensorShape::create({shape.begin(), shape.size()}).value())
                               .value()),
               {input});
}

ValueId unary(Fragment &f, c::UnaryKind kind, ValueId input) {
  return f.one(c::Operation(c::Unary{kind}), {input});
}

ValueId binary(Fragment &f, c::BinaryKind kind, ValueId a, ValueId b) {
  return f.one(c::Operation(c::Binary{kind}), {a, b});
}

ValueId filled(Fragment &f, const c::TensorDesc &desc, float value) {
  return f.constant(desc, bytes(std::vector<float>(*desc.element_count(), value)));
}

ValueId sum(Fragment &f, ValueId input, std::int32_t axis) {
  return f.one(c::Operation(c::Reduce{
                   .kind = c::ReductionKind::kSum,
                   .axes = c::Axes::from(std::array{axis}).value(),
                   .keep_dimensions = true,
                   .accumulation_type = c::DataType::kFloat32}), {input});
}

ValueId reduce(Fragment &f, ValueId value, std::int32_t axis,
               c::ReductionKind kind) {
  return f.one(c::Operation(c::Reduce{
      .kind = kind, .axes = c::Axes::from(std::span(&axis, 1)).value(),
      .accumulation_type = c::DataType::kFloat32}), {value});
}

ValueId slice(Fragment &f, ValueId input, std::int32_t axis,
                      std::int64_t first, std::int64_t count) {
  auto view = f.one(c::Operation(c::View::slice(axis, first, count).value()), {input});
  return f.one(c::Operation(c::Copy{}), {view});
}

ValueId permute(Fragment &f, ValueId input,
                        std::initializer_list<std::int32_t> axes) {
  auto view = f.one(c::Operation(c::View::permute({axes.begin(), axes.size()}).value()), {input});
  return f.one(c::Operation(c::Copy{}), {view});
}

ValueId concat(Fragment &f, ValueId a, ValueId b, std::int32_t axis) {
  return f.one(c::Operation(c::Concat{.axis = axis}), {a, b});
}
}
