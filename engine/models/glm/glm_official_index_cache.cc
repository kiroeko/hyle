#include "glm_official_index_cache.h"
#include "../detail/math_helpers.h"
#include <numeric>
#include <cmath>

namespace glm_official_index_cache {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
namespace glm=hyle::inference::model_definition::ops;
Value cast(sem::Fragment &f, Value x, c::DataType type) {
  return f.one(c::Operation(c::Cast{type}), {x});
}

Value bfloat_boundary(sem::Fragment &f, Value x) {
  return cast(f, cast(f, x, c::DataType::kBFloat16), c::DataType::kFloat32);
}

Quantized quantize(sem::Fragment &f, Value x, std::int64_t rows) {
  const auto &desc = f.description(x);
  if (desc.rank() != 2 || desc.dimensions()[0] != rows || desc.dimensions()[1] != kDepth)
    throw std::runtime_error("GLM quantizer requires the actual whole row extent");
  const auto input = cast(f, x, c::DataType::kFloat32);
  const auto transformed = f.one(c::Operation(c::Hadamard{}), {input});
  x = bfloat_boundary(f, transformed);
  const auto encoded = f.append(c::model_ops::float8_block_quantize(
      c::model_ops::Float8ScalePolicy::kPowerOfTwo, kFloor), {x}, 2);
  return {encoded[0], encoded[1], x, transformed, input};
}
Value score_encoded(sem::Fragment &f, Value bank, Value scales,
                           const Quantized &query, Value raw_weights,
                           std::int64_t visible) {
  std::vector<std::int32_t> prefix(std::size_t(visible), 0);
  std::iota(prefix.begin(), prefix.end(), 0);
  const auto valid = f.constant(sem::dense({visible}, c::DataType::kInt32), sem::bytes(prefix));
  const auto key = cast(f, f.one(c::Operation(c::Gather{}), {bank, valid}), c::DataType::kFloat32);
  const auto key_scale = glm::reshape(f, f.one(c::Operation(c::Gather{}), {scales, valid}), {1, visible});
  auto dot = f.one(c::Operation(c::MatMul{}),
      {cast(f, query.codes, c::DataType::kFloat32), glm::permute(f, key, {1, 0})});
  dot = glm::unary(f, c::UnaryKind::kRelu, dot);
  auto gate = glm::binary(f, c::BinaryKind::kMultiply, raw_weights,
      glm::filled(f, sem::dense({1}), 1.0F / std::sqrt(float(kHeads))));
  gate = glm::binary(f, c::BinaryKind::kMultiply,
      glm::reshape(f, gate, {kHeads, 1}), query.scale);
  gate = glm::binary(f, c::BinaryKind::kMultiply, gate,
      glm::filled(f, sem::dense({1}), 1.0F / std::sqrt(float(kDepth))));
  auto score = glm::reshape(f, glm::reduce(f,
      glm::binary(f, c::BinaryKind::kMultiply, dot, gate), 0), {1, visible});
  return glm::binary(f, c::BinaryKind::kMultiply, score, key_scale);
}
}
