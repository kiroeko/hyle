#include "glm_target_verifier_fragment.h"

namespace glm_target_verifier_fragment {
Ports append(sem::Fragment &fragment, Value final_normalized_taps,
                    Value original_shared_head, std::int64_t hidden ,
                    std::int64_t vocabulary ) {
  if (hidden <= 0 || vocabulary <= 0 ||
      vocabulary > std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("GLM K1 verifier needs positive hidden and an I32 vocabulary extent");
  const auto tap_description = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{kRows, hidden});
  const auto head_description = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{vocabulary, hidden});
  if (!tap_description || !head_description || !tap_description->required_bytes() ||
      !head_description->required_bytes() ||
      final_normalized_taps.ordinal() >= fragment.descriptions.size() ||
      original_shared_head.ordinal() >= fragment.descriptions.size() ||
      fragment.description(final_normalized_taps) != *tap_description ||
      fragment.description(original_shared_head) != *head_description)
    throw std::runtime_error("GLM K1 verifier needs bounded final normalized BF16[2,H] taps and original BF16[V,H] head");

  // The boundary is AFTER target model.norm. Never apply the distinct NextN
  // shared_head.norm, normalize again, or narrow a caller's F32 tap implicitly.
  // Constant ownership/authentication belongs to the caller's ordinary Graph.
  Ports result;
  result.logits = fragment.one(c::model_ops::dense_linear_f32(),
      {final_normalized_taps, original_shared_head});
  const auto top = fragment.append(c::Operation(c::TopK{.count = 1, .stable = true}),
      {result.logits}, 2);
  result.scores = top[0]; result.ids = top[1];
  const auto scalar_shape = c::TensorShape::create(std::array<std::int64_t, 1>{1}).value();
  for (std::size_t row = 0; row < result.row_ids.size(); ++row) {
    const auto one = fragment.one(c::Operation(c::View::slice(0, std::int64_t(row), 1).value()),
        {result.ids});
    const auto flattened = fragment.one(c::Operation(c::View::reshape(scalar_shape).value()), {one});
    result.row_ids[row] = fragment.one(c::Operation(c::Copy{}), {flattened});
  }
  return result;
}
}
