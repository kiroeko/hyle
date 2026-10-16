#include "hyle/compute/sampling_ops.h"

#include <array>
#include <climits>

namespace hyle::compute::sampling_ops {
namespace {
Status validate_counters(const TensorDesc &input) noexcept {
  if (!input.valid() || !input.is_plain() || !input.is_dense() ||
      input.data_type() != DataType::kUInt32 || input.rank() != 2 ||
      input.dimensions()[1] != 4)
    return Status::InvalidArgument(
        "sampling counters require dense U32[R,4] key/draw words");
  return Status::Ok();
}
} // namespace

Status RandomBits::infer_output_descriptions(
    std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs) const noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return Status::InvalidArgument("RandomBits requires one input/output");
  HYLE_RETURN_IF_ERROR(validate_counters(inputs[0]));
  if (word_count > INT32_MAX)
    return Status::OutOfRange("RandomBits word count exceeds I32");
  const std::array<std::int64_t, 2> shape{inputs[0].dimensions()[0],
                                          word_count};
  HYLE_ASSIGN_OR_RETURN(auto output, TensorDesc::dense(DataType::kUInt32, shape));
  outputs[0] = output;
  return Status::Ok();
}

Status Categorical::infer_output_descriptions(
    std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs) const noexcept {
  if (inputs.size() != 2 ||
      outputs.size() != static_cast<std::size_t>(return_log_probability ? 2 : 1))
    return Status::InvalidArgument("Categorical input/output arity differs");
  HYLE_RETURN_IF_ERROR(validate_counters(inputs[1]));
  const auto &logits = inputs[0];
  const auto type = logits.data_type();
  if (!logits.valid() || !logits.is_plain() || !logits.is_dense() ||
      logits.rank() != 2 ||
      (type != DataType::kFloat16 && type != DataType::kBFloat16 &&
       type != DataType::kFloat32))
    return Status::InvalidArgument(
        "Categorical requires dense F16/BF16/F32[R,V] log-weights");
  if (logits.dimensions()[0] != inputs[1].dimensions()[0])
    return Status::InvalidArgument("Categorical counter row count differs");
  if (logits.dimensions()[1] == 0)
    return Status::InvalidArgument("Categorical requires positive vocabulary");
  if (logits.dimensions()[1] > INT32_MAX)
    return Status::OutOfRange("Categorical vocabulary exceeds I32 indices");
  const std::array<std::int64_t, 1> shape{logits.dimensions()[0]};
  HYLE_ASSIGN_OR_RETURN(auto indices,
                        TensorDesc::dense(DataType::kInt32, shape));
  outputs[0] = indices;
  if (return_log_probability) {
    HYLE_ASSIGN_OR_RETURN(auto scores,
                          TensorDesc::dense(DataType::kFloat32, shape));
    outputs[1] = scores;
  }
  return Status::Ok();
}
} // namespace hyle::compute::sampling_ops
