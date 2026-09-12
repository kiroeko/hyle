#include "hyle/compute/operation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <utility>

namespace hyle::compute {
namespace {

Status validate_logical_f32(const TensorDesc& description) noexcept {
  if (!description.valid()) {
    return Status::InvalidArgument("operation tensor description is invalid");
  }
  if (description.data_type() != DataType::kFloat32) {
    return Status::Unimplemented(
        "operation currently requires float32 logical tensors");
  }
  return Status::Ok();
}

}  // namespace

std::string_view operation_name(OperationKind operation) noexcept {
  switch (operation) {
    case OperationKind::kRmsNorm:
      return "rms-norm";
    case OperationKind::kMatMul:
      return "matmul";
    case OperationKind::kSiluAndMul:
      return "silu-and-mul";
    case OperationKind::kUnknown:
      break;
  }
  return "unknown";
}

Result<TensorDesc> infer_output_description(
    Operation operation, std::span<const TensorDesc> inputs) noexcept {
  switch (operation.kind()) {
    case OperationKind::kRmsNorm: {
      if (inputs.size() != 1 && inputs.size() != 2) {
        return std::unexpected(Status::InvalidArgument(
            "RMSNorm requires one input and optional weight"));
      }
      const RmsNorm& rms_norm = *operation.rms_norm();
      if (!std::isfinite(rms_norm.epsilon) || rms_norm.epsilon < 0.0F) {
        return std::unexpected(Status::InvalidArgument(
            "RMSNorm epsilon must be finite and non-negative"));
      }
      Status status = validate_logical_f32(inputs[0]);
      if (!status.ok()) return std::unexpected(std::move(status));
      if (inputs[0].rank() == 0) {
        return std::unexpected(
            Status::InvalidArgument("RMSNorm input requires rank >= 1"));
      }
      if (inputs.size() == 2) {
        status = validate_logical_f32(inputs[1]);
        if (!status.ok()) return std::unexpected(std::move(status));
        if (inputs[1].rank() != 1 ||
            inputs[1].dimensions()[0] != inputs[0].dimensions().back()) {
          return std::unexpected(Status::InvalidArgument(
              "RMSNorm weight must be a logical [hidden] tensor"));
        }
      }
      return TensorDesc::dense(DataType::kFloat32, inputs[0].dimensions());
    }

    case OperationKind::kMatMul: {
      if (inputs.size() != 2) {
        return std::unexpected(
            Status::InvalidArgument("MatMul requires exactly two inputs"));
      }
      Status status = validate_logical_f32(inputs[0]);
      if (!status.ok()) return std::unexpected(std::move(status));
      status = validate_logical_f32(inputs[1]);
      if (!status.ok()) return std::unexpected(std::move(status));
      if (inputs[0].rank() != 2 || inputs[1].rank() != 2) {
        return std::unexpected(
            Status::Unimplemented("MatMul v2 requires rank-2 tensors"));
      }
      if (inputs[0].dimensions()[1] != inputs[1].dimensions()[0]) {
        return std::unexpected(
            Status::InvalidArgument("MatMul inner dimensions do not match"));
      }
      const std::array<std::int64_t, 2> output_shape{
          inputs[0].dimensions()[0], inputs[1].dimensions()[1]};
      return TensorDesc::dense(DataType::kFloat32, output_shape);
    }

    case OperationKind::kSiluAndMul: {
      if (inputs.size() != 1) {
        return std::unexpected(Status::InvalidArgument(
            "SiluAndMul requires exactly one packed input"));
      }
      Status status = validate_logical_f32(inputs[0]);
      if (!status.ok()) return std::unexpected(std::move(status));
      if (inputs[0].rank() == 0) {
        return std::unexpected(Status::InvalidArgument(
            "SiluAndMul input requires rank >= 1"));
      }
      const auto dimensions = inputs[0].dimensions();
      if ((dimensions.back() & 1) != 0) {
        return std::unexpected(Status::InvalidArgument(
            "SiluAndMul packed dimension must be even"));
      }
      std::array<std::int64_t, kMaxTensorRank> output_shape{};
      std::copy(dimensions.begin(), dimensions.end(), output_shape.begin());
      output_shape[dimensions.size() - 1] /= 2;
      return TensorDesc::dense(
          DataType::kFloat32,
          std::span(output_shape.data(), dimensions.size()));
    }
    case OperationKind::kUnknown:
      break;
  }
  return std::unexpected(Status::Unimplemented("operation kind is unknown"));
}

}  // namespace hyle::compute
