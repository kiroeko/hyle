#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "native_device.h"

namespace hyle::compute::iree_vulkan {
enum class ProgramKind {
  ScanInclusive,
  ScanExclusive,
  Float8Block128,
  Float8Group32,
  Float32ToBFloat16,
  Float16ToBFloat16,
  Nf4Decode,
  Nf4NestedDecode,
  AwqDecode,
  Symmetric4Decode,
  MlxAffineLinear,
  MlxAffineGrouped,
  MlxMx4Linear,
  MlxMx4Grouped,
  MlxMx8Linear,
  MlxMx8Grouped,
  MlxAffineEmbedding,
  OfficialWeightDecodeF32,
  OfficialWeightDecodeBf16,
  OfficialRoundtrip8Group64,
  OfficialRoundtrip8Group128,
  OfficialRoundtrip4Group32,
  OfficialLinearF32,
  OfficialLinearPowerF32,
  OfficialGrouped8F32,
  OfficialGrouped4F32,
  OfficialEncodedBlock128,
  Iq1XsGrouped,
  Iq1XxsGrouped,
  Iq1XxxsGrouped,
  Asymmetric4Decode,
  OfficialGrouped8PowerF32,
  MlxAffineLinearAsStored,
  MlxAffineGroupedAsStored,
  OfficialLinearReciprocalF32,
  OfficialGrouped8ReciprocalF32,
  OfficialQuantizeDivide,
  OfficialQuantizePower,
  OfficialQuantizeReciprocal
};
enum class ArtifactScalar { Float32, Float16, BFloat16 };
constexpr bool iq1_program(ProgramKind kind) noexcept {
  return kind >= ProgramKind::Iq1XsGrouped && kind <= ProgramKind::Iq1XxxsGrouped;
}
constexpr std::int64_t iq1_block_bytes(ProgramKind kind) noexcept {
  return kind == ProgramKind::Iq1XsGrouped ? 46
         : kind == ProgramKind::Iq1XxsGrouped ? 42 : 38;
}
constexpr bool artifact_program(ProgramKind kind) noexcept {
  return kind == ProgramKind::Nf4Decode || kind == ProgramKind::Nf4NestedDecode ||
         kind == ProgramKind::AwqDecode || kind == ProgramKind::Symmetric4Decode ||
         kind == ProgramKind::Asymmetric4Decode;
}
constexpr bool mlx_program(ProgramKind kind) noexcept {
  return (kind >= ProgramKind::MlxAffineLinear &&
          kind <= ProgramKind::MlxAffineEmbedding) ||
         kind == ProgramKind::MlxAffineLinearAsStored ||
         kind == ProgramKind::MlxAffineGroupedAsStored;
}
constexpr bool mlx_as_stored(ProgramKind kind) noexcept {
  return kind == ProgramKind::MlxAffineLinearAsStored ||
         kind == ProgramKind::MlxAffineGroupedAsStored;
}
constexpr bool mlx_affine(ProgramKind kind) noexcept {
  return kind == ProgramKind::MlxAffineLinear ||
         kind == ProgramKind::MlxAffineGrouped ||
         kind == ProgramKind::MlxAffineEmbedding || mlx_as_stored(kind);
}
constexpr bool mlx_grouped(ProgramKind kind) noexcept {
  return kind == ProgramKind::MlxAffineGrouped || kind == ProgramKind::MlxMx4Grouped ||
         kind == ProgramKind::MlxMx8Grouped || kind == ProgramKind::MlxAffineGroupedAsStored;
}
constexpr bool official_quantize(ProgramKind kind) noexcept {
  return kind >= ProgramKind::OfficialQuantizeDivide &&
         kind <= ProgramKind::OfficialQuantizeReciprocal;
}
constexpr bool official_program(ProgramKind kind) noexcept {
  return (kind >= ProgramKind::OfficialWeightDecodeF32 &&
          kind <= ProgramKind::OfficialEncodedBlock128) ||
         kind == ProgramKind::OfficialGrouped8PowerF32 ||
         kind == ProgramKind::OfficialLinearReciprocalF32 ||
         kind == ProgramKind::OfficialGrouped8ReciprocalF32 || official_quantize(kind);
}
constexpr bool official_grouped(ProgramKind kind) noexcept {
  return kind == ProgramKind::OfficialGrouped8F32 ||
         kind == ProgramKind::OfficialGrouped4F32 ||
         kind == ProgramKind::OfficialGrouped8PowerF32 ||
         kind == ProgramKind::OfficialGrouped8ReciprocalF32;
}
constexpr bool official_roundtrip(ProgramKind kind) noexcept {
  return kind >= ProgramKind::OfficialRoundtrip8Group64 &&
         kind <= ProgramKind::OfficialRoundtrip4Group32;
}
constexpr bool official_checked(ProgramKind kind) noexcept {
  return official_program(kind) && kind != ProgramKind::OfficialWeightDecodeF32 &&
         kind != ProgramKind::OfficialWeightDecodeBf16 &&
         kind != ProgramKind::OfficialEncodedBlock128;
}
// One current private by-value ABI. The concrete version is part of exported
// NativeProgram::create symbols, so old callers cannot silently pass old fields.
struct ProgramSpecV4 final {
  ProgramKind kind = ProgramKind::ScanInclusive;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t depth = 0; // MatMul reduction dimension; ignored by Scan.
  // Artifact output [rows, columns]; original scale and final output rounding
  // remain distinct. The group is a prepared scalar, not a per-call search.
  std::uint32_t group_size = 64;
  std::uint32_t bits = 4; // MLX affine 4/5/6/8; MX formats retain fixed 4/8.
  // MLX dot inputs retain their actual storage types. Mixed affine sources
  // widen only selected scalars in the original group program.
  ArtifactScalar activation_type = ArtifactScalar::Float32;
  ArtifactScalar scale_type = ArtifactScalar::Float32;
  ArtifactScalar output_type = ArtifactScalar::Float32;
  std::int64_t experts = 1; // Embedding: original vocabulary row count.
  std::int64_t selection_slots = 1;
  std::int64_t activation_slots = 1;
  // MLX grouped only: original activation token stride in scalar elements.
  // Zero uses the dense token stride; activation_slots=1 also represents a
  // true zero-slot-stride View, without a T*S*K staging expansion.
  std::int64_t activation_token_stride = 0;
  float absmax_floor = 1.0e-10F;
};
using ProgramSpec = ProgramSpecV4;
// MLX keeps columns=N and depth=K in the original packed [N,K] bank.
// AsStored changes the activation/reduction axis, never the storage geometry.
constexpr std::int64_t mlx_input_depth(const ProgramSpec &spec) noexcept {
  return mlx_as_stored(spec.kind) ? spec.columns : spec.depth;
}
constexpr std::int64_t mlx_output_columns(const ProgramSpec &spec) noexcept {
  return mlx_as_stored(spec.kind) ? spec.depth : spec.columns;
}
// Flat physical input span, excluding padding after the last token. Admission
// checks the bounded I32 shader-index domain before forming any products.
constexpr std::int64_t mlx_activation_elements(const ProgramSpec &spec) noexcept {
  constexpr std::int64_t limit = INT32_MAX / 4;
  if (!mlx_grouped(spec.kind)) return 0;
  const auto depth = mlx_input_depth(spec);
  if (spec.rows <= 0 || depth <= 0 || spec.selection_slots <= 0 ||
      spec.rows % spec.selection_slots ||
      (spec.activation_slots != 1 && spec.activation_slots != spec.selection_slots) ||
      depth > limit / spec.activation_slots) return 0;
  const auto last = spec.activation_slots * depth;
  const auto tokens = spec.rows / spec.selection_slots;
  const auto stride = spec.activation_token_stride ? spec.activation_token_stride : last;
  if (stride < last || (tokens > 1 && stride > (limit - last) / (tokens - 1))) return 0;
  return (tokens - 1) * stride + last;
}

struct BufferSpan final {
  VkBuffer buffer = VK_NULL_HANDLE;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
};
// Only ready immutable native storage. The HAL wrapper retains owner until
// its last reference retires; it never maps or frees the caller's allocation.
struct BorrowedInput final {
  std::size_t index = 0;
  VkDevice device = VK_NULL_HANDLE;
  std::uint64_t buffer_bytes = 0;
  BufferSpan storage;
  std::shared_ptr<const void> owner;
};
struct WorkState final {
  Status status;
  bool pending = false;
};
using CompletionNotify = void (*)(void *, const Status &) noexcept;

// One original VM context and native input storage per invocation. Host calls
// are thread-compatible. GPU input copies must be enqueued on the owner's
// queue before invoke. Outputs remain private until retirement and, for Scan,
// the caller's overflow-flag readback. They cannot be published on failure.
class NativeProgram final {
public:
  NativeProgram(const NativeProgram &) = delete;
  NativeProgram &operator=(const NativeProgram &) = delete;
  HYLE_IREE_VULKAN_INTERNAL_API ~NativeProgram();
  HYLE_IREE_VULKAN_INTERNAL_API static Result<std::unique_ptr<NativeProgram>>
  create(NativeDevice device, ProgramSpec spec) noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API static Result<std::unique_ptr<NativeProgram>>
  create(NativeDevice device, ProgramSpec spec,
         std::span<const BorrowedInput> borrowed) noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API std::size_t input_count() const noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API Result<BufferSpan>
  input(std::size_t index) const noexcept;
  // The optional notification runs on the original proactor after GPU output
  // readiness. It may inspect output, but must not wait, destroy or re-invoke
  // this slot. wait/poll include the notification's return in retirement.
  HYLE_IREE_VULKAN_INTERNAL_API WorkState
  invoke(CompletionNotify notify = nullptr, void *user_data = nullptr) noexcept;
  // Only from this slot's completion notification, after enqueueing native
  // commands on the shared physical queue. Extends the slot through that
  // accepted native batch and a second notification on the original proactor.
  // This is not a drain proof for unsubmitted/deferred provider work.
  HYLE_IREE_VULKAN_INTERNAL_API Status
  after_native_submissions(CompletionNotify notify, void *user_data) noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API WorkState poll() noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API WorkState wait() noexcept;
  HYLE_IREE_VULKAN_INTERNAL_API Result<BufferSpan>
  output(std::size_t index) const noexcept;

private:
  struct State;
  explicit NativeProgram(std::unique_ptr<State> state) noexcept;
  WorkState observe(bool wait) noexcept;
  std::unique_ptr<State> state_;
};
} // namespace hyle::compute::iree_vulkan
