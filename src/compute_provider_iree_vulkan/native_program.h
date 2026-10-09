#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

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
  OfficialLinearBf16,
  OfficialLinearPowerF32,
  OfficialLinearPowerBf16,
  OfficialGrouped8F32,
  OfficialGrouped8Bf16,
  OfficialGrouped4Bf16,
  OfficialEncodedBlock128
};
enum class ArtifactScalar { Float32, Float16, BFloat16 };
constexpr bool artifact_program(ProgramKind kind) noexcept {
  return kind == ProgramKind::Nf4Decode || kind == ProgramKind::Nf4NestedDecode ||
         kind == ProgramKind::AwqDecode || kind == ProgramKind::Symmetric4Decode;
}
constexpr bool mlx_program(ProgramKind kind) noexcept {
  return kind >= ProgramKind::MlxAffineLinear &&
         kind <= ProgramKind::MlxAffineEmbedding;
}
constexpr bool mlx_grouped(ProgramKind kind) noexcept {
  return kind == ProgramKind::MlxAffineGrouped || kind == ProgramKind::MlxMx4Grouped ||
         kind == ProgramKind::MlxMx8Grouped;
}
constexpr bool official_program(ProgramKind kind) noexcept {
  return kind >= ProgramKind::OfficialWeightDecodeF32 &&
         kind <= ProgramKind::OfficialEncodedBlock128;
}
constexpr bool official_grouped(ProgramKind kind) noexcept {
  return kind == ProgramKind::OfficialGrouped8F32 ||
         kind == ProgramKind::OfficialGrouped8Bf16 ||
         kind == ProgramKind::OfficialGrouped4Bf16;
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
struct ProgramSpec final {
  ProgramKind kind = ProgramKind::ScanInclusive;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t depth = 0; // MatMul reduction dimension; ignored by Scan.
  // Artifact output [rows, columns]; original scale and final output rounding
  // remain distinct. The group is a prepared scalar, not a per-call search.
  std::uint32_t group_size = 64;
  ArtifactScalar scale_type = ArtifactScalar::Float32;
  ArtifactScalar output_type = ArtifactScalar::Float32;
  std::int64_t experts = 1; // Embedding: original vocabulary row count.
  std::int64_t selection_slots = 1;
  std::int64_t activation_slots = 1;
  float absmax_floor = 1.0e-10F;
};
struct BufferSpan final {
  VkBuffer buffer = VK_NULL_HANDLE;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
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
