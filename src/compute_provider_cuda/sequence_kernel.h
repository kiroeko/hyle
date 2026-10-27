#pragma once

#include "native_kernel.h"
#include <cstdint>

namespace hyle::compute::cuda_native::native {

enum class SequenceKind { kGatedDeltaNet, kCausalConv };
struct SequenceLayout final {
  SequenceKind kind = SequenceKind::kGatedDeltaNet;
  std::uint64_t rows = 0, batches = 0;
  std::uint64_t key_heads = 0, key_depth = 0, value_heads = 0, value_depth = 0;
  std::uint64_t gate_width = 0, snapshots = 1;
  std::uint64_t channels = 0, taps = 0;
  float scale = 1.0F;
};
struct SequenceProgram final {
  const void *validate = nullptr, *payload = nullptr;
  unsigned validation_blocks = 0, payload_blocks = 0;
  SequenceLayout layout{};
};
// All pointers remain explicit admitted ports; no allocation or private state.
struct SequenceBindings final {
  const void *inputs[7]{};
  void *outputs[2]{};
};
cudaError_t sequence_prepare(ElementType, const SequenceLayout &,
                             SequenceProgram *) noexcept;
cudaError_t sequence_launch(const SequenceProgram &, SequenceBindings,
                            unsigned *status, cudaStream_t) noexcept;

} // namespace hyle::compute::cuda_native::native
