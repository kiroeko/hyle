#pragma once

#include "vulkan_program_internal.h"

namespace hyle::compute::vulkan::detail {

// One prepared provider boundary. Native copies are owned by the route; the
// original program owns its VM, compressed inputs and private output storage.
struct IreeStage final {
  struct Input final {
    std::uint32_t value = 0;
    VkDeviceSize offset = 0;
    iree_vulkan::BufferSpan destination;
  };
  Run *parent = nullptr;
  std::unique_ptr<iree_vulkan::NativeProgram> program;
  std::array<Input, 6> inputs;
  std::size_t input_count = 0;
  std::uint32_t output = 0;
  std::size_t output_bytes = 0;
  std::shared_ptr<Run> input_copy, output_copy, flag_copy;
  std::shared_ptr<Buffer> flag;
  bool reading_flag = false;
  Status bind() noexcept;
  static void completed(void *, const Status &) noexcept;
};

Result<std::shared_ptr<IreeStage>>
create_iree_stage(const std::shared_ptr<Run> &, const Dispatch &,
                  std::span<const PlanValue>) noexcept;

} // namespace hyle::compute::vulkan::detail
