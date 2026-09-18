#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/vulkan_plan.h"
#include "vulkan_internal.h"

namespace hyle::compute::vulkan::detail {

struct Pipeline final {
  std::shared_ptr<Device> device;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline handle = VK_NULL_HANDLE;
  std::uint32_t bindings = 0;
  std::uint32_t push_bytes = 0;
  ~Pipeline();
};

struct Dispatch final {
  std::shared_ptr<Pipeline> pipeline;
  std::shared_ptr<const ggml_vulkan::detail::RegionRecipe> region;
  std::vector<std::uint32_t> values;
  std::vector<std::byte> push;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  bool tensor_byte_origins = false;
};

struct NativeEncoding final {
  std::vector<Dispatch> dispatches;
};

struct ValueAlias final {
  std::uint32_t root = 0;
  VkDeviceSize byte_offset = 0;
};

struct ProgramState final {
  std::shared_ptr<Runtime> runtime;
  std::vector<PlanValue> values;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<TensorDesc> input_descriptions;
  std::vector<TensorDesc> output_descriptions;
  std::vector<Dispatch> dispatches;
  std::vector<ValueAlias> aliases;
  // UINT32_MAX for inputs/constants. Equal indices share storage only when
  // cold liveness proves non-overlap; all published outputs live to program
  // end.
  std::vector<std::uint32_t> storage_blocks;
  std::vector<VkDeviceSize> block_bytes;
};

struct ResultValues final {
  std::vector<Tensor> tensors;
};

struct DescriptorBinding final {
  std::uint32_t value = 0;
  std::uint32_t input_port = 0;
  bool byte_origin = false;
  VkDescriptorBufferInfo buffer{};
  VkWriteDescriptorSet write{};
};

struct InvocationState final {
  struct RegionInput final {
    std::size_t stage = 0;
    std::size_t port = 0;
    std::uint32_t value = 0;
    std::uint32_t input_port = 0;
  };
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<Run> run;
  std::shared_ptr<ResultValues> results;
  std::vector<DescriptorBinding> dynamic_bindings;
  std::vector<RegionInput> region_inputs;
  std::vector<std::shared_ptr<Buffer>> old_inputs;
  std::vector<long> output_owner_counts;
  cd::ResultSlot<Results> slot;
  bool bound = false;
};

struct RecorderState final {
  std::shared_ptr<Runtime> runtime;
  std::vector<PlanValue> values;
  std::vector<Dispatch> dispatches;
  bool finished = false;
};

Result<std::shared_ptr<Pipeline>>
create_pipeline(const std::shared_ptr<Device> &device,
                const SpirvDescription &description) noexcept;
Status
record(const std::shared_ptr<Run> &run, std::span<const PlanValue> values,
       std::span<const Dispatch> dispatches,
       std::vector<DescriptorBinding> *dynamic_bindings = nullptr) noexcept;
Status plan_storage(ProgramState &program) noexcept;
Status allocate_values(const ProgramState &program, Run &run) noexcept;
Result<VkDescriptorBufferInfo>
descriptor_buffer(const Buffer *buffer, const TensorDesc &description,
                  bool byte_origin, VkDeviceSize relative_offset = 0) noexcept;
Status encode_commands(Run &run, std::span<const Dispatch> dispatches) noexcept;
bool origins_changed(const Run &run,
                     std::span<const Dispatch> dispatches) noexcept;
Status lower_operation(
    ProgramState &program, const PlanStep &step,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> &packs) noexcept;
bool managed_operation(const ProgramState &, const PlanStep &) noexcept;
Status lower_topk(ProgramState &, const PlanStep &) noexcept;
Status lower_region(ProgramState &, const Plan &, std::size_t first,
                    std::size_t end,
                    std::span<const std::size_t> last_uses) noexcept;
Status record_regions(const std::shared_ptr<Run> &, std::span<const PlanValue>,
                      std::span<const Dispatch>,
                      std::vector<DescriptorBinding> *) noexcept;
Status bind_regions(Run &) noexcept;
void refresh_region_bindings(Run &) noexcept;

} // namespace hyle::compute::vulkan::detail
