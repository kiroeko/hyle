#include "vulkan_device_lengths.h"
#include "vulkan_carrier.h"
#include "vulkan_program_internal.h"
#include "hyle/compute/vulkan_batch.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <mutex>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace shaders { std::span<const std::uint32_t> device_lengths_paged() noexcept; }
namespace {
constexpr std::uint32_t bindings = 11, origins = 9, push_bytes = origins * sizeof(std::uint32_t);
Result<std::vector<TensorDesc>> descriptions(const DeviceLengthsPagedRead &d) noexcept try {
  HYLE_RETURN_IF_ERROR(qualify_device_lengths_paged_schema(d.schema, d.scale, d.causal, d.window_left));
  const auto c = d.schema.capacity();
  if (!d.rows || d.rows > c.max_query_rows)
    return std::unexpected(Status::InvalidArgument("Vulkan device lengths need a nonempty bounded Q carrier"));
  HYLE_ASSIGN_OR_RETURN(auto q, batch_query_description(c, d.rows));
  HYLE_ASSIGN_OR_RETURN(auto k, batch_cache_description(d.schema, BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto v, batch_cache_description(d.schema, BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto lengths, TensorDesc::dense(DataType::kInt32,
      std::array<std::int64_t, 1>{c.max_requests}));
  // Private physical view of the original readonly metadata, not a newly
  // minted public layout identity. The Graph caller retains that typed owner.
  HYLE_ASSIGN_OR_RETURN(auto metadata, TensorDesc::dense(DataType::kInt32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(d.schema.metadata_offsets().bytes/4)}));
  std::vector<TensorDesc> result{q, k, v, lengths, lengths, metadata};
  if (d.guard_writes) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{1}));
    result.push_back(status); result.push_back(status);
  }
  return result;
} catch (const std::bad_alloc &) {
  return std::unexpected(Status::ResourceExhausted("Vulkan device length descriptions"));
}
std::uint32_t encoding(DataType type) noexcept { return type == DataType::kBFloat16 ? 2U : 1U; }
struct PagedLaunch final {
  TensorDesc partials;
  std::uint32_t splits = 0;
  std::uint32_t groups = 0;
};
// The public cold query and actual native lowering use the same physical
// qualification. This function reads immutable descriptors and device limits;
// it creates no VkPipeline, backing allocation or execution authority.
Result<PagedLaunch> qualify_launch(const VkPhysicalDeviceLimits &limits,
    const DeviceLengthsPagedRead &description,
    std::span<const TensorDesc> inputs, const TensorDesc &output) noexcept {
  if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
      limits.maxPerStageDescriptorStorageBuffers < bindings || limits.maxDescriptorSetStorageBuffers < bindings ||
      limits.maxPushConstantsSize < push_bytes || limits.maxComputeSharedMemorySize < 512)
    return std::unexpected(Status::Unimplemented("Vulkan device length descriptor/workgroup limits"));
  for (const auto &input : inputs)
    if (*input.element_count() > UINT32_MAX || *input.required_bytes() > limits.maxStorageBufferRange)
      return std::unexpected(Status::Unimplemented("Vulkan device length input indexing/buffer limits"));
  if (*output.element_count() > UINT32_MAX || *output.required_bytes() > limits.maxStorageBufferRange)
    return std::unexpected(Status::Unimplemented("Vulkan device length output indexing/buffer limits"));
  const auto c = description.schema.capacity();
  const auto splits = std::min(32U, std::max(1U, (c.max_kv_rows + 511U) / 512U));
  const auto groups = std::uint64_t(description.rows) * c.query_heads * splits;
  if (groups > limits.maxComputeWorkGroupCount[0])
    return std::unexpected(Status::Unimplemented("Vulkan device length dispatch limit"));
  HYLE_ASSIGN_OR_RETURN(auto partials, TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{static_cast<std::int64_t>(groups), c.value_head_dim + 2}));
  if (*partials.element_count() > UINT32_MAX || *partials.required_bytes() > limits.maxStorageBufferRange)
    return std::unexpected(Status::Unimplemented("Vulkan device length partial buffer limit"));
  return PagedLaunch{std::move(partials), splits, static_cast<std::uint32_t>(groups)};
}
} // namespace

Status qualify_device_lengths_paged_schema(const BatchSchema &schema, float scale,
    bool causal, std::int32_t window_left) noexcept {
  if (!schema.valid() || !std::isfinite(scale))
    return Status::InvalidArgument("Vulkan device lengths need a schema and finite scale");
  const auto c = schema.capacity();
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal, window_left));
  const auto half = [](DataType t) { return t == DataType::kFloat16 || t == DataType::kBFloat16; };
  if (schema.mla_layout() != MlaCacheLayout::kNone || c.storage != BatchStorage::kPaged ||
      !half(c.query_type) || !half(c.output_type) ||
      !half(c.key_storage.data_type()) || !half(c.value_storage.data_type()) ||
      c.key_storage != CacheStorage::plain(c.key_storage.data_type()).value() ||
      c.value_storage != CacheStorage::plain(c.value_storage.data_type()).value() ||
      c.qk_head_dim%2 || c.value_head_dim%2 || c.value_head_dim>256)
    return Status::Unimplemented("Vulkan device lengths require whole halfword pairs and at most256 value channels");
  if (c.max_requests >= INT32_MAX || c.max_query_rows > INT32_MAX ||
      c.max_pages > INT32_MAX || c.max_page_refs > INT32_MAX ||
      c.max_kv_rows > INT32_MAX || c.page_size > INT32_MAX ||
      schema.metadata_offsets().bytes > UINT32_MAX)
    return Status::Unimplemented("Vulkan device length metadata indexing limit");
  return Status::Ok();
}

Status DeviceLengthsPagedRead::infer_output_descriptions(
    std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto expected, descriptions(*this));
  if (inputs.size() != expected.size() || outputs.size() != 1 ||
      !std::equal(inputs.begin(), inputs.end(), expected.begin()))
    return Status::InvalidArgument("Vulkan device length original port descriptions differ");
  HYLE_ASSIGN_OR_RETURN(outputs[0], batch_output_description(schema.capacity(), rows));
  return Status::Ok();
}

Result<Program> prepare_device_lengths_paged_graph(std::shared_ptr<Runtime> runtime,
    BatchSchema schema, std::uint32_t rows, float scale, bool causal,
    std::int32_t window_left, const WorkloadContext &workload, bool guard_writes) noexcept try {
  DeviceLengthsPagedRead d{std::move(schema), rows, scale, causal, window_left, guard_writes};
  HYLE_ASSIGN_OR_RETURN(auto inputs, descriptions(d));
  Plan plan; plan.workload = workload;
  for (const auto &input : inputs) {
    plan.inputs.push_back(static_cast<std::uint32_t>(plan.values.size()));
    plan.values.push_back({input, {}});
  }
  TensorDesc output;
  HYLE_RETURN_IF_ERROR(d.infer_output_descriptions(inputs, std::span(&output, 1)));
  const auto id = static_cast<std::uint32_t>(plan.values.size());
  plan.values.push_back({output, {}}); plan.outputs = {id};
  plan.steps.push_back({Operation::custom(d), plan.inputs, {id}});
  return prepare_plan(runtime, plan, PlanQueuePolicy::kRuntimeQueueOnly);
} catch (const std::bad_alloc &) {
  return std::unexpected(Status::ResourceExhausted("Vulkan device length cold recipe"));
}

Status lower_device_lengths_paged(ProgramState &program, const PlanStep &step) noexcept try {
  const auto d = step.operation.custom_as<DeviceLengthsPagedRead>();
  if (!d || step.inputs.size() != (d->guard_writes ? 8U : 6U) || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan device length lowering ports differ");
  std::vector<TensorDesc> inputs;
  for (auto id : step.inputs) inputs.push_back(program.values[id].description);
  TensorDesc output;
  HYLE_RETURN_IF_ERROR(d->infer_output_descriptions(inputs, std::span(&output, 1)));
  if (output != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan device length output differs");
  HYLE_ASSIGN_OR_RETURN(auto launch, qualify_launch(program.runtime->device->properties.limits,
      *d, inputs, output));
  const auto c = d->schema.capacity();
  const auto splits = launch.splits, groups = launch.groups;
  const auto partial = static_cast<std::uint32_t>(program.values.size());
  program.values.push_back({std::move(launch.partials), {}}); program.aliases.push_back({partial, 0});
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto status, TensorDesc::dense(DataType::kInt32, {}));
    const auto id = static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({status, {}}); program.aliases.push_back({id, 0});
    program.status_value = id;
  }
  const auto checked = *program.status_value;
  const std::array<std::uint32_t, bindings> values{step.inputs[0], step.inputs[1], checked,
      step.inputs[2], step.inputs[3], step.inputs[4], step.inputs[5], partial, step.outputs[0],
      d->guard_writes ? step.inputs[6] : checked, d->guard_writes ? step.inputs[7] : checked};
  const auto m = d->schema.metadata_offsets();
  for (std::uint32_t phase = 0; phase < 3; ++phase) {
    const std::array<SpecializationConstant, 24> constants{{
      {0,phase}, {1,d->rows}, {2,c.query_heads}, {3,c.kv_heads}, {4,c.qk_head_dim},
      {5,c.value_head_dim}, {6,c.max_requests}, {7,c.max_pages}, {8,c.max_page_refs},
      {9,c.page_size}, {10,c.max_kv_rows}, {11,encoding(c.query_type)},
      {12,encoding(c.key_storage.data_type())}, {13,encoding(c.value_storage.data_type())},
      {14,encoding(c.output_type)}, {15,splits}, {16,d->causal ? 1U : 0U},
      {17,std::bit_cast<std::uint32_t>(d->window_left)}, {18,std::bit_cast<std::uint32_t>(d->scale)},
      {19,static_cast<std::uint32_t>(m.query_indptr/4)}, {20,static_cast<std::uint32_t>(m.kv_indptr/4)},
      {21,static_cast<std::uint32_t>(m.page_indices/4)}, {22,static_cast<std::uint32_t>(m.last_page_lengths/4)},
      {23,d->guard_writes ? 1U : 0U}}};
    Dispatch dispatch;
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {shaders::device_lengths_paged(), bindings, push_bytes, constants}));
    dispatch.values.assign(values.begin(), values.end());
    dispatch.push.resize(push_bytes); dispatch.tensor_byte_origins = true;
    // Scalar writer-status bindings retain ordinary native descriptor offsets.
    // Only the nine data/status/partial/output ports need byte-origin rebasing.
    dispatch.tensor_byte_origin_count = origins; dispatch.checked_status = true;
    dispatch.groups = {phase == 0 ? 1U : phase == 1 ? static_cast<std::uint32_t>(groups)
        : d->rows*c.query_heads, 1, 1};
    program.dispatches.push_back(std::move(dispatch));
  }
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan device length pipeline metadata");
}
} // namespace hyle::compute::vulkan::detail

namespace hyle::compute::vulkan {
Result<bool> supports_device_lengths_paged(const Context &context, BatchCapacity capacity,
    const ResourceBudget &budget) noexcept {
  return supports_device_lengths_paged(context, capacity, capacity.max_query_rows, budget);
}

Result<bool> supports_device_lengths_paged(const Context &context, BatchCapacity capacity,
    std::uint32_t query_rows, const ResourceBudget &budget) noexcept {
  if (!context.valid())
    return std::unexpected(Status::FailedPrecondition("Vulkan device lengths query requires its original Context"));
  const auto &runtime = detail::Access::runtime(context);
  {
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed)
      return std::unexpected(Status::FailedPrecondition("Vulkan device lengths Context is closed"));
    HYLE_RETURN_IF_ERROR(runtime->failure);
  }
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(capacity));
  if (!query_rows || query_rows > capacity.max_query_rows)
    return std::unexpected(Status::InvalidArgument("Vulkan device lengths require query rows within the capacity"));
  HYLE_ASSIGN_OR_RETURN(auto schema, BatchSchema::create(capacity, {.resource_budget = budget}));
  const auto domain = detail::qualify_device_lengths_paged_schema(schema, 1.F, true, -1);
  if (domain.IsUnimplemented()) return false;
  HYLE_RETURN_IF_ERROR(domain);
  const detail::DeviceLengthsPagedRead description{std::move(schema), query_rows, 1.F, true, -1, true};
  HYLE_ASSIGN_OR_RETURN(auto inputs, detail::descriptions(description));
  HYLE_ASSIGN_OR_RETURN(auto output, batch_output_description(capacity, query_rows));
  auto launch = detail::qualify_launch(runtime->device->properties.limits, description, inputs, output);
  if (!launch && launch.error().IsUnimplemented()) return false;
  if (!launch) return std::unexpected(launch.error());
  // The native implementation still reports driver/pipeline-private storage
  // as unattributed; a strict budget cannot be advertised as qualified.
  return !budget.valid() || budget.allows_unattributed();
}
} // namespace hyle::compute::vulkan
