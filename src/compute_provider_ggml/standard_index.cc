#include "standard_index.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <new>
#include <numeric>
#include <type_traits>
#include <utility>

#include "ggml.h"

namespace hyle::compute::ggml_cpu::standard_index {
namespace {

struct Entry final {
  double key;
  std::int32_t index;
};
struct Program final {
  Options options;
  Operation operation;
  TensorDesc input;
  std::size_t axis = 0;
  std::size_t width = 0;
  std::size_t rows = 0;
  std::size_t inner = 1;
  std::size_t item_bytes = 0;
  std::size_t scratch_bytes = 0;
};
struct Invocation final {
  mem::AllocatorRef allocator;
  void *scratch = nullptr;
  std::size_t bytes = 0;
  mem::HostMapping input;
  std::array<mem::HostMapping, 2> outputs;

  ~Invocation() {
    if (scratch)
      allocator.deallocate(scratch, bytes, alignof(Entry));
  }
};

Status allocation_failure() noexcept {
  return Status::ResourceExhausted(
      "CPU standard-index storage allocation failed");
}
std::size_t selected_axis(std::int32_t axis, std::size_t rank) noexcept {
  return static_cast<std::size_t>(
      axis < 0 ? axis + static_cast<std::int32_t>(rank) : axis);
}
template <class T> T load(const std::byte *address) noexcept {
  T value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}
double key(const std::byte *address, DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return load<float>(address);
  case DataType::kInt32:
    return load<std::int32_t>(address);
  case DataType::kFloat16:
    return ggml_fp16_to_fp32(load<ggml_fp16_t>(address));
  case DataType::kBFloat16:
    return ggml_bf16_to_fp32(load<ggml_bf16_t>(address));
  default:
    return 0; // Qualified before program creation.
  }
}
// The row ordinal enumerates every axis except the selected scan/sort axis.
// Only byte addressing belongs to this adapter; numerical work remains in
// the original standard algorithms and ggml's public scalar conversion.
std::size_t input_row(const Program &program, std::size_t row) noexcept {
  std::size_t offset = 0;
  for (std::size_t dimension = program.input.rank(); dimension-- > 0;) {
    if (dimension == program.axis)
      continue;
    const auto extent =
        static_cast<std::size_t>(program.input.dimensions()[dimension]);
    offset += (row % extent) *
              static_cast<std::size_t>(program.input.byte_strides()[dimension]);
    row /= extent;
  }
  return offset;
}
std::size_t output_row(const Program &program, std::size_t row,
                       std::size_t width) noexcept {
  return (row / program.inner) * width * program.inner + row % program.inner;
}

void preserve(Status &first, Status next) noexcept {
  if (first.ok() && !next.ok())
    first = std::move(next);
}
Status close(Invocation &invocation, Status status,
             std::span<const TensorView> outputs) noexcept {
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto &mapping = invocation.outputs[i];
    if (mapping && status.ok() &&
        !outputs[i].storage().resource().properties().has(
            mem::MemoryProperty::kHostDeviceCoherent))
      preserve(status, mapping.flush());
    preserve(status, mapping.close());
  }
  preserve(status, invocation.input.close());
  return status;
}

template <class Storage, class Accumulator>
Status scan(const Program &program, Invocation &invocation) noexcept {
  const auto *source = invocation.input.bytes().data();
  auto *wide = static_cast<Accumulator *>(invocation.scratch);
  const auto stride =
      static_cast<std::size_t>(program.input.byte_strides()[program.axis]);
  for (std::size_t row = 0; row < program.rows; ++row) {
    const auto *input = source + input_row(program, row);
    auto *begin = wide + row * program.width;
    for (std::size_t column = 0; column < program.width; ++column)
      begin[column] = load<Storage>(input + column * stride);
    std::inclusive_scan(begin, begin + program.width, begin,
                        std::plus<Accumulator>{});
  }
  if constexpr (std::is_same_v<Storage, std::int32_t>) {
    // Even exclusive scan checks the final inclusive total. No output is
    // mapped until every row passes; invalid prefixes cannot publish a prefix.
    if (std::any_of(wide, wide + program.rows * program.width, [](auto value) {
          return value < INT32_MIN || value > INT32_MAX;
        }))
      return Status::OutOfRange("CPU standard Scan prefix exceeds int32");
  }
  return Status::Ok();
}
template <class Storage, class Accumulator>
void publish_scan(const Program &program, Invocation &invocation,
                  std::byte *output) noexcept {
  const auto *wide = static_cast<const Accumulator *>(invocation.scratch);
  const bool exclusive = program.operation.get_if<Scan>()->exclusive;
  for (std::size_t row = 0; row < program.rows; ++row) {
    auto *target =
        output + output_row(program, row, program.width) * sizeof(Storage);
    for (std::size_t column = 0; column < program.width; ++column) {
      const Storage value =
          exclusive && column == 0
              ? Storage{}
              : static_cast<Storage>(
                    wide[row * program.width + column - (exclusive ? 1 : 0)]);
      std::memcpy(target + column * program.inner * sizeof(Storage), &value,
                  sizeof(value));
    }
  }
}
void top_k(const Program &program, Invocation &invocation, std::byte *values,
           std::byte *indices) noexcept {
  const auto &operation = *program.operation.get_if<TopK>();
  auto *entries = static_cast<Entry *>(invocation.scratch);
  const auto *source = invocation.input.bytes().data();
  const auto stride =
      static_cast<std::size_t>(program.input.byte_strides()[program.axis]);
  const auto before = [&](const Entry &left, const Entry &right) noexcept {
    const bool left_nan = std::isnan(left.key),
               right_nan = std::isnan(right.key);
    if (left_nan != right_nan)
      return !left_nan;
    if (!left_nan && left.key != right.key)
      return operation.largest ? left.key > right.key : left.key < right.key;
    return left.index < right.index;
  };
  for (std::size_t row = 0; row < program.rows; ++row) {
    const auto *input = source + input_row(program, row);
    for (std::size_t column = 0; column < program.width; ++column)
      entries[column] = {
          key(input + column * stride, program.input.data_type()),
          static_cast<std::int32_t>(column)};
    std::partial_sort(entries, entries + operation.count,
                      entries + program.width, before);
    const auto offset = output_row(program, row, operation.count);
    for (std::size_t column = 0; column < operation.count; ++column) {
      const auto target = offset + column * program.inner;
      const auto index = entries[column].index;
      // Preserve source storage bits (including NaN payloads and signed zero).
      std::memcpy(values + target * program.item_bytes,
                  input + static_cast<std::size_t>(index) * stride,
                  program.item_bytes);
      std::memcpy(indices + target * sizeof(index), &index, sizeof(index));
    }
  }
}
Status run(const Program &program, Invocation &invocation,
           ConstTensorView input,
           std::span<const TensorView> outputs) noexcept {
  if (program.rows == 0 || program.width == 0)
    return Status::Ok();
  const auto finish = [&](Status status) {
    return close(invocation, std::move(status), outputs);
  };
  if (!input.storage().resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    auto status = input.storage().invalidate();
    if (!status.ok())
      return status;
  }
  auto mapping = input.storage().map_host();
  if (!mapping)
    return std::move(mapping).error();
  invocation.input = std::move(*mapping);
  const bool integer = program.input.data_type() == DataType::kInt32;
  const bool scanning = program.operation.get_if<Scan>() != nullptr;
  if (scanning) {
    auto status = integer
                      ? scan<std::int32_t, std::int64_t>(program, invocation)
                      : scan<float, float>(program, invocation);
    if (!status.ok())
      return finish(std::move(status));
  }
  std::array<std::byte *, 2> targets{};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto output_mapping =
        outputs[i].storage().map_host(mem::HostAccess::kWrite);
    if (!output_mapping)
      return finish(std::move(output_mapping).error());
    invocation.outputs[i] = std::move(*output_mapping);
    auto bytes = invocation.outputs[i].writable_bytes();
    if (!bytes)
      return finish(std::move(bytes).error());
    targets[i] = bytes->data();
  }
  if (scanning) {
    if (integer)
      publish_scan<std::int32_t, std::int64_t>(program, invocation, targets[0]);
    else
      publish_scan<float, float>(program, invocation, targets[0]);
  } else {
    top_k(program, invocation, targets[0], targets[1]);
  }
  return finish(Status::Ok());
}
void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  if (value) {
    static_cast<Invocation *>(value)->~Invocation();
    allocator.deallocate(value, sizeof(Invocation), alignof(Invocation));
  }
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView, void **output) noexcept {
  *output = nullptr;
  const auto &program = *static_cast<const Program *>(value);
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return allocation_failure();
  auto *invocation = ::new (storage) Invocation;
  invocation->allocator = allocator;
  invocation->bytes = program.scratch_bytes;
  *output = invocation;
  if (program.scratch_bytes) {
    invocation->scratch =
        allocator.allocate(program.scratch_bytes, alignof(Entry));
    if (!invocation->scratch)
      return allocation_failure();
  }
  return Status::Ok();
}
void destroy_program(void *value, mem::AllocatorRef allocator) noexcept {
  if (value) {
    static_cast<Program *>(value)->~Program();
    allocator.deallocate(value, sizeof(Program), alignof(Program));
  }
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
detail::EnqueueResult submit(void *value, void *slot, Queue &,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<const Program *>(value);
  if (inputs.size() != 1 ||
      outputs.size() != (program.operation.get_if<Scan>() ? 1U : 2U))
    return detail::EnqueueResult::rejected(
        Status::InvalidArgument("CPU standard-index bindings differ"));
  struct Call final {
    const Program &program;
    Invocation &invocation;
    ConstTensorView input;
    std::span<const TensorView> outputs;
    Status status = Status::Ok();
  } call{program, *static_cast<Invocation *>(slot), inputs[0], outputs};
  program.options.executor.parallel_for(
      1,
      [](void *value, std::size_t, std::size_t) noexcept {
        auto &work = *static_cast<Call *>(value);
        work.status =
            run(work.program, work.invocation, work.input, work.outputs);
      },
      &call);
  return detail::EnqueueResult::accepted_command(std::move(call.status));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

bool supports(Operation operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  const auto *scan = operation.get_if<Scan>();
  const auto *topk = operation.get_if<TopK>();
  if ((!scan && !topk) || inputs.size() != 1 ||
      outputs.size() != (scan ? 1U : 2U) || !inputs[0].is_plain() ||
      !inputs[0].valid())
    return false;
  const auto count = inputs[0].element_count();
  if (!count || *count > INT32_MAX)
    return false;
  const auto type = inputs[0].data_type();
  if (type != DataType::kInt32 && type != DataType::kFloat32 &&
      !(topk && (type == DataType::kFloat16 || type == DataType::kBFloat16)))
    return false;
  std::array<TensorDesc, 2> inferred;
  return infer_output_descriptions(operation, inputs,
                                   std::span(inferred).first(outputs.size()))
             .ok() &&
         std::ranges::equal(std::span(inferred).first(outputs.size()), outputs);
}
ProviderRuntimeProperties runtime() noexcept {
  return {.mode = ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = ProviderWorkspaceOwner::kProvider,
          .replay_class = ProviderReplayClass::kPrepared,
          .blocks_host = true,
          .uses_private_threads = false,
          .supports_concurrent_invocations = true,
          .capture_compatible = false};
}
ProviderRouteCostFacts costs() noexcept {
  return {.workspace = WorkspaceRequirement{},
          .persistent_bytes = sizeof(Program),
          .copy_bytes = std::nullopt,
          .repack_bytes = 0,
          .synchronization_count = 0,
          .dispatch_count = 1,
          .compilation_storage = {.resident_bytes = 0, .peak_bytes = 0}};
}
ProviderPrepareOutcome prepare(Options options,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route != kRoute || !request.shape.exact() ||
      !supports(request.operation, request.inputs, request.outputs))
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("CPU standard-index domain is not admitted"));
  Program program{options, request.operation, request.inputs[0]};
  const auto *scan = request.operation.get_if<Scan>();
  program.axis =
      selected_axis(scan ? scan->axis : request.operation.get_if<TopK>()->axis,
                    program.input.rank());
  program.width =
      static_cast<std::size_t>(program.input.dimensions()[program.axis]);
  const auto count = program.input.element_count().value();
  program.rows = program.width ? count / program.width : 0;
  program.item_bytes = data_type_size(program.input.data_type());
  for (std::size_t i = program.axis + 1; i < program.input.rank(); ++i)
    program.inner *= static_cast<std::size_t>(program.input.dimensions()[i]);
  const auto units = scan ? count : (count == 0 ? 0 : program.width);
  const auto unit_bytes = scan ? (program.input.data_type() == DataType::kInt32
                                      ? sizeof(std::int64_t)
                                      : sizeof(float))
                               : sizeof(Entry);
  if (units > SIZE_MAX / unit_bytes)
    return ProviderPrepareOutcome::resource_declined(allocation_failure());
  program.scratch_bytes = units * unit_bytes;
  void *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage)
    return ProviderPrepareOutcome::resource_declined(allocation_failure());
  auto *prepared = ::new (storage) Program(program);
  *output = {.vtable = &kPrepared,
             .state = prepared,
             .workspace = {},
             .runtime = runtime(),
             .persistent_bytes = sizeof(Program),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::ggml_cpu::standard_index
