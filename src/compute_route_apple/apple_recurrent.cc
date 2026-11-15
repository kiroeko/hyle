#include "apple_recurrent.h"

#include <array>
#include <new>
#include <limits>
#include <optional>
#include <utility>

#include <mlx/allocator.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "apple_internal.h"
#include "apple_native_event.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
struct Allocation final {
  mx::allocator::Buffer buffer;
  ~Allocation() {
    if (buffer.ptr() != nullptr)
      mx::allocator::free(buffer);
  }
};

mx::array packed_array(std::size_t bytes) {
  mx::array result(mx::Shape{static_cast<int>(bytes / sizeof(float))},
                   mx::float32, nullptr, {});
  Allocation owner{mx::allocator::malloc(bytes)};
  if (owner.buffer.ptr() == nullptr)
    throw std::bad_alloc();
  result.set_data(owner.buffer);
  owner.buffer = mx::allocator::Buffer(nullptr);
  return result;
}

mx::array output_view(const mx::array &packed, const mx::Shape &shape,
                       std::size_t offset, const mx::Event *event = nullptr) {
  mx::array result(shape, mx::float32, nullptr, {});
  result.copy_shared_buffer(packed, result.strides(), result.flags(),
                            result.size(), static_cast<std::int64_t>(offset));
  if (event) {
    result.attach_event(*event);
    result.set_status(mx::array::evaluated);
  } else result.set_status(mx::array::available);
  return result;
}

ggml_metal::BufferView view(const mx::array &value) {
  // offset() is metadata. Do not call data(), which can materialize private
  // Metal storage on the host. The native import checks device/storage/extent.
  return {const_cast<void *>(value.buffer().ptr()),
          static_cast<std::size_t>(value.offset())};
}
} // namespace

Result<std::unique_ptr<Recurrent>>
Recurrent::create(Operation operation, std::span<const TensorDesc> inputs,
                  std::size_t proven_index_port, std::int64_t proven_index_extent) noexcept {
  try {
    HYLE_ASSIGN_OR_RETURN(
        auto native, ggml_metal::NativeProgram::create(operation, inputs));
    auto result = std::make_unique<Recurrent>();
    result->program = std::move(native);
    result->input_descriptions.assign(inputs.begin(), inputs.end());
    result->proven_index_port = proven_index_port;
    result->proven_index_extent = proven_index_extent;
    result->host_indices = proven_index_extent == 0 && result->program->reads_indices();
    result->output_shapes.resize(result->program->output_count());
    for (std::size_t port = 0; port < result->program->output_count(); ++port)
      for (auto d : result->program->output_description(port).dimensions())
        result->output_shapes[port].push_back(static_cast<int>(d));
    return result;
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

void Recurrent::detach_inputs() noexcept { program->detach_bindings(); }

Result<bool> Recurrent::prepare_signal(mx::Stream stream) noexcept {
  if (!program->can_signal()) return false;
  try {
    completion_event = mx::Event(stream);
    pending_inputs.reserve(input_descriptions.size());
    return true;
  } catch (...) { return std::unexpected(native_exception()); }
}
Status Recurrent::finish() noexcept {
  const auto status = program->finish();
  pending_inputs.clear(); // NativeProgram checked and drained the original prefix.
  return status;
}

Result<std::vector<mx::array>>
Recurrent::run(std::span<const mx::array> inputs, mx::Stream stream) {
  return run_impl(inputs, stream, {});
}
Result<std::vector<mx::array>>
Recurrent::submit(std::span<const mx::array> inputs, mx::Stream stream,
                  void (*completed)(void *) noexcept, void *state) {
  if (!completion_event.valid() || program->pending())
    return std::unexpected(Status::FailedPrecondition("native Metal event is unprepared or still pending"));
  if (signal_value == (std::numeric_limits<std::uint64_t>::max)())
    return std::unexpected(Status::ResourceExhausted("native Metal event generation exhausted"));
  completion_event.set_value(++signal_value);
  return run_impl(inputs, stream,
      {completion_event.cast<mx::metal::EventImpl>().mtl_event(), signal_value, completed, state});
}
Result<std::vector<mx::array>>
Recurrent::run_impl(std::span<const mx::array> inputs, mx::Stream stream,
                    ggml_metal::NativeSignal signal) {
  if (inputs.size() > 8)
    return std::unexpected(Status::InvalidArgument(
        "Apple native region exceeds eight input buffers"));
  if (inputs.size() != input_descriptions.size())
    return std::unexpected(Status::InvalidArgument(
        "Apple native region input count differs from preparation"));
  // NativeProgram admits at most eight inputs. Keep only owners of actual
  // CONT copies; already-dense inputs remain owned by the caller's span.
  std::array<std::optional<mx::array>, 8> dense;
  std::size_t copied = 0;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    const auto &description = input_descriptions[i];
    if (description.is_plain() && !description.is_dense()) {
      // A non-dense native port was qualified at cold preparation. It must
      // retain its actual carrier layout: a CONT would expand shared slots
      // while the native program still addresses their original A1 storage.
      bool matches = description.is_plain() &&
          description.data_type() == DataType::kFloat32 &&
          input.dtype() == mx::float32 &&
          static_cast<std::size_t>(input.ndim()) == description.rank();
      for (std::size_t axis = 0; matches && axis < description.rank(); ++axis) {
        matches = input.shape(axis) == description.dimensions()[axis];
        if (matches && description.dimensions()[axis] > 1) {
          const auto stride = description.byte_strides()[axis];
          matches = stride % sizeof(float) == 0 &&
              std::cmp_equal(input.strides()[axis], static_cast<std::size_t>(stride / sizeof(float)));
        }
      }
      if (!matches)
        return std::unexpected(Status::InvalidArgument(
            "Apple native strided input differs from its prepared layout"));
      continue;
    }
    if (!input.flags().row_contiguous) {
      if (program->input_is_packed(i))
        return std::unexpected(Status::InvalidArgument(
            "Apple packed GGUF input must be row-contiguous"));
      dense[i] = mx::contiguous(input, false, stream);
      ++copied;
    }
  }
  if (copied) {
    encode_native_input_waits(inputs, stream);
    std::vector<mx::array> pending;
    pending.reserve(copied);
    for (auto &input : dense)
      if (input) pending.push_back(*input);
    mx::async_eval(std::move(pending));
    mx::synchronize(stream);
    for (auto &input : dense)
      if (input) input->wait();
  }
  auto packed = packed_array(program->output_bytes());
  std::array<ggml_metal::BufferView, 9> ports;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ports[i] = view(dense[i] ? *dense[i] : inputs[i]);
  ports[inputs.size()] = view(packed);
  const auto active_ports = std::span(ports).first(inputs.size() + 1);
  // Prepare all throwing metadata/owners before the native prefix is accepted.
  std::vector<mx::array> outputs;
  outputs.reserve(output_shapes.size());
  for (std::size_t port = 0; port < output_shapes.size(); ++port)
    outputs.push_back(output_view(packed, output_shapes[port],
                                  program->output_offset(port) /
                                      sizeof(float), signal.valid() ? &completion_event : nullptr));
  if (signal.valid()) {
    pending_inputs.clear();
    for (std::size_t i = 0; i < inputs.size(); ++i)
      pending_inputs.push_back(dense[i] ? *dense[i] : inputs[i]);
    std::array<ggml_metal::NativeWait, 8> waits;
    std::size_t wait_count = 0;
    for (const auto &input : pending_inputs) {
      const auto &event = input.event();
      if (event.valid() && !event.is_signaled())
        waits[wait_count++] = {event.cast<mx::metal::EventImpl>().mtl_event(), event.value()};
    }
    auto status = program->submit(active_ports, signal, proven_index_port,
        proven_index_extent, std::span(waits).first(wait_count));
    if (!status.ok()) { pending_inputs.clear(); return std::unexpected(std::move(status)); }
  } else {
    HYLE_RETURN_IF_ERROR(proven_index_extent > 0
        ? program->run_with_proven_indices(active_ports, proven_index_port, proven_index_extent)
        : program->run(active_ports));
  }
  return outputs;
}
} // namespace hyle::compute::apple::detail
