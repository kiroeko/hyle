#include "apple_recurrent.h"

#include <array>
#include <new>
#include <optional>
#include <utility>

#include <mlx/allocator.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "apple_internal.h"

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
                       std::size_t offset) {
  mx::array result(shape, mx::float32, nullptr, {});
  result.copy_shared_buffer(packed, result.strides(), result.flags(),
                            result.size(), static_cast<std::int64_t>(offset));
  result.set_status(mx::array::available);
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

Result<std::vector<mx::array>>
Recurrent::run(std::span<const mx::array> inputs, mx::Stream stream) {
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
  HYLE_RETURN_IF_ERROR(proven_index_extent > 0
      ? program->run_with_proven_indices(active_ports, proven_index_port, proven_index_extent)
      : program->run(active_ports));
  packed.set_status(mx::array::available);
  std::vector<mx::array> outputs;
  outputs.reserve(output_shapes.size());
  for (std::size_t port = 0; port < output_shapes.size(); ++port)
    outputs.push_back(output_view(packed, output_shapes[port],
                                  program->output_offset(port) /
                                      sizeof(float)));
  return outputs;
}
} // namespace hyle::compute::apple::detail
