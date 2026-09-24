#include "apple_recurrent.h"

#include <array>
#include <new>

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
Recurrent::create(Operation operation, std::span<const TensorDesc> inputs) noexcept {
  try {
    HYLE_ASSIGN_OR_RETURN(
        auto native, ggml_metal::NativeProgram::create(operation, inputs));
    auto result = std::make_unique<Recurrent>();
    result->program = std::move(native);
    result->output_shapes.resize(result->program->output_count());
    for (std::size_t port = 0; port < result->program->output_count(); ++port)
      for (auto d : result->program->output_description(port).dimensions())
        result->output_shapes[port].push_back(static_cast<int>(d));
    return result;
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<std::vector<mx::array>>
Recurrent::run(std::span<const mx::array> inputs, mx::Stream stream) {
  std::vector<mx::array> dense(inputs.begin(), inputs.end());
  bool copied = false;
  for (std::size_t i = 0; i < dense.size(); ++i) {
    auto &input = dense[i];
    if (!input.flags().row_contiguous) {
      if (program->input_is_packed(i))
        return std::unexpected(Status::InvalidArgument(
            "Apple packed GGUF input must be row-contiguous"));
      input = mx::contiguous(input, false, stream);
      copied = true;
    }
  }
  if (copied) {
    mx::async_eval(dense);
    mx::synchronize(stream);
    for (auto &input : dense)
      input.wait();
  }
  auto packed = packed_array(program->output_bytes());
  std::vector<ggml_metal::BufferView> ports(dense.size() + 1);
  for (std::size_t i = 0; i < dense.size(); ++i)
    ports[i] = view(dense[i]);
  ports.back() = view(packed);
  HYLE_RETURN_IF_ERROR(program->run(ports));
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
