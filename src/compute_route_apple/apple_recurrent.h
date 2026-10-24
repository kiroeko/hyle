#pragma once

#include "hyle/compute/apple.h"
#include "../compute_provider_ggml_metal/native_program.h"

namespace hyle::compute::apple::detail {
// Each Invocation owns its original ggml backend/graph and imported wrappers.
// MLX owns every payload, including the packed scores/state output allocation.
struct Recurrent final {
  std::unique_ptr<ggml_metal::NativeProgram> program;
  std::vector<TensorDesc> input_descriptions;
  std::vector<mlx::core::Shape> output_shapes;
  std::size_t proven_index_port = static_cast<std::size_t>(-1);
  std::int64_t proven_index_extent = 0;

  static Result<std::unique_ptr<Recurrent>>
  create(Operation operation, std::span<const TensorDesc> inputs,
         std::size_t proven_index_port = static_cast<std::size_t>(-1),
         std::int64_t proven_index_extent = 0) noexcept;
  void detach_inputs() noexcept;
  Result<std::vector<mlx::core::array>>
  run(std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
};
} // namespace hyle::compute::apple::detail
