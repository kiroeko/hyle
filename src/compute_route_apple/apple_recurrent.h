#pragma once

#include "hyle/compute/apple.h"
#include "../compute_provider_ggml_metal/native_program.h"

namespace hyle::compute::apple::detail {
// Each Invocation owns its original ggml backend/graph and imported wrappers.
// MLX owns every payload, including the packed scores/state output allocation.
struct Recurrent final {
  std::unique_ptr<ggml_metal::NativeProgram> program;
  mlx::core::Shape scores_shape;
  mlx::core::Shape state_shape;

  static Result<std::unique_ptr<Recurrent>>
  create(std::span<const TensorDesc> inputs) noexcept;
  Result<std::vector<mlx::core::array>>
  run(std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
};
} // namespace hyle::compute::apple::detail
