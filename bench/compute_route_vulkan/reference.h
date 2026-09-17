#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "hyle/compute/vulkan.h"

namespace hyle::bench::vulkan {
namespace v = compute::vulkan;

struct NativeStep {
  v::Kernel kernel;
  std::vector<std::uint32_t> bindings;
  std::vector<std::byte> push;
  std::array<std::uint32_t, 3> groups;
};

// Independent cold recipe from the pinned ggml shader interfaces. No call to
// HYLE typed/Graph lowering or access to the candidate's generated SPIR-V.
class Reference {
public:
  Reference(const v::Context &, int rows, int inner, int hidden, int output,
            const std::array<std::vector<float>, 3> &weights);
  v::Program owning_program(const v::Context &) const;
  std::array<v::Tensor, 8> values(const v::Context &,
                                  const v::Tensor &input) const;
  v::RecordedProgram record(const v::Context &,
                            const std::array<v::Tensor, 8> &) const;

private:
  std::array<compute::TensorDesc, 8> descriptions_;
  std::array<v::Tensor, 3> weights_;
  std::vector<NativeStep> steps_;
};

} // namespace hyle::bench::vulkan
