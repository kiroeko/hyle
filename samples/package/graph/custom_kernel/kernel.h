#pragma once

#include <cstddef>
#include <span>
#include <string_view>

#include <hyle/compute/custom_kernel.h>

#if defined(_WIN32)
#if defined(HYLE_CUSTOM_EXAMPLE_BUILD)
#define HYLE_CUSTOM_EXAMPLE_API __declspec(dllexport)
#else
#define HYLE_CUSTOM_EXAMPLE_API __declspec(dllimport)
#endif
#else
#define HYLE_CUSTOM_EXAMPLE_API __attribute__((visibility("default")))
#endif

namespace example {

// This ID identifies the entire schema, including its descriptor layout and
// exact numerical semantics: f32 fma(input, scale, bias), with one rounding.
// The executable constructs it; the library decodes it.
struct Affine final {
  static constexpr hyle::compute::OperationId operation_id{0x4558414d504c45ULL,
                                                           1};
  static constexpr std::string_view operation_name = "example-affine";
  float bias = 0;
  float scale = 2;

  hyle::compute::OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1},
            .outputs = 1,
            .alias = hyle::compute::AliasPolicy::kMayAliasInput0};
  }
  hyle::Status infer_output_descriptions(
      std::span<const hyle::compute::TensorDesc> inputs,
      std::span<hyle::compute::TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1)
      return hyle::Status::InvalidArgument(
          "affine expects one input and output");
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

HYLE_CUSTOM_EXAMPLE_API hyle::Result<hyle::compute::Provider>
make_provider(std::size_t chunk_elements = 64) noexcept;

} // namespace example
