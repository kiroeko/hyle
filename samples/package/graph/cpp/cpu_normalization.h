#pragma once

#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/providers/native.h>
#include <hyle/graph/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/graph/providers/onednn.h>
#include <hyle/mem/buffer.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace sample_cpu_normalization {
namespace c = hyle::compute;
namespace g = hyle::graph;

inline hyle::Status run() {
  constexpr std::size_t rows = 2, width = 7;
  HYLE_ASSIGN_OR_RETURN(
      auto data,
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 2>{rows, width}));
  HYLE_ASSIGN_OR_RETURN(
      auto parameter, c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 1>{width}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(data));
  HYLE_ASSIGN_OR_RETURN(auto scale, builder.input(parameter));
  HYLE_ASSIGN_OR_RETURN(auto normalized, builder.add(c::RmsNorm{}, input));
  HYLE_ASSIGN_OR_RETURN(
      auto weighted,
      builder.add(c::Binary{c::BinaryKind::kMultiply}, normalized, scale));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Binary{c::BinaryKind::kAdd},
                                                 weighted, input));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());

  std::array<float, rows * width> values{}, actual{};
  std::array<float, width> scales{};
  for (std::size_t i = 0; i < width; ++i)
    scales[i] = 0.5F + static_cast<float>(i) / 16.F;
  HYLE_ASSIGN_OR_RETURN(auto source,
                        hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                  sizeof(values), 64));
  HYLE_ASSIGN_OR_RETURN(auto weights,
                        hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                  sizeof(scales), 64));
  HYLE_ASSIGN_OR_RETURN(auto result,
                        hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                  sizeof(actual), 64));
  HYLE_RETURN_IF_ERROR(weights.view().write(std::as_bytes(std::span(scales))));
  HYLE_ASSIGN_OR_RETURN(auto input_view,
                        c::ConstTensorView::create(data, source.view()));
  HYLE_ASSIGN_OR_RETURN(auto scale_view,
                        c::ConstTensorView::create(parameter, weights.view()));
  HYLE_ASSIGN_OR_RETURN(auto output_view,
                        c::TensorView::create(data, result.view()));

  // Ordinary route first. The second pass explicitly qualifies the installed
  // mature region even if a future default rule selects another valid route.
  for (bool qualify_region : {false, true}) {
    c::Context context;
    if (qualify_region) {
      const std::array providers{
          hyle::compute::ProviderBinding{g::onednn::provider(), 100},
          hyle::compute::ProviderBinding{c::native::provider(), 200}};
      HYLE_ASSIGN_OR_RETURN(
          context,
          c::Context::create(c::cpu::device(), {.providers = providers}));
    } else {
      HYLE_ASSIGN_OR_RETURN(context, g::cpu::create_context());
    }
    HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(logical, context));
    if (qualify_region) {
      HYLE_ASSIGN_OR_RETURN(auto prefix, compiled.report.step(0));
      if (compiled.report.region_count() != 1 ||
          compiled.report.step_count() != 2 || prefix.node_count != 2 ||
          prefix.provider.id != g::onednn::kProviderId ||
          !compiled.report.mixed_providers())
        return hyle::Status::Internal(
            "installed normalization prefix is absent");
    }
    HYLE_ASSIGN_OR_RETURN(auto invocation,
                          compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(invocation.bind(input, input_view));
    HYLE_RETURN_IF_ERROR(invocation.bind(scale, scale_view));
    HYLE_RETURN_IF_ERROR(invocation.bind(output, output_view));
    const std::array queues{
        g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i) - 7 + repeat) / 8.F;
      HYLE_RETURN_IF_ERROR(
          source.view().write(std::as_bytes(std::span(values))));
      HYLE_ASSIGN_OR_RETURN(auto completion, invocation.submit(queue));
      HYLE_RETURN_IF_ERROR(completion.wait());
      HYLE_RETURN_IF_ERROR(
          result.view().read(std::as_writable_bytes(std::span(actual))));
      for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0;
        for (std::size_t col = 0; col < width; ++col) {
          const double value = values[row * width + col];
          squares += value * value;
        }
        const double inverse =
            1.0 / std::sqrt(squares / width + c::RmsNorm{}.epsilon);
        for (std::size_t col = 0; col < width; ++col) {
          const std::size_t i = row * width + col;
          const double expected = values[i] * inverse * scales[col] + values[i];
          if (!std::isfinite(actual[i]) ||
              std::abs(actual[i] - expected) > 3.e-5)
            return hyle::Status::Internal(
                "installed normalization result disagrees with oracle");
        }
      }
    }
  }
  return hyle::Status::Ok();
}
} // namespace sample_cpu_normalization
