#pragma once

#include <cstdlib>
#include <memory>

#include <mlx/transforms.h>

// Opt-in, finite cause capture. It never replaces a model oracle or its gates.
namespace apple_dspark_numeric_capture {
namespace base = cpu_c4_model_branches;
namespace ds = base::ds;
namespace mx = mlx::core;

inline std::size_t differences(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) throw std::runtime_error("capture geometry differs");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}
struct State {
  base::Model model;
  std::size_t step = 0;
  std::vector<float> attention, inverse, grouped, cosines, sines;
  explicit State(base::Model value) : model(value) {
    std::vector<float> angles;
    const auto frequency = ds::rope_frequencies(false, true);
    for (int position = 129; position <= 135; ++position)
      for (float f : frequency) angles.push_back(float(position) * f);
    auto context = hyle::compute::apple::Context::create().value();
    const std::array specs{hyle::compute::apple::InputSpec{{224}, mx::float32}};
    auto program = context.compile([](const auto &input, mx::Stream stream) {
      return std::vector{mx::cos(input[0], stream), mx::sin(input[0], stream)};
    }, specs).value();
    auto invocation = program.create_invocation().value();
    auto status = invocation.bind(std::array{mx::array(angles.data(), {224})});
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    auto work = invocation.submit().value();
    auto result = work.wait().value();
    auto copy = [](const mx::array &a) {
      return std::vector<float>(a.data<float>(), a.data<float>() + a.size());
    };
    cosines = copy(result.get()->arrays()[0]);
    sines = copy(result.get()->arrays()[1]);
    std::size_t coefficient_diff = 0;
    for (std::size_t i = 0; i < angles.size(); ++i)
      coefficient_diff += cosines[i] != std::cos(angles[i]) || sines[i] != std::sin(angles[i]);
    std::printf("[DSPARK-CAUSE] model=%s trig_coefficients_differ=%zu/224\n",
        model.profile->model_name.data(), coefficient_diff);
    status = context.close();
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  }
  std::vector<float> rotated(bool fused) const {
    auto output = attention;
    for (std::size_t i = 0; i < output.size(); i += 512) {
      const auto token = i / (model.heads * 512);
      const auto position_offset = (step % 3 + token) * 32;
      for (int pair = 0; pair < 32; ++pair) {
        const auto offset = i + 448 + pair * 2;
        const float a = attention[offset], b = attention[offset + 1];
        const float c = cosines[position_offset + pair], s = sines[position_offset + pair];
        if (fused) {
          output[offset] = base::fp8::bfloat(std::fma(-b, s, a * c));
          output[offset + 1] = base::fp8::bfloat(std::fma(b, c, a * s));
        } else {
          volatile float ac = a * c, bs = b * s, as = a * s, bc = b * c;
          output[offset] = base::fp8::bfloat(ac - bs);
          output[offset + 1] = base::fp8::bfloat(as + bc);
        }
      }
    }
    return output;
  }
  std::vector<float> grouped_equation() const {
    const auto n = model.groups * 1024;
    const d5_official_fp8_fragments::Case shape{model.profile, n, 4096, true, "wo_a"};
    std::vector<float> output(base::block * n);
    for (std::int64_t token = 0; token < base::block; ++token)
      for (std::int64_t row = 0; row < n; ++row) {
        double sum = 0;
        for (int term = 0; term < 2; ++term) {
          const auto col = (row * 17 + 283 + term * 2069) % 4096;
          const auto code = std::size_t((row + term * 3) % 8);
          const float coefficient = base::fp8::bfloat(d5_official_fp8_fragments::kValues[code] *
              d5_official_fp8_fragments::scale(shape, row / 128, col / 128, 283));
          sum += double(inverse[(token * model.groups + row / 1024) * 4096 + col]) * coefficient;
        }
        output[token * n + row] = base::fp8::bfloat(float(sum));
      }
    return output;
  }
  void observe(std::size_t port, std::span<const float> actual) {
    if (port == 1) attention.assign(actual.begin(), actual.end());
    if (port == 2) {
      inverse.assign(actual.begin(), actual.end());
      std::vector<std::int32_t> positions(base::block);
      std::iota(positions.begin(), positions.end(), int(129 + step % 3));
      const auto cpu = base::rotary_rows(attention, model.heads, positions, true);
      std::printf("[DSPARK-CAUSE] model=%s step=%zu inverse cpu=%zu gpu_unfused=%zu gpu_fused=%zu\n",
          model.profile->model_name.data(), step, differences(actual, cpu),
          differences(actual, rotated(false)), differences(actual, rotated(true)));
    }
    if (port == 3) {
      grouped.assign(actual.begin(), actual.end());
      const auto local = grouped_equation();
      std::printf("[DSPARK-CAUSE] model=%s step=%zu grouped_actual_input_diff=%zu/%zu\n",
          model.profile->model_name.data(), step, differences(actual, local), actual.size());
    }
    if (port == 4) {
      base::fp8::ModelLinear source{{}, model.hidden, model.groups * 1024, 457, true, false};
      const auto unfused = source.reference(grouped);
      source.fused_scale_accumulation = true;
      const auto fused = source.reference(grouped);
      std::printf("[DSPARK-CAUSE] model=%s step=%zu final_actual_input_unfused=%zu fused=%zu/%zu\n",
          model.profile->model_name.data(), step, differences(actual, unfused),
          differences(actual, fused), actual.size());
      ++step;
    }
  }
};
inline void attach(base::Fragment &fragment, const base::Model &model) {
  if (!std::getenv("HYLE_APPLE_DSPARK_CAUSE_CAPTURE") || fragment.region != 0x0c040102) return;
  auto state = std::make_shared<State>(model);
  fragment.observe = [original = std::move(fragment.observe), state]
      (std::size_t port, std::span<const float> actual) {
    if (original) original(port, actual);
    state->observe(port, actual);
  };
}
} // namespace apple_dspark_numeric_capture
