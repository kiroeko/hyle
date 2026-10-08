#pragma once

#include <memory>
#include <mlx/fast.h>
#include <mlx/transforms.h>

// Integration peer for the original MLX math. Independent FP8/weight/norm
// equations and primitive accuracy gates remain; no product lowering is used.
namespace apple_dspark_native_reference {
namespace mx = mlx::core;
namespace base = cpu_c4_model_branches;
struct Native final {
  std::vector<float> cosines, sines;
  static constexpr int positions = 136, pairs = 32;
  explicit Native() {
    std::vector<float> angles;
    for (bool inverse : {false, true}) {
      const auto frequency = base::ds::rope_frequencies(false, inverse);
      for (int position = 0; position < positions; ++position)
        for (float f : frequency) angles.push_back(float(position) * f);
    }
    auto context = hyle::compute::apple::Context::create().value();
    const std::array specs{hyle::compute::apple::InputSpec{{int(angles.size())}, mx::float32}};
    auto program = context.compile([](const auto &input, mx::Stream stream) {
      return std::vector{mx::cos(input[0], stream), mx::sin(input[0], stream)};
    }, specs).value();
    auto invocation = program.create_invocation().value();
    auto status = invocation.bind(std::array{mx::array(angles.data(), {int(angles.size())})});
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    auto work = invocation.submit().value();
    auto result = work.wait().value();
    cosines = copy(result.get()->arrays()[0]);
    sines = copy(result.get()->arrays()[1]);
    status = context.close();
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  }
  static std::vector<float> copy(const mx::array &array) {
    return {array.data<float>(), array.data<float>() + array.size()};
  }
  std::vector<float> rotate(std::span<const float> input, std::int64_t heads,
      std::span<const std::int32_t> p, bool inverse) const {
    if (input.size() != p.size() * heads * 512)
      throw std::runtime_error("native reference rotary shape differs");
    std::vector<float> result(input.begin(), input.end());
    for (std::size_t token = 0; token < p.size(); ++token) {
      if (p[token] < 0 || p[token] >= positions)
        throw std::runtime_error("native reference rotary position not fixed");
      const auto coefficient = ((inverse ? positions : 0) + p[token]) * pairs;
      for (std::int64_t head = 0; head < heads; ++head)
        for (int pair = 0; pair < pairs; ++pair) {
          const auto i = (token * heads + head) * 512 + 448 + 2 * pair;
          const float c = cosines[coefficient + pair], s = sines[coefficient + pair];
          // Fix the two F32 products and sum before any BF16 boundary.
          volatile float ac = input[i] * c, bs = input[i + 1] * s;
          volatile float as = input[i] * s, bc = input[i + 1] * c;
          result[i] = ac - bs;
          result[i + 1] = as + bc;
        }
    }
    return base::ds::bfloat_boundary(std::move(result));
  }
  static std::vector<float> attention(std::span<const float> query,
      std::span<const float> keys, std::span<const float> values,
      std::span<const float> sinks, float scale) {
    const int heads = int(sinks.size()), tokens = int(base::block), length = int(base::window + base::block);
    if (query.size() != std::size_t(tokens * heads * 512) ||
        keys.size() != std::size_t(length * 512) || values.size() != keys.size())
      throw std::runtime_error("native reference attention shape differs");
    const auto stream = mx::default_stream(mx::Device::gpu);
    auto q = mx::transpose(mx::array(query.data(), {1, tokens, heads, 512}), {0, 2, 1, 3}, stream);
    auto k = mx::transpose(mx::array(keys.data(), {1, length, 1, 512}), {0, 2, 1, 3}, stream);
    auto v = mx::transpose(mx::array(values.data(), {1, length, 1, 512}), {0, 2, 1, 3}, stream);
    auto mask = mx::zeros({tokens, length}, mx::float32, stream);
    auto output = mx::fast::scaled_dot_product_attention(q, k, v, scale, "array", mask,
        mx::array(sinks.data(), {heads}), false, stream);
    output = mx::contiguous(mx::transpose(output, {0, 2, 1, 3}, stream), false, stream);
    mx::async_eval({output});
    mx::synchronize(stream);
    output.wait();
    return copy(output);
  }
};
inline void bind(base::Model &model) {
  auto native = std::make_shared<Native>();
  model.rotation_reference = [native](auto input, auto heads, auto positions, bool inverse) {
    return native->rotate(input, heads, positions, inverse);
  };
  model.attention_reference = Native::attention;
}
} // namespace apple_dspark_native_reference
