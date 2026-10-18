#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>

// One bounded capture of the published output projection. It does not change
// the fixture, independent whole-chain reference, or any acceptance gate.
namespace apple_fp4_main_numeric_capture {
namespace ds = d5_deepseek_official_attention;
inline std::size_t differences(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) throw std::runtime_error("FP4 capture geometry differs");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}
inline void attach(ds::Fragment &fragment, const cpu_c4_model_branches::Model &model) {
  if (!std::getenv("HYLE_APPLE_FP4_MAIN_CAUSE") || fragment.region != 0x080a3405 ||
      model.hidden != 7168) return;
  struct State {
    std::size_t step = 0;
    std::vector<float> grouped;
    std::vector<std::vector<float>> expected_grouped, expected_final;
    ds::fp8::ModelLinear source;
  };
  auto state = std::make_shared<State>();
  state->source = {{}, model.hidden, model.groups * 1024, 293, true, false};
  for (const auto &replay : fragment.replays) {
    state->expected_grouped.push_back(replay.outputs.at(13));
    state->expected_final.push_back(replay.outputs.at(14));
  }
  fragment.observe = [state](std::size_t port, std::span<const float> actual) {
    if (port == 13) state->grouped.assign(actual.begin(), actual.end());
    if (port != 14) return;
    const auto local_unfused = state->source.reference(state->grouped);
    const auto whole_unfused = state->source.reference(state->expected_grouped.at(state->step));
    state->source.fused_scale_accumulation = true;
    const auto local_fused = state->source.reference(state->grouped);
    const auto whole_fused = state->source.reference(state->expected_grouped.at(state->step));
    state->source.fused_scale_accumulation = false;
    const auto &expected = state->expected_final.at(state->step);
    float error = 0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (const auto delta = std::abs(actual[i] - expected[i]); delta > error) {
        error = delta;
        worst = i;
      }
    std::printf("[FP4-MAIN-CAUSE] step=%zu grouped_diff=%zu local_unfused=%zu local_fused=%zu whole_unfused=%zu whole_fused=%zu worst=%zu actual=%g expected=%g max_error=%g\n",
        state->step, differences(state->grouped, state->expected_grouped.at(state->step)),
        differences(actual, local_unfused), differences(actual, local_fused),
        differences(actual, whole_unfused), differences(actual, whole_fused),
        worst, actual[worst], expected[worst], error);
    ++state->step;
  };
}
} // namespace apple_fp4_main_numeric_capture
