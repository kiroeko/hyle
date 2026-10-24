#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "glm_original_attention_reference.h"
#include "glm_moe_geometry.h"

// Original GLM MoE local references, with explicit model-local geometry. Read authenticated original bytes
// and actual rounded predecessors; never feed reference values into the Graph.
namespace glm_original_moe_reference {
namespace attention = glm_original_attention_reference;
namespace scalar = glm_original_indexer_reference::detail;
namespace lm = glm_shared_vocabulary_reference;
using Geometry=glm_moe_geometry::Geometry;
inline constexpr auto kFlashGeometry=glm_moe_geometry::kFlash;
inline constexpr auto kProGeometry=glm_moe_geometry::kPro;
inline constexpr std::size_t kExperts=std::size_t(kFlashGeometry.experts),kTopK=kFlashGeometry.slots;
inline constexpr float kScaling=kFlashGeometry.route_scale,kSumEpsilon=1.0e-20F;

struct Fp8Matrix final {
  std::span<const std::byte> codes, scales;
  std::size_t outputs, width;
};

inline bool expect_router(std::span<const float> actual,
    std::span<const std::uint16_t> actual_input,
    std::span<const std::byte> original_bfloat_weight,
    const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  const auto experts=std::size_t(model.experts);
  if (!experts || actual_input.empty() || actual_input.size() >
      std::numeric_limits<std::size_t>::max() / experts / 2 ||
      actual.size() != experts || original_bfloat_weight.size() != experts * actual_input.size() * 2) {
    ADD_FAILURE() << "original GLM BF16 router extent"; return false;
  }
  std::vector<float> input(actual_input.size());
  for (std::size_t column = 0; column < input.size(); ++column) {
    lm::detail::normal_bfloat(actual_input[column]);
    input[column] = lm::detail::finite_bfloat(actual_input[column]);
  }
  double maximum_error = 0, maximum_ratio = 0;
  for (std::size_t expert = 0; expert < experts; ++expert) {
    // Reuse the independently checked blocked-F64 BF16 dot certificate from
    // the full vocabulary head, without expanding this original BF16 bank.
    const auto reference = lm::detail::dot(input,
        original_bfloat_weight.subspan(expert * input.size() * 2, input.size() * 2));
    const auto error = scalar::distance(actual[expert], reference.center);
    if (!std::isfinite(actual[expert]) || error > reference.bound) {
      ADD_FAILURE() << "original GLM router[" << expert << "] actual=" << actual[expert]
          << " F64=" << reference.center << " error=" << error
          << " F32 accumulation bound=" << reference.bound; return false;
    }
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, reference.bound ? error / reference.bound : 0.0);
  }
  std::printf("GLM original BF16 router outputs=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      experts, maximum_error, maximum_ratio);
  return true;
}

inline bool expect_fp8_projection(std::span<const float> actual,
    std::span<const std::uint16_t> actual_input, Fp8Matrix matrix,
    const char *role, bool full,glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  arithmetic.require();
  return attention::expect_fp8_matrix(actual, actual_input, matrix.codes,
      matrix.scales, matrix.outputs, matrix.width, role, full,arithmetic.policy,arithmetic.absmax_floor);
}

// A source(globalID) returns that exact original expert's row-major E4 codes
// and F32 block128 scales. One common input row or one actual row per selected
// slot is admitted. No global matrix expansion, modulo-ID mapping or padding
// expert is used by this independent reference.
template<class Source>
bool expect_selected_projection(std::span<const float> actual,
    std::span<const std::uint16_t> actual_input,
    std::span<const std::int32_t> global_ids, Source &&source,
    const char *role,bool full,const Geometry &model=kFlashGeometry,glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  glm_moe_geometry::require(model);
  const auto experts=std::size_t(model.experts),topk=model.slots;
  if (global_ids.size() != topk) {
    ADD_FAILURE() << role << " selected projection needs actual top8 global IDs"; return false;
  }
  for (const auto global : global_ids)
    if (global < 0 || std::size_t(global) >= experts) {
      ADD_FAILURE() << role << " original global expert ID is out of range"; return false;
    }
  const auto first = source(global_ids.front());
  if (!first.outputs || !first.width || first.outputs > std::numeric_limits<std::size_t>::max() / topk ||
      first.width > std::numeric_limits<std::size_t>::max() / topk ||
      actual.size() != topk * first.outputs ||
      (actual_input.size() != first.width && actual_input.size() != topk * first.width)) {
    ADD_FAILURE() << role << " selected original E4 projection extent"; return false;
  }
  for (std::size_t slot = 0; slot < topk; ++slot) {
    const auto matrix = source(global_ids[slot]);
    if (matrix.outputs != first.outputs || matrix.width != first.width) {
      ADD_FAILURE() << role << " selected original experts have different geometries"; return false;
    }
    const auto input_row = actual_input.size() == first.width ? 0 : slot;
    if (!expect_fp8_projection(actual.subspan(slot * first.outputs, first.outputs),
        actual_input.subspan(input_row * first.width, first.width), matrix, role, full,arithmetic)) return false;
  }
  return true;
}

namespace detail {
inline bool exact(float actual, float expected, const char *role, std::size_t index) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " expected=" << expected;
    return false;
  }
  return true;
}
// Preserve the existing semantic local nonlinear gate. This is separate from
// every analytic matrix/reduction bound and never spans a BF16 discontinuity.
inline bool local(float actual, float expected, const char *role, std::size_t index) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(double(actual) - double(expected)) > 8.0e-5 + 4.0e-5 * std::abs(double(expected))) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " local expected=" << expected;
    return false;
  }
  return true;
}
inline double sigmoid(double value) {
  if (value >= 0) return 1.0 / (1.0 + std::exp(-value));
  const auto exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}
template<class Term>
attention::detail::Dot sum(std::size_t count, Term &&term) {
  scalar::Sum reference;
  double absolute = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = term(i);
    // These actual F32 terms need a normal F32 quantum so every admitted
    // reduction tree, including cancellation, is independent of FTZ.
    if (!std::isfinite(value) || (value != 0 && std::abs(value) < 0x1p-103))
      throw std::runtime_error("original GLM MoE reduction needs separate underflow analysis");
    reference.add(value); absolute += std::abs(value);
  }
  const double g64 = scalar::gamma(count, 0x1p-53);
  const double upper_sum = absolute == 0 ? 0 : scalar::upper(absolute / (1 - g64));
  if (!std::isfinite(upper_sum) || upper_sum > double(std::numeric_limits<float>::max()) / 4)
    throw std::runtime_error("original GLM MoE reduction needs separate overflow analysis");
  const double center = reference.value();
  const double oracle = scalar::upper_add(scalar::distance(center, reference.plain),
      scalar::upper_multiply(g64, upper_sum));
  const double native = scalar::upper_multiply(scalar::gamma(count, 0x1p-24), upper_sum);
  return {center, absolute == 0 ? 0 : scalar::upper_add(native, oracle)};
}
inline bool reduced(float actual, const attention::detail::Dot &reference,
    const char *role, std::size_t index) {
  const auto error = scalar::distance(actual, reference.center);
  if (!std::isfinite(actual) || error > reference.bound) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " F64=" << reference.center
        << " error=" << error << " analytic F32 bound=" << reference.bound; return false;
  }
  return true;
}
} // namespace detail

struct RoutingValues final {
  std::span<const float> logits, scores, selection, selected_scores;
  std::span<const float> sum, denominator, normalized, weights;
  std::span<const std::int32_t> ids;
};

inline bool expect_routing(const RoutingValues &actual,
    std::span<const std::byte> original_f32_bias,const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  const auto experts=std::size_t(model.experts),topk=model.slots;
  if (actual.logits.size() != experts || actual.scores.size() != experts ||
      actual.selection.size() != experts || original_f32_bias.size() != experts * 4 ||
      actual.ids.size() != topk || actual.selected_scores.size() != topk ||
      actual.sum.size() != 1 || actual.denominator.size() != 1 ||
      actual.normalized.size() != topk || actual.weights.size() != topk) {
    ADD_FAILURE() << "original GLM routing boundary extent"; return false;
  }
  for (std::size_t expert = 0; expert < experts; ++expert) {
    if (!std::isfinite(actual.logits[expert]) || actual.scores[expert] < 0 || actual.scores[expert] > 1 ||
        !detail::local(actual.scores[expert], float(detail::sigmoid(actual.logits[expert])),
            "original GLM actual-logit sigmoid", expert)) return false;
    const auto bias = attention::detail::read_f32(original_f32_bias, expert);
    if (!std::isfinite(bias)) throw std::runtime_error("original GLM correction bias is nonfinite");
    if (!detail::exact(actual.selection[expert], float(double(actual.scores[expert]) + double(bias)),
        "original GLM actual-score F32 bias addition", expert)) return false;
  }
  // n_group=topk_group=1: the original top2 group score cannot exclude an
  // expert. Hyle generic TopK is sorted+stable; verify its actual predecessor,
  // without asserting the same tie order for unsorted vendor top-k kernels.
  std::vector<std::int32_t> order(experts);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return actual.selection[std::size_t(left)] > actual.selection[std::size_t(right)];
  });
  for (std::size_t slot = 0; slot < topk; ++slot) {
    if (actual.ids[slot] != order[slot]) {
      ADD_FAILURE() << "original GLM actual selection TopK slot=" << slot
          << " global=" << actual.ids[slot] << " expected=" << order[slot]; return false;
    }
    if (!detail::exact(actual.selected_scores[slot], actual.scores[std::size_t(actual.ids[slot])],
        "original GLM unbiased gathered score", slot)) return false;
  }
  const auto reference = detail::sum(topk, [&](std::size_t slot) { return double(actual.selected_scores[slot]); });
  if (!detail::reduced(actual.sum[0], reference, "original GLM selected-score sum", 0) ||
      !detail::exact(actual.denominator[0], float(double(actual.sum[0]) + double(kSumEpsilon)),
          "original GLM actual sum plus1e-20", 0)) return false;
  if (!(actual.denominator[0] > 0)) {
    ADD_FAILURE() << "original GLM routing denominator must be positive"; return false;
  }
  for (std::size_t slot = 0; slot < topk; ++slot) {
    if (!detail::local(actual.normalized[slot], float(double(actual.selected_scores[slot]) / double(actual.denominator[0])),
        "original GLM selected-score normalization", slot) ||
        !detail::exact(actual.weights[slot], float(double(actual.normalized[slot]) * double(model.route_scale)),
            "original GLM exactly once routed scale2.5", slot)) return false;
  }
  return true;
}

inline bool expect_mapping(std::span<const std::int32_t> global_ids,
    std::span<const std::int32_t> global_to_local, std::span<const std::int32_t> local_ids,
    std::span<const std::int32_t> resident_globals,const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  const auto experts=std::size_t(model.experts),topk=model.slots;
  if (global_ids.size() != topk || local_ids.size() != topk ||
      global_to_local.size() != experts || resident_globals.empty()) {
    ADD_FAILURE() << "original GLM compact expert map extent"; return false;
  }
  std::vector<bool> selected(experts,false);
  for (std::size_t slot = 0; slot < topk; ++slot) {
    const auto global = global_ids[slot], local = local_ids[slot];
    if (global < 0 || std::size_t(global) >= experts || local < 0 ||
        std::size_t(local) >= resident_globals.size() ||
        global_to_local[std::size_t(global)] != local || resident_globals[std::size_t(local)] != global) {
      ADD_FAILURE() << "original GLM exact global-to-local identity slot=" << slot; return false;
    }
    selected[std::size_t(global)] = true;
  }
  for (std::size_t global = 0; global < experts; ++global)
    if (!selected[global] && global_to_local[global] != -1) {
      ADD_FAILURE() << "original GLM unselected expert must not have an active map entry"; return false;
    }
  return true;
}

inline bool expect_activation(std::span<const float> gate_clipped,
    std::span<const float> up_clipped, std::span<const float> actual_gated,
    std::span<const std::uint16_t> actual_gate,std::span<const std::uint16_t> actual_up,
    const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  if (actual_gate.empty() || actual_gate.size() != actual_up.size() ||
      gate_clipped.size() != actual_gate.size() || up_clipped.size() != actual_gate.size() ||
      actual_gated.size() != actual_gate.size()) {
    ADD_FAILURE() << "original GLM local clamped SwiGLU extent"; return false;
  }
  for (std::size_t i = 0; i < actual_gate.size(); ++i) {
    const auto gate = lm::detail::finite_bfloat(actual_gate[i]);
    const auto up = lm::detail::finite_bfloat(actual_up[i]);
    if (!detail::exact(gate_clipped[i], (model.swiglu_limit?std::min(gate,*model.swiglu_limit):gate), "original GLM gate clip before SiLU", i) ||
        !detail::exact(up_clipped[i], (model.swiglu_limit?std::clamp(up,-*model.swiglu_limit,*model.swiglu_limit):up), "original GLM up clip", i)) return false;
    const float expected = float(double(gate_clipped[i]) * detail::sigmoid(gate_clipped[i]) * double(up_clipped[i]));
    if (!detail::local(actual_gated[i], expected, "original GLM actual clipped SwiGLU", i)) return false;
  }
  return true;
}

inline bool expect_weighted_down(std::span<const float> actual,
    std::span<const float> raw_down,std::span<const float> actual_weights,
    const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  const auto topk=model.slots;
  if (actual_weights.size() != topk || raw_down.empty() || raw_down.size() % topk || actual.size() != raw_down.size()) {
    ADD_FAILURE() << "original GLM post-down routing weight extent"; return false;
  }
  const auto width = raw_down.size() / topk;
  for (std::size_t i = 0; i < raw_down.size(); ++i) {
    const float expected = float(double(raw_down[i]) * double(actual_weights[i / width]));
    if (expected != 0 && !std::isnormal(expected))
      throw std::runtime_error("original GLM weighted down needs separate underflow analysis");
    if (!detail::exact(actual[i], expected, "original GLM weight after down before BF16 store", i)) return false;
  }
  return true;
}

inline bool expect_routed_sum(std::span<const float> actual,
    std::span<const std::uint16_t> actual_weighted_down,const Geometry &model=kFlashGeometry) {
  glm_moe_geometry::require(model);
  const auto topk=model.slots;
  if (actual.empty() || actual.size() > std::numeric_limits<std::size_t>::max() / topk ||
      actual_weighted_down.size() != actual.size() * topk) {
    ADD_FAILURE() << "original GLM routed8 reduction extent"; return false;
  }
  for (std::size_t column = 0; column < actual.size(); ++column) {
    const auto reference = detail::sum(topk, [&](std::size_t slot) {
      return double(lm::detail::finite_bfloat(actual_weighted_down[slot * actual.size() + column]));
    });
    if (!detail::reduced(actual[column], reference, "original GLM actual weighted BF16 sum8", column)) return false;
  }
  return true;
}
} // namespace glm_original_moe_reference
