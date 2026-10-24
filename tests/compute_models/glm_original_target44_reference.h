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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "glm_hyper_connection_reference.h"
#include "glm_original_target44_fragment.h"
#include "glm_shared_vocabulary_reference.h"

// Local original-byte/actual-predecessor checks, not an executable model or a
// propagated ideal end-to-end oracle. HC deliberately follows fixed06992's
// torch/unfused branch. TileLang fused norm has a different raw-square versus
// BF16 numerator boundary and is not certified bit-identical here.
namespace glm_original_target44_reference {
namespace graph = glm_original_target44_fragment;
namespace roles = glm_original_target44_roles;
namespace lm = glm_shared_vocabulary_reference;
namespace hc = dspark_original_hc_reference;
namespace scalar = glm_original_indexer_reference::detail;
using Role = roles::Role;
using F32 = std::span<const float>;
using BF16 = std::span<const std::uint16_t>;
using Bytes = std::span<const std::byte>;
inline constexpr std::size_t kHidden = 4096, kStreams = 4, kHeads = 64;
inline constexpr std::size_t kDepth = 128, kWidth = kHeads * kDepth, kChannels = 3 * kWidth;

namespace detail = glm_hyper_connection_reference::detail;

inline bool expect_projection(F32 raw, BF16 stored, BF16 input, Bytes original,
    std::size_t outputs, std::size_t width, bool full, std::string_view role) {
  SCOPED_TRACE(std::string(role));
  if (!lm::expect_logits(raw, input, original, outputs, width, full)) return false;
  // Keep deterministic rows as well as the existing actual-top4 checks after
  // the first complete projection. They use the very same dot certificate.
  if (!full) {
    const auto rows = input.size() / width;
    for (std::size_t row = 0; row < rows; ++row) {
      const auto source = detail::widen(input.subspan(row * width, width));
      for (std::size_t sample = 0; sample < 7; ++sample) {
        const auto output = sample * (outputs - 1) / 6;
        const auto dot = lm::detail::dot(source, original.subspan(output * width * 2, width * 2));
        if (scalar::distance(raw[row * outputs + output], dot.center) > dot.bound) {
          ADD_FAILURE() << role << " fixed row=" << output << " exceeds original-byte dot bound"; return false;
        }
      }
    }
  }
  return glm_original_indexer_reference::expect_bfloat_store(stored, raw, role);
}

inline bool expect_conv(F32 raw, F32 activated_raw, BF16 stored, BF16 samples,
                       Bytes original, std::string_view role) {
  if (raw.size() != kWidth || activated_raw.size() != kWidth || stored.size() != kWidth ||
      samples.size() != 4 * kWidth || original.size() != kWidth * 4 * 2) {
    ADD_FAILURE() << role << " actual convolution extent"; return false;
  }
  for (std::size_t channel = 0; channel < kWidth; ++channel) {
    std::array<float, 4> actual_input{};
    for (std::size_t tap = 0; tap < 4; ++tap) actual_input[tap] = lm::detail::finite_bfloat(samples[tap * kWidth + channel]);
    const auto dot = lm::detail::dot(actual_input, original.subspan(channel * 8, 8));
    if (scalar::distance(raw[channel], dot.center) > dot.bound) {
      ADD_FAILURE() << role << " channel=" << channel << " original4-tap F64 dot bound=" << dot.bound; return false;
    }
  }
  std::vector<float> expected(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) expected[i] = float(double(raw[i]) * hc::detail::sigmoid(raw[i]));
  return detail::local(activated_raw, expected, role) &&
      glm_original_indexer_reference::expect_bfloat_store(stored, activated_raw, role);
}

template<class ReadValue>
inline bool observe(const roles::Data &data, const graph::Result &model, ReadValue &&actual, bool full) {
  const auto f32 = [&](graph::Value value) { return detail::floats(actual(value)); };
  const auto bf16 = [&](graph::Value value) { return detail::bfloats(actual(value)); };
  const auto &ports = model.ports;
  const auto norm = [&](const graph::NormPorts &part, Role gamma) {
    return glm_hyper_connection_reference::expect_norm(part, data.norm(gamma), actual);
  };
  const auto hc_pre = [&](const graph::HcPorts &part, Role fn, Role scales, Role bases, Role gamma) {
    return glm_hyper_connection_reference::expect_pre(part, data.bytes(fn), data.floats(scales), data.floats(bases), data.norm(gamma), actual);
  };
  const auto hc_post = [&](const graph::PostPorts &part, const graph::HcPorts &pre) {
    return glm_hyper_connection_reference::expect_post(part, pre, actual);
  };
  if (!hc_pre(ports.attention_hc, Role::kHcAttnFn, Role::kHcAttnScale, Role::kHcAttnBase, Role::kInputNorm)) return false;
  const auto &kda = ports.kda;
  constexpr std::array projection_roles{Role::kQuery, Role::kKey, Role::kValue, Role::kForgetA,
      Role::kForgetB, Role::kBeta, Role::kGateA, Role::kGateB, Role::kOutput};
  for (std::size_t i = 0; i < kda.projections.size(); ++i) {
    const auto role = projection_roles[i]; const auto &spec = roles::spec(role); const auto &part = kda.projections[i];
    auto predecessor = ports.attention_hc.norm.stored;
    if (i == std::size_t(graph::Projection::kForgetB)) predecessor = kda.projections[std::size_t(graph::Projection::kForgetA)].stored;
    if (i == std::size_t(graph::Projection::kGateB)) predecessor = kda.projections[std::size_t(graph::Projection::kGateA)].stored;
    if (i == std::size_t(graph::Projection::kOutput)) predecessor = kda.gated_stored;
    if (!detail::same_bfloat(bf16(part.input), bf16(predecessor), "original44 actual dense predecessor")) return false;
    if (!expect_projection(f32(part.raw), bf16(part.stored), bf16(part.input), data.bytes(role),
        std::size_t(spec.shape[0]), std::size_t(spec.shape[1]), full, spec.name)) return false;
  }
  const auto previous_history = f32(kda.history_input); const auto history = bf16(kda.history_stored);
  const auto window = bf16(kda.window), next_history = bf16(kda.next_history);
  if (history.size() != 3 * kChannels || window.size() != 4 * kChannels || next_history.size() != history.size()) {
    ADD_FAILURE() << "original44 chronological QKV history extent"; return false;
  }
  if (!glm_original_indexer_reference::expect_bfloat_store(history, previous_history, "KDA actual imported history BF16") ||
      !lm::expect_widen(previous_history, history) ||
      !detail::same_bfloat(BF16(window).first(history.size()), history, "KDA actual three history rows") ||
      !detail::same_bfloat(next_history, BF16(window).subspan(kChannels), "KDA next history rows1..3") ||
      !lm::expect_widen(f32(model.history), next_history)) return false;
  constexpr std::array conv_roles{Role::kQueryConv, Role::kKeyConv, Role::kValueConv};
  for (std::size_t leg = 0; leg < 3; ++leg) {
    const auto &part = kda.conv[leg]; const auto samples = bf16(part.samples);
    const auto projected = bf16(kda.projections[leg].stored);
    if (samples.size() != 4 * kWidth || projected.size() != kWidth) { ADD_FAILURE() << "KDA QKV lane extent"; return false; }
    for (std::size_t channel = 0; channel < kWidth; ++channel) {
      if (window[3 * kChannels + leg * kWidth + channel] != projected[channel]) {
        ADD_FAILURE() << "KDA history newest row does not retain actual QKV projection"; return false;
      }
      for (std::size_t tap = 0; tap < 4; ++tap)
        if (samples[tap * kWidth + channel] != window[tap * kChannels + leg * kWidth + channel]) {
          ADD_FAILURE() << "KDA QKV lane did not read actual chronological window"; return false;
        }
    }
    if (!expect_conv(f32(part.raw), f32(part.activated_raw), bf16(part.stored), samples,
        data.bytes(conv_roles[leg]), roles::spec(conv_roles[leg]).name)) return false;
  }
  const auto l2 = [&](const graph::L2Ports &part, BF16 source) {
    const auto input = f32(part.input), square = f32(part.square), sum = f32(part.sum);
    const auto denominator = f32(part.denominator), inverse = f32(part.inverse), output = f32(part.output);
    if (input.size() != kWidth || square.size() != kWidth || output.size() != kWidth ||
        sum.size() != kHeads || denominator.size() != kHeads || inverse.size() != kHeads || !lm::expect_widen(input, source)) {
      ADD_FAILURE() << "original44 F32 QK normalization extent"; return false;
    }
    for (std::size_t i = 0; i < kWidth; ++i)
      if (!detail::rounded(square[i], double(input[i]) * input[i], "QK actual square", i)) return false;
    std::vector<float> expected_inverse(kHeads);
    for (std::size_t head = 0; head < kHeads; ++head) {
      if (!detail::reduction(sum[head], F32(square).subspan(head * kDepth, kDepth), 1, "QK actual128-square sum") ||
          !detail::rounded(denominator[head], double(sum[head]) + double(graph::kL2Epsilon), "QK sum plus1e-6", head)) return false;
      expected_inverse[head] = float(1.0 / std::sqrt(double(denominator[head])));
    }
    if (!detail::local(inverse, expected_inverse, "QK actual inverse L2")) return false;
    for (std::size_t i = 0; i < kWidth; ++i)
      if (!detail::rounded(output[i], double(input[i]) * inverse[i / kDepth], "QK F32 normalized output", i)) return false;
    return true;
  };
  if (!l2(kda.query, bf16(kda.conv[0].stored)) || !l2(kda.key, bf16(kda.conv[1].stored)) ||
      !lm::expect_widen(f32(kda.value), bf16(kda.conv[2].stored))) return false;
  const auto alpha = f32(kda.alpha), forget_add = f32(kda.forget_add), forget_product = f32(kda.forget_product);
  const auto forget_sigmoid = f32(kda.forget_sigmoid), decay = f32(kda.decay), beta = f32(kda.beta);
  const auto forget_input = detail::widen(bf16(kda.projections[std::size_t(graph::Projection::kForgetB)].stored));
  const auto beta_input = detail::widen(bf16(kda.projections[std::size_t(graph::Projection::kBeta)].stored));
  if (alpha.size() != kHeads || forget_add.size() != kWidth || forget_product.size() != kWidth ||
      forget_sigmoid.size() != kWidth || decay.size() != kWidth || beta.size() != kHeads) {
    ADD_FAILURE() << "original44 safe gate extent"; return false;
  }
  std::vector<float> expected_alpha(kHeads);
  for (std::size_t head = 0; head < kHeads; ++head) expected_alpha[head] = float(std::exp(double(data.floats(Role::kALog)[head])));
  if (!detail::local(alpha, expected_alpha, "KDA exp(original A_log)") ||
      !detail::local(beta, detail::sigmoid(beta_input), "KDA sigmoid actual beta BF16")) return false;
  for (std::size_t i = 0; i < kWidth; ++i)
    if (!detail::rounded(forget_add[i], double(forget_input[i]) + data.floats(Role::kDtBias)[i], "KDA actual f plus dt", i) ||
        !detail::rounded(forget_product[i], double(forget_add[i]) * alpha[i / kDepth], "KDA actual alpha product", i) ||
        !detail::rounded(decay[i], double(forget_sigmoid[i]) * -5.0, "KDA safe decay -5", i)) return false;
  if (!detail::local(forget_sigmoid, detail::sigmoid(forget_product), "KDA actual safe gate sigmoid")) return false;

  // Same scalar recurrence as the existing KDA reference, started at these
  // actual F32 ports. Physical state is [H,V,K] in both fixed06992 and Hyle.
  const auto query = f32(kda.query.output), key = f32(kda.key.output), value = f32(kda.value);
  const auto initial_state = f32(kda.state_input), state = f32(kda.state_output), attended = f32(kda.recurrent_raw);
  if (initial_state.size() != kHeads * kDepth * kDepth || state.size() != initial_state.size() || attended.size() != kWidth) {
    ADD_FAILURE() << "original44 recurrent state extent"; return false;
  }
  std::vector<float> expected_state(state.size()), expected_output(kWidth);
  for (std::size_t head = 0; head < kHeads; ++head)
    for (std::size_t v = 0; v < kDepth; ++v) {
      const auto row = (head * kDepth + v) * kDepth;
      std::array<double, kDepth> decayed{}; scalar::Sum prediction;
      for (std::size_t k = 0; k < kDepth; ++k) {
        decayed[k] = double(initial_state[row + k]) * std::exp(double(decay[head * kDepth + k]));
        prediction.add(decayed[k] * key[head * kDepth + k]);
      }
      const double delta = double(beta[head]) * (double(value[head * kDepth + v]) - prediction.value());
      scalar::Sum output;
      for (std::size_t k = 0; k < kDepth; ++k) {
        expected_state[row + k] = float(decayed[k] + delta * key[head * kDepth + k]);
        // Output is checked from actual F32 state, not the ideal state above.
        output.add(double(state[row + k]) * query[head * kDepth + k]);
      }
      expected_output[head * kDepth + v] = float(output.value() / std::sqrt(double(kDepth)));
    }
  if (!detail::local(state, expected_state, "KDA actual per-step F32 state") ||
      !detail::local(attended, expected_output, "KDA actual-state output with1/sqrt128") ||
      !glm_original_indexer_reference::expect_bfloat_store(bf16(kda.recurrent_stored), attended, "KDA recurrent output BF16") ||
      !lm::expect_widen(f32(kda.output_norm_input), bf16(kda.recurrent_stored))) return false;
  const auto norm_input = f32(kda.output_norm_input), norm_raw = f32(kda.output_norm_raw);
  const auto gate_sigmoid = f32(kda.gate_sigmoid), gated_raw = f32(kda.gated_raw);
  const auto gate_input = detail::widen(bf16(kda.projections[std::size_t(graph::Projection::kGateB)].stored));
  const auto gamma = detail::widen(data.norm(Role::kOutputNorm));
  if (!detail::local(norm_raw, hc::rms(norm_input, gamma, roles::kNormEpsilon), "KDA actual BF16 output RMS1e-5") ||
      !detail::local(gate_sigmoid, detail::sigmoid(gate_input), "KDA sigmoid actual g_b BF16")) return false;
  if (gated_raw.size() != kWidth || norm_raw.size() != kWidth || gate_sigmoid.size() != kWidth) {
    ADD_FAILURE() << "original44 norm-gate extent"; return false;
  }
  for (std::size_t i = 0; i < kWidth; ++i)
    if (!detail::rounded(gated_raw[i], double(norm_raw[i]) * gate_sigmoid[i], "KDA actual RMS times sigmoid", i)) return false;
  if (!glm_original_indexer_reference::expect_bfloat_store(bf16(kda.gated_stored), gated_raw, "KDA fused norm-gate single BF16 store") ||
      !detail::same_bfloat(bf16(ports.attention_post.branch), bf16(kda.projections[std::size_t(graph::Projection::kOutput)].stored), "HC attention actual O branch") ||
      !hc_post(ports.attention_post, ports.attention_hc) ||
      !detail::same_bfloat(bf16(ports.ffn_hc.residual), bf16(ports.attention_post.stored), "HC FFN actual preceding four streams") ||
      !hc_pre(ports.ffn_hc, Role::kHcFfnFn, Role::kHcFfnScale, Role::kHcFfnBase, Role::kPostNorm) ||
      !detail::same_bfloat(bf16(ports.ffn_post.branch), bf16(model.moe.output), "HC FFN actual original MoE branch") ||
      !detail::same_bfloat(bf16(ports.ffn_post.residual), bf16(ports.attention_post.stored), "HC FFN actual residual") ||
      !hc_post(ports.ffn_post, ports.ffn_hc)) return false;
  const auto expanded = f32(ports.ffn_post.visible), contract = f32(ports.contract_raw);
  if (expanded.size() != kStreams * kHidden || contract.size() != kHidden) {
    ADD_FAILURE() << "original44 final contract extent"; return false;
  }
  for (std::size_t column = 0; column < kHidden; ++column) {
    std::array<float, kStreams> streams{};
    for (std::size_t stream = 0; stream < kStreams; ++stream) streams[stream] = expanded[stream * kHidden + column];
    if (!detail::reduction(contract[column], streams, 4, "target final actual four-stream mean")) return false;
  }
  const auto stored_contract = bf16(ports.contract_stored);
  return glm_original_indexer_reference::expect_bfloat_store(stored_contract, contract, "target mean BF16 before model norm") &&
      lm::expect_widen(f32(ports.model_norm.input), stored_contract) && norm(ports.model_norm, Role::kModelNorm) &&
      lm::expect_widen(f32(model.mhc), bf16(ports.ffn_post.stored)) &&
      lm::expect_widen(f32(model.tap), bf16(ports.model_norm.stored));
}
} // namespace glm_original_target44_reference
