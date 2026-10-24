#pragma once

#include "glm_fp8_arithmetic.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "glm_hyper_connection_reference.h"
#include "glm_original_attention_reference.h"
#include "glm_original_target43_fragment.h"
#include "glm_original_target43_roles.h"

// Only the original43 shell and q_a prefix. Existing original-byte MLA,
// indexer and MoE observers check those same actual subgraphs separately.
// Upstream42's four streams are a declared synthetic BF16 input, not a claim
// of an original complete target or a task-quality result.
namespace glm_original_target43_reference {
namespace graph = glm_original_target43_fragment;
namespace roles = glm_original_target43_roles;
namespace common = glm_hyper_connection_reference;
namespace attention = glm_original_attention_reference;
namespace lm = glm_shared_vocabulary_reference;
namespace detail = glm_hyper_connection_reference::detail;
using Value = graph::Value;
using Role = roles::Role;
using RawRole = roles::RawRole;

// q_a is appended by the shared sparse-MLA builder between43prepare/finish.
// Each field is its real producer Value, observed by the caller's same Graph.
struct QueryPorts final {
  Value input, raw, stored, norm_input, norm_raw, norm_stored, norm_visible;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for (auto *value : {&input, &raw, &stored, &norm_input, &norm_raw, &norm_stored, &norm_visible}) visitor(*value);
  }
};

template<class ReadValue>
inline bool observe(const roles::Data &data, const graph::Result &model,
                    const QueryPorts &query, ReadValue &&actual, bool full,
                    glm_fp8_arithmetic::Arithmetic arithmetic={}) {
  arithmetic.require();
  const auto f32 = [&](Value value) { return detail::floats(actual(value)); };
  const auto bf16 = [&](Value value) { return detail::bfloats(actual(value)); };
  const auto &ports = model.ports;
  if (!common::expect_pre(ports.attention_hc, data.bytes(RawRole::kHcAttnFn),
      data.floats(Role::kHcAttnScale), data.floats(Role::kHcAttnBase), data.norm(Role::kInputNorm), actual)) return false;

  const auto input = bf16(query.input), stored = bf16(query.stored);
  const auto raw = f32(query.raw);
  if (!detail::same_bfloat(input, bf16(ports.attention_hc.norm.stored), "original43 q_a actual HC-normalized input") ||
      !attention::expect_fp8_matrix(raw, input, data.bytes(RawRole::kQueryCodes), data.bytes(RawRole::kQueryScale),
          std::size_t(roles::kQueryRank), std::size_t(roles::kHidden), "original43_q_a", full,
          arithmetic.policy,arithmetic.absmax_floor) ||
      !glm_original_indexer_reference::expect_bfloat_store(stored, raw, "original43 q_a actual BF16 store") ||
      !lm::expect_widen(f32(query.norm_input), stored)) return false;
  const glm_hyper_connection_fragment::NormPorts query_norm{
      query.norm_input, query.norm_raw, query.norm_stored, query.norm_visible};
  if (!common::expect_norm(query_norm, data.norm(Role::kQueryNorm), actual)) return false;

  // The MLA observer starts from the actual q_aNorm BF16 and HC input above.
  // Its raw O BF16 output is the attention branch, with no NextN EH/add/norm
  // inserted before this first HC post.
  if (!detail::same_bfloat(bf16(ports.attention_post.residual), bf16(ports.attention_hc.residual), "original43 attention retained four streams") ||
      !common::expect_post(ports.attention_post, ports.attention_hc, actual) ||
      !detail::same_bfloat(bf16(ports.ffn_hc.residual), bf16(ports.attention_post.stored), "original43 FFN actual attention post")) return false;
  const auto postnorm = data.attention.norm(glm_original_attention_roles::Parameter::kPostNorm);
  if (!common::expect_pre(ports.ffn_hc, data.bytes(RawRole::kHcFfnFn),
      data.floats(Role::kHcFfnScale), data.floats(Role::kHcFfnBase), postnorm, actual) ||
      !detail::same_bfloat(bf16(ports.ffn_post.branch), bf16(model.moe.output), "original43 actual original MoE branch") ||
      !detail::same_bfloat(bf16(ports.ffn_post.residual), bf16(ports.attention_post.stored), "original43 FFN retained four streams") ||
      !common::expect_post(ports.ffn_post, ports.ffn_hc, actual)) return false;
  return detail::same_bfloat(bf16(model.stored), bf16(ports.ffn_post.stored), "original43 output four-stream BF16") &&
      lm::expect_widen(f32(model.mhc), bf16(model.stored));
}
} // namespace glm_original_target43_reference
