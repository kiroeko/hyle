#pragma once

#include "d5_dynamic_fp8_fragments.h"
#include "d5_moe_execution_fragments.h"

namespace d5_grouped_fp8_fragments {
namespace dynamic = d5_dynamic_fp8_fragments;
namespace fp8 = d5_official_fp8_fragments;
namespace c = fp8::c;
namespace sem = fp8::sem;

inline float scale(std::int64_t expert, std::int64_t row, std::int64_t group) {
  return float(17 + (expert * 5 + row * 3 + group + 13) % 11) / 1024.0F;
}
inline std::vector<std::byte> weight_bytes(std::int64_t e, std::int64_t n, std::int64_t k) {
  const auto desc = c::float8_grouped_weight_description(e, n, k).value();
  std::vector<std::byte> bytes(desc.required_bytes().value());
  const auto nb = (n + 127) / 128, kb = k / 128;
  for (std::int64_t ex = 0; ex < e; ++ex) {
    for (std::int64_t row = 0; row < n; ++row)
      for (int term = 0; term < 2; ++term) {
        const auto col = (row * 17 + 31 + ex * 7 + term * (k / 2 + 13)) % k;
        bytes[(ex * n + row) * k + col] = std::byte{fp8::kCodes[(row + ex + term * 3) % 8]};
      }
    for (std::int64_t row = 0; row < nb; ++row)
      for (std::int64_t block = 0; block < kb; ++block) {
        const float value = scale(ex, row, block);
        std::memcpy(bytes.data() + e * n * k + ((ex * nb + row) * kb + block) * 4, &value, 4);
      }
  }
  return bytes;
}
inline std::vector<float> reference(std::span<const float> input,
                                    std::span<const std::int32_t> ids,
                                    std::int64_t n, std::int64_t k, bool shared, bool bf16) {
  const auto encoded = dynamic::quantized(input, k, false);
  std::vector<float> output(ids.size() * n);
  for (std::size_t slot = 0; slot < ids.size(); ++slot) {
    const auto ex = ids[slot];
    const auto token = std::int64_t(shared ? slot / 8 : slot);
    for (std::int64_t row = 0; row < n; ++row) {
      double value = 0;
      for (int term = 0; term < 2; ++term) {
        const auto col = (row * 17 + 31 + ex * 7 + term * (k / 2 + 13)) % k;
        value += encoded[token * k + col] * fp8::kValues[(row + ex + term * 3) % 8] *
                 scale(ex, row / 128, col / 128);
      }
      output[slot * n + row] = bf16 ? dynamic::bfloat(float(value)) : float(value);
    }
  }
  return output;
}
inline sem::ValueId weight(sem::Fragment &f, std::int64_t e, std::int64_t n, std::int64_t k) {
  return f.constant(c::float8_grouped_weight_description(e, n, k).value(), weight_bytes(e,n,k));
}
inline sem::Fragment projection(const fp8::Case &shape, std::int64_t m, bool bf16,
                                 bool shared, std::int64_t experts = 9) {
  sem::Fragment f{.profile = shape.profile, .region = 0x0b0606};
  const auto a = shared ? 1 : 8;
  auto input = f.input(sem::dense({m, a, shape.k}, bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat32));
  auto ids = f.input(sem::dense({m, 8}, c::DataType::kInt32));
  auto out = f.one(c::model_ops::grouped_float8_linear(),
                   {weight(f, experts, shape.n, shape.k), input, ids});
  f.outputs = {dynamic::visible(f, out, bf16)};
  for (int seed : {0, 1}) {
    auto x = dynamic::input_values(m * a, shape.k, seed, bf16);
    std::vector<std::int32_t> selected(m * 8);
    for (std::size_t i = 0; i < selected.size(); ++i)
      selected[i] = static_cast<std::int32_t>((i * 3 + seed * 2) % experts);
    if (seed) selected[0] = selected[1] = 2;
    f.replays.push_back({{dynamic::input_bytes(x, bf16), sem::bytes(selected)},
                         {reference(x, selected, shape.n, shape.k, shared, bf16)}});
  }
  dynamic::strict_output_check(f, bf16);
  return f;
}

inline sem::Fragment glm_moe(bool flash) {
  namespace moe = d5_moe_execution_fragments;
  namespace core = d5_moe_semantic_fragments;
  const auto &model = d5_moe_execution_cases::kCases[flash ? 10 : 7];
  const auto &up = fp8::kCases[flash ? 6 : 4];
  const auto &down = fp8::kCases[flash ? 7 : 5];
  const fp8::Case packed{up.profile, up.n * 2, up.k, false, "packed gate/up"};
  sem::Fragment f{.profile = model.profile, .region = 0x0b0607};
  auto input = f.input(sem::dense({1, model.hidden}));
  auto route = moe::routing(f, model, input);
  auto cast = f.one(c::Operation(c::Cast{.output_type = c::DataType::kBFloat16}), {input});
  auto slots = core::reshape(f, cast, {1, 1, model.hidden});
  auto projected = f.one(c::model_ops::grouped_float8_linear(),
      {weight(f, 9, packed.n, packed.k), slots, route.ids});
  const auto gated = [&](sem::ValueId value) {
    return f.one(c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kPackedHalves}), {value});
  };
  auto activation = gated(projected);
  auto routed = f.one(c::model_ops::grouped_float8_linear(),
      {weight(f, 9, down.n, down.k), activation, route.ids});
  auto routed_f32 = dynamic::visible(f, routed, true);
  auto combined = moe::sum(f, core::binary(f, c::BinaryKind::kMultiply, routed_f32,
      core::reshape(f, route.weights, {1, 8, 1})), 1);
  auto shared_up = f.one(dynamic::operation(false), {cast, dynamic::weight(f, packed)});
  auto shared_activation = gated(shared_up);
  auto shared = f.one(dynamic::operation(false), {shared_activation, dynamic::weight(f, down)});
  auto shared_f32 = dynamic::visible(f, shared, true);
  auto output = core::binary(f, c::BinaryKind::kAdd, core::reshape(f, combined, {1, model.hidden}), shared_f32);
  f.outputs = {moe::gathered_ids(f, model, route.ids), route.weights,
      dynamic::visible(f, projected, true), dynamic::visible(f, activation, true),
      routed_f32, combined, shared_f32, output};
  f.exact_outputs = {0};
  f.bfloat_chain_outputs = {2, 3, 4, 5, 6, 7};
  const auto activate = [&](std::span<const float> values, std::int64_t rows) {
    std::vector<float> result(rows * up.n);
    for (std::int64_t row = 0; row < rows; ++row) for (std::int64_t col = 0; col < up.n; ++col) {
      const double gate = values[row * packed.n + col], u = values[row * packed.n + up.n + col];
      result[row * up.n + col] = dynamic::bfloat(float(gate / (1 + std::exp(-gate)) * u));
    }
    return result;
  };
  for (int seed : {0, 1}) {
    auto x = moe::activation(model, seed);
    const auto selected = moe::route_reference(model, x);
    auto g = reference(x, selected.ids, packed.n, packed.k, true, true);
    auto a = activate(g, 8);
    auto r = reference(a, selected.ids, down.n, down.k, false, true);
    std::vector<float> sum(model.hidden);
    for (std::int64_t col = 0; col < model.hidden; ++col) {
      double value = 0;
      for (int slot = 0; slot < 8; ++slot) value += double(r[slot * model.hidden + col]) * selected.weights[slot];
      sum[col] = float(value);
    }
    auto sg = dynamic::reference(packed, x, true, false);
    auto sa = activate(sg, 1);
    auto sh = dynamic::reference(down, sa, true, false);
    std::vector<float> out(model.hidden);
    for (std::int64_t col = 0; col < model.hidden; ++col) out[col] = sum[col] + sh[col];
    f.replays.push_back({{sem::bytes(x)}, {selected.tags, selected.weights, std::move(g),
        std::move(a), std::move(r), std::move(sum), std::move(sh), std::move(out)}});
  }
  return f;
}

template <class Run> void projection_batch(Run run) {
  for (bool flash : {false, true}) {
    const auto &up = fp8::kCases[flash ? 6 : 4];
    const auto &down = fp8::kCases[flash ? 7 : 5];
    const fp8::Case packed{up.profile, up.n * 2, up.k, false, "grouped gate/up"};
    SCOPED_TRACE(up.profile->model_name);
    for (const auto &shape : {packed, down}) for (auto m : {1, 3})
      for (bool bf16 : {false, true}) for (bool shared : {false, true}) {
        SCOPED_TRACE(shape.role);
        SCOPED_TRACE(m);
        SCOPED_TRACE(bf16);
        SCOPED_TRACE(shared);
        run(projection(shape, m, bf16, shared));
      }
  }
  const fp8::Case tail{fp8::kCases[4].profile, 129, 128, false, "prefill/tail"};
  // >64 slots, >1024 slots, and generic full GLM expert counts. Same operation,
  // native program and independent sparse oracle; no testing-only provider.
  for (auto m : {17, 129}) for (auto experts : {9, 256, 288})
    for (bool bf16 : {false, true}) for (bool shared : {false, true})
      run(projection(tail, m, bf16, shared, experts));
}
} // namespace d5_grouped_fp8_fragments
