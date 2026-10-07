#pragma once
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS)
#include "cpu_c5_gguf_fixture.h"
#include "d5_kda_full_fragments.h"
#include "d5_deepseek_remaining_fragments.h"

// Actual GGUF convolution parameters enter an existing complete recurrent
// consumer. Only storage views/concatenation are added; its independent oracle,
// state feedback, changed input and numerical contract remain intact.
namespace cpu_c5_parameters {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
using sem::Fragment;

struct ConvCase {
  const char *id;
  unsigned profile;
  std::array<std::int64_t, 4> gguf_shape;
  unsigned rank, parts;
};
inline constexpr std::array convolutions{
    ConvCase{"gc3c587739220f12d", 0, {4, 6144, 0, 0}, 2, 1},
    ConvCase{"gf4c46eb70564ef43", 22, {4, 10240, 0, 0}, 2, 1},
    ConvCase{"g7651f3980ebb670e", 47, {4, 20480, 0, 0}, 2, 1},
    ConvCase{"gd57295091e4ad903", 56, {4, 1, 12288, 0}, 3, 3},
    ConvCase{"g31709c3de72cf996", 62, {4, 1, 12288, 1}, 4, 3},
    ConvCase{"g89aa3265ac4fdf68", 77, {4, 1, 8192, 0}, 3, 3}};

inline Fragment convolution(const ConvCase &item,
                            const c::ModelArtifactProfileDescription *identity) {
  const bool qwen = identity->model_repository.starts_with("Qwen/");
  auto f = qwen ? sem::gdn(identity, 4) : d5_kda_full_fragments::complete(identity);
  auto node = std::ranges::find_if(f.nodes, [](const auto &value) {
    return value.operation.template custom_as<c::model_ops::CausalConv1d>().has_value();
  });
  if (node == f.nodes.end()) throw std::runtime_error("missing recurrent convolution");
  const auto old = node->inputs[1];
  auto constant = std::ranges::find_if(f.constants, [&](auto value) {
    return value.ordinal() == old.ordinal();
  });
  if (constant == f.constants.end()) throw std::runtime_error("convolution weight is not constant");
  const auto index = static_cast<std::size_t>(constant - f.constants.begin());
  const auto expected = f.description(old);
  if (expected.data_type() != c::DataType::kFloat32 || expected.rank() != 2)
    throw std::runtime_error("unexpected recurrent convolution contract");
  const auto channels = expected.dimensions()[0] / item.parts;
  const auto taps = expected.dimensions()[1];
  std::vector<std::int64_t> physical;
  for (unsigned axis = item.rank; axis; --axis) physical.push_back(item.gguf_shape[axis - 1]);
  auto raw = c::TensorDesc::dense(c::DataType::kFloat32, physical).value();
  const auto part_bytes = raw.required_bytes().value();
  if (part_bytes * item.parts != f.constant_bytes[index].size())
    throw std::runtime_error("actual GGUF convolution size differs from family consumer");
  auto original_bytes = std::move(f.constant_bytes[index]);
  f.constants.erase(f.constants.begin() + index);
  f.constant_bytes.erase(f.constant_bytes.begin() + index);
  auto original_nodes = std::move(f.nodes);
  f.nodes.clear();
  std::vector<sem::ValueId> parts;
  for (unsigned part = 0; part < item.parts; ++part) {
    const auto first = original_bytes.begin() + part * part_bytes;
    auto stored = f.constant(raw, {first, first + part_bytes});
    const std::array normalized{channels, taps};
    auto view = c::View::reshape(c::TensorShape::create(normalized).value()).value();
    parts.push_back(f.one(c::Operation(view), {stored}));
  }
  auto bound = parts[0];
  for (unsigned part = 1; part < item.parts; ++part)
    bound = f.one(c::Operation(c::Concat{.axis = 0}), {bound, parts[part]});
  for (auto &value : original_nodes)
    for (auto &input : value.inputs)
      if (input.ordinal() == old.ordinal()) input = bound;
  for (auto &value : original_nodes) f.nodes.push_back(std::move(value));
  return f;
}

inline std::vector<unsigned> new_deepseek_profiles() {
  std::vector<unsigned> result;
  for (unsigned index = 0; index < std::size(cpu_c5_inputs::profiles); ++index) {
    const auto &value = cpu_c5_inputs::profiles[index];
    if (value.family >= 6 && (std::string_view(value.variant) == "UD-Q4_K_XL" ||
                             std::string_view(value.variant) == "UD-Q8_K_XL"))
      result.push_back(index);
  }
  if (result.size() != 4) throw std::runtime_error("missing fixed DeepSeek GGUF profiles");
  return result;
}

inline d5_moe_execution_cases::Case new_deepseek_case(
    const c::ModelArtifactProfileDescription *identity) {
  using Storage = d5_moe_execution_cases::Storage;
  const bool pro = identity->model_name == "DeepSeek-V4-Pro-0813";
  auto result = cpu_c4_model_branches::models[pro ? 0 : 1].main_case();
  result.profile = identity;
  result.gguf = true;
  result.router_bf16 = true;
  result.gate = result.up = result.down = Storage::kMxfp4;
  const auto plain = identity->variant == "UD-Q4_K_XL" ? Storage::kQ8_0 : Storage::kBFloat16;
  result.shared_gate = result.shared_up = result.shared_down = plain;
  result.latent_storage = Storage::kBFloat16;
  return result;
}

template <class Run>
void new_deepseek_hyper_head(const d5_moe_execution_cases::Case &model, Run run) {
  auto fragment = d5_deepseek_remaining_fragments::hyper_connection(model);
  std::vector<std::vector<float>> observed, reference;
  for (const auto &replay : fragment.replays) reference.push_back(replay.outputs.back());
  fragment.observe = [&](std::size_t port, std::span<const float> values) {
    if (port == 6) observed.emplace_back(values.begin(), values.end());
  };
  run(std::move(fragment));
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  ASSERT_EQ(observed.size(), reference.size());
  const bool packed = model.shared_gate == d5_moe_execution_cases::Storage::kQ8_0;
  const cpu_c5_inputs::Case item{"C5NewDeepSeekHead", cpu_c5_inputs::Kind::kHead,
      packed ? GGML_TYPE_Q8_0 : GGML_TYPE_BF16, c::GgufQuantizedType::kQ8_0, packed,
      model.hidden, 129280, 1, 0, 0};
  const cpu_c5_gguf_fixture::Bank bank(item);
  // Keep every vocabulary row. Only resident head rows are sharded, as in F1.
  for (std::int64_t first = 0; first < item.n; first += 4096) {
    const auto count = std::min<std::int64_t>(4096, item.n - first);
    auto head = cpu_c5_gguf_fixture::projection(item, bank, model.profile, count, first);
    head.replays.clear();
    for (std::size_t replay = 0; replay < observed.size(); ++replay) {
      // mHC already passed its independent complete oracle above. Preserve its
      // actual feedback into the next program, and test that program against
      // the same input: legal F32 reduction differences can cross a BF16 RNE
      // boundary, so a different reference activation is not a head oracle.
      const auto dot = bank.dots(observed[replay]);
      std::vector<float> expected(count);
      for (std::int64_t row = 0; row < count; ++row)
        expected[row] = dot[(first + row) % cpu_c5_gguf_fixture::prototypes];
      head.replays.push_back({{sem::bytes(observed[replay])}, {std::move(expected)}});
    }
    run(std::move(head));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}
} // namespace cpu_c5_parameters
#endif
