// Cold source policy and digest; no device, model payload or network needed.
#include "../../engine/models/qwen35/precision_profile.h"
#include "../../engine/models/qwen35/cache_plan.h"
#include "../../engine/inference/artifact_digest.h"

#include <cstdio>
#include <string>

namespace {
std::size_t checks = 0;
void require(bool value) {
  if (!value) throw std::runtime_error("precision/digest check failed");
  ++checks;
}
void digest(std::string_view input, std::string_view expected) {
  require(hyle::inference::artifact::sha256(std::as_bytes(std::span(input))) == expected);
}
template<class F> void rejects(F action) {
  try { action(); } catch (const std::runtime_error &) { ++checks; return; }
  throw std::runtime_error("invalid source precision was accepted");
}
}
int main(int argc, char **argv) {
  using namespace qwen35;
  using hyle::compute::DataType;
  digest("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  digest("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  digest(std::string(1000000, 'a'),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  prepared::Roles roles;
  require(PrecisionProfile{}.activation_type() == DataType::kBFloat16);
  require(PrecisionProfile::from_roles(roles) == PrecisionProfile{});
  prepared::Role source;
  source.dtype = "F16"; source.projection_codec = prepared::ProjectionCodec::kDense;
  roles.selected.push_back(source);
  require(PrecisionProfile::from_roles(roles).activation_type() == DataType::kBFloat16);
  roles.selected.front().projection_codec = prepared::ProjectionCodec::kNf4DoubleQuant;
  rejects([&] { (void)PrecisionProfile::from_roles(roles); });
  roles.nf4_policy.enabled = true;
  const auto original = PrecisionProfile::from_roles(roles);
  require(original.is_f16() && original.activation_type() == DataType::kFloat16);
  roles.selected.clear(); // A selected ordinary non-linear F16 subset keeps source policy.
  require(PrecisionProfile::from_roles(roles) == original);
  roles.float8_policy.enabled = true;
  rejects([&] { (void)PrecisionProfile::from_roles(roles); });
  roles.float8_policy.enabled = false; roles.ct_mixed_policy.kind = prepared::CtMixedPolicy::Kind::kChannelToken;
  rejects([&] { (void)PrecisionProfile::from_roles(roles); });
  roles.ct_mixed_policy.kind = prepared::CtMixedPolicy::Kind::kNone; roles.ct_asymmetric_policy.enabled = true;
  rejects([&] { (void)PrecisionProfile::from_roles(roles); });
  roles.ct_asymmetric_policy.enabled = false; roles.cache_calibration.push_back({});
  rejects([&] { (void)PrecisionProfile::from_roles(roles); });
  prepared::Roles e4;
  e4.ct_mixed_policy.kind = prepared::CtMixedPolicy::Kind::kChannelToken;
  e4.cache_policy = prepared::CachePolicy::kFloat8OriginalCalibration;
  e4.cache_calibration.push_back({});
  require(PrecisionProfile::from_roles(e4).activation_type() == DataType::kBFloat16);
  prepared::Roles gguf;
  gguf.representation = prepared::SourceRepresentation::kGgufConverted;
  prepared::Role encoded;
  encoded.name = "model.language_model.embed_tokens.weight";
  encoded.meaning = prepared::ParameterMeaning::kEmbeddingRows;
  encoded.dtype = "Q8_0";
  gguf.selected.push_back(encoded);
  encoded.name = "model.language_model.layers.0.mlp.gate_proj.weight";
  encoded.meaning = prepared::ParameterMeaning::kLinearWeight;
  gguf.selected.push_back(encoded);
  const auto f32 = PrecisionProfile::from_roles(gguf);
  require(f32.activation_type() == DataType::kFloat32 &&
      f32.cache_type() == DataType::kBFloat16 && f32.activation_bytes() == 4);
  gguf.selected.back().dtype = "Q4_K";
  gguf.selected.front().dtype = "Q6_K";
  require(PrecisionProfile::from_roles(gguf) == f32);
  encoded.name = "mtp.fc.weight";
  gguf.selected.push_back(encoded);
  require(PrecisionProfile::from_roles(gguf) == PrecisionProfile{});
  require(PrecisionProfile::from_roles(gguf, true) == f32);
  require(!compact_draft_rows(0, 4) && compact_draft_rows(2, 4) &&
      !compact_draft_rows(4, 4) && !compact_draft_rows(5, 4));
  gguf.selected.back().dtype = "BF16";
  require(PrecisionProfile::from_roles(gguf, true) == PrecisionProfile{});
  gguf.selected.pop_back();
  gguf.selected.back().dtype = "Q3_K";
  require(PrecisionProfile::from_roles(gguf) == PrecisionProfile{});
  gguf.selected.back().dtype = "Q4_K";
  gguf.representation = prepared::SourceRepresentation::kHfOriginal;
  require(PrecisionProfile::from_roles(gguf) == PrecisionProfile{});
  require(original.cache_type() == DataType::kFloat16 && original.activation_bytes() == 2);
  ModelGeometry g;
  g.hidden = 2048; g.intermediate = 6144; g.vocabulary = 248320;
  g.query_heads = 8; g.kv_heads = 2; g.key_depth = g.value_depth = 256; g.rotary = 64;
  g.gdn_key_heads = g.gdn_value_heads = 16;
  g.gdn_key_depth = g.gdn_value_depth = 128;
  g.conv_kernel = 4; g.context_limit = 32768; g.rope_theta = 10000; g.epsilon = 1e-6F;
  g.layers = "GA";
  const auto plan = [&](PrecisionProfile precision) {
    const auto history = std::size_t{3 * 6144} * precision.activation_bytes();
    return cache::plan(g, {{1024,1024},{}}, 128, 1, 16,
        history + 16 * 128 * 128 * 4, 1, 1, false, true, 2, 0, precision);
  };
  const auto bf16_plan = plan({}), f32_plan = plan(f32);
  require(f32_plan.hidden_row_bytes == 8192 && bf16_plan.hidden_row_bytes == 4096);
  require(f32_plan.row_bytes == bf16_plan.row_bytes && f32_plan.state_slots == 3);
  require(f32_plan.backing_bytes == bf16_plan.backing_bytes + 122880);
  // The early device-capability probe uses these BatchCapacity fields before
  // proposal assets are attached. Final resource accounting is rebuilt later.
  g.mtp_layers = 1;
  const auto probe = [&](PrecisionProfile precision) {
    return cache::plan(g, {{1024,1024},{1024,1024}}, 128, 1, 128,
        0, 1, 4, true, false, 3, 2, precision, 3);
  };
  const auto before_assets = probe({}), compact_assets = probe(f32);
  require(PrecisionProfile{}.cache_type() == f32.cache_type() &&
      before_assets.max_requests == compact_assets.max_requests &&
      before_assets.max_query_rows == compact_assets.max_query_rows &&
      before_assets.max_kv_rows == compact_assets.max_kv_rows &&
      before_assets.max_pages == compact_assets.max_pages &&
      before_assets.max_page_refs == compact_assets.max_page_refs);
  require(before_assets.hidden_row_bytes != compact_assets.hidden_row_bytes &&
      before_assets.backing_bytes != compact_assets.backing_bytes);
  e4.ct_mixed_policy.kind = prepared::CtMixedPolicy::Kind::kBlock128;
  require(PrecisionProfile::from_roles(e4).activation_type() == DataType::kBFloat16);
  e4.cache_policy = prepared::CachePolicy::kBFloat16OmitCalibration;
  require(PrecisionProfile::from_roles(e4).activation_type() == DataType::kBFloat16);
  // Optional fixed original-state inventory. This never opens a weight body.
  if (argc == 2) {
    std::ifstream manifest(argv[1]);
    require(bool(manifest));
    std::string line; std::size_t states = 0, bytes = 0;
    while (std::getline(manifest, line)) {
      std::istringstream row(line); prepared::Nf4State state;
      std::string offset, path, extra;
      require(bool(row >> state.logical_shape[0] >> state.logical_shape[1] >> offset
          >> state.original_json_sha256 >> std::quoted(path)) && !(row >> extra));
      const auto number = std::from_chars(offset.data(), offset.data() + offset.size(), state.offset_f32_bits, 16);
      require(number.ec == std::errc{} && number.ptr == offset.data() + offset.size());
      std::ifstream input(path, std::ios::binary);
      require(bool(input));
      const std::string raw{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      require(hyle::inference::artifact::sha256(std::as_bytes(std::span(raw))) == state.original_json_sha256);
      prepared::validate_nf4_state_values(raw, state);
      bytes += raw.size(); ++states;
    }
    require(states == 352 && bytes == 60553);
  } else require(argc == 1);
  std::printf("qwen precision/digest: %zu checks passed\n", checks);
}
