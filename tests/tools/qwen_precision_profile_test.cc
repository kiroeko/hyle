// Cold source policy and digest; no device, model payload or network needed.
#include "../../engine/models/qwen35/precision_profile.h"
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
