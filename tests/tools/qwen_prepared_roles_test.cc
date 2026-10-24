// Standard-library-only metadata regression: no weight body, SDK or device IO.
#include "../../engine/models/qwen35/prepared_roles.h"
#include "qwen_geometry_fixture.h"

#include <cstdio>
#include <array>
#include <utility>

namespace {
namespace p = qwen35::prepared;
namespace fs = std::filesystem;
using Inventory = std::map<std::string, p::Role>;
const std::string identity(64, 'a');
const auto geometry = qwen_test::q35();
const auto gguf_geometry = qwen_test::q35(true);
std::size_t checks = 0;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
  ++checks;
}
template<class Read>
void rejects(Read &&read, std::string_view diagnostic) {
  try { read(); }
  catch (const std::runtime_error &error) {
    if (std::string_view(error.what()).find(diagnostic) != std::string_view::npos) {
      ++checks;
      return;
    }
    throw std::runtime_error(std::string("unexpected parser diagnostic: ") + error.what());
  }
  throw std::runtime_error("invalid prepared roles were accepted");
}
void write(const fs::path &path, std::string_view text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
  if (!output) throw std::runtime_error("cannot write prepared metadata fixture");
}
std::string row(const p::Role &role) {
  std::ostringstream output;
  output << role.name << '\t' << role.dtype << '\t' << role.shape.size();
  for (const auto dim : role.shape) output << '\t' << dim;
  output << '\t' << std::quoted(role.source.string()) << '\t' << role.offset << '\t' << role.bytes << '\n';
  return output.str();
}
std::string manifest(const Inventory &roles) {
  std::string text = std::string(p::role_signature) + '\n' + identity + '\n' + qwen_test::geometry_line(geometry) + '\n';
  for (const auto &[name, role] : roles) {
    (void)name;
    text += row(role);
  }
  return text;
}
void add(Inventory &roles, std::string name, std::string dtype, std::vector<std::int64_t> shape) {
  // Sources deliberately do not exist. A metadata parser must not fetch or load
  // payloads; the loader separately verifies each actual file and byte range.
  p::Role role{name, std::move(dtype), std::move(shape),
      name.starts_with("mtp.") ? "cached MTP shard.bin" : "text shard.bin", 0, 0};
  role.bytes = p::physical_bytes(role.dtype, role.shape);
  role.offset = 4096 + static_cast<std::uint64_t>(roles.size()) * 64;
  roles.emplace(std::move(name), std::move(role));
}
Inventory dense(bool mtp = true) {
  Inventory roles;
  for (const auto &[name, parameter] : p::parameter_inventory(mtp, geometry))
    add(roles, name, "BF16", parameter.shape);
  return roles;
}
Inventory quantized(bool nvfp4) {
  Inventory roles;
  for (const auto &[name, parameter] : p::parameter_inventory(true, geometry)) {
    const auto base = name.ends_with(".weight") ? name.substr(0, name.size() - 7) : name;
    const bool gptq_excluded = base.ends_with(".in_proj_a") || base.ends_with(".in_proj_b");
    if (!parameter.linear || name.starts_with("mtp.") || (!nvfp4 && gptq_excluded)) {
      add(roles, name, "BF16", parameter.shape);
      continue;
    }
    const auto n = parameter.shape[0], k = parameter.shape[1];
    if (nvfp4) {
      add(roles, name, "U8", {n, k / 2});
      add(roles, base + ".weight_scale", "F8_E4M3", {n, k / 16});
      add(roles, base + ".weight_scale_2", "F32", {});
      add(roles, base + ".input_scale", "F32", {});
    } else {
      add(roles, base + ".qweight", "I32", {k / 8, n});
      add(roles, base + ".qzeros", "I32", {k / 32, n / 8});
      add(roles, base + ".scales", "F16", {k / 32, n});
      add(roles, base + ".g_idx", "I32", {k});
    }
  }
  return roles;
}
Inventory ct_quantized() {
  Inventory roles;
  for (const auto &[name, parameter] : p::parameter_inventory(true, geometry)) {
    const auto base = name.ends_with(".weight") ? name.substr(0, name.size() - 7) : name;
    const bool excluded = base.ends_with(".in_proj_a") || base.ends_with(".in_proj_b") || name == "mtp.fc.weight";
    if (!parameter.linear || excluded) {
      add(roles, name, "BF16", parameter.shape);
      continue;
    }
    const auto n = parameter.shape[0], k = parameter.shape[1];
    add(roles, base + ".weight_packed", "I32", {n, k / 8});
    add(roles, base + ".weight_scale", "BF16", {n, k / 32});
    add(roles, base + ".weight_shape", "I64", {2});
  }
  return roles;
}
std::uint64_t metadata_fingerprint(const Inventory &roles) {
  std::string text;
  for (const auto &[name, role] : roles) {
    text += name + '\t' + role.dtype + '\t';
    for (std::size_t i = 0; i < role.shape.size(); ++i) {
      if (i) text += ',';
      text += std::to_string(role.shape[i]);
    }
    text += '\t' + std::to_string(role.bytes) + '\n';
  }
  std::uint64_t hash = 14695981039346656037ull;
  for (const auto ch : text) hash = (hash ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
  return hash;
}
void fp8_blocks_and_policy(const fs::path &directory) {
  auto roles = dense();
  const std::array names{"model.language_model.layers.0.mlp.gate_proj.weight", "mtp.layers.0.self_attn.k_proj.weight"};
  for (const auto *name : names) {
    auto &w = roles.at(name); w.dtype = "F8_E4M3"; w.bytes /= 2;
    add(roles, w.name.substr(0, w.name.size() - 7) + ".weight_scale_inv", "BF16",
        {(w.shape[0] + 127) / 128, w.shape[1] / 128});
  }
  const auto path = directory / "fp8.tsv";
  auto text = manifest(roles);
  const auto policy_at = text.find('\n', text.find('\n') + 1);
  const auto line_end = text.find('\n', policy_at + 1);
  text.insert(line_end + 1, std::string(p::fp8_policy_line) + '\n');
  write(path, text);
  const auto all = p::read_roles(path, true), selected = p::read_roles(path, false);
  require(all.selected.size() == 337 && selected.selected.size() == 321 && all.float8_policy.enabled,
      "FP8 selected physical inventory or policy differs");
  require(std::ranges::count_if(all.selected, [](const auto &r) { return r.projection_codec == p::ProjectionCodec::kFloat8Block128; }) == 2,
      "FP8 block128 codec must be derived from complete physical groups");
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "explicit reader policy");
  auto invalid = roles; invalid.erase("mtp.layers.0.self_attn.k_proj.weight_scale_inv");
  rejects([&] { (void)p::select_roles(invalid, identity, false, geometry, {true}); }, "missing prepared role");
  invalid = roles; invalid.at("mtp.layers.0.self_attn.k_proj.weight_scale_inv").dtype = "F32";
  invalid.at("mtp.layers.0.self_attn.k_proj.weight_scale_inv").bytes *= 2;
  rejects([&] { (void)p::select_roles(invalid, identity, false, geometry, {true}); }, "type/shape differs");
  rejects([&] { (void)p::select_roles(dense(), identity, true, geometry, {true}); }, "no block128 weights");
  auto wrong = text; wrong.replace(wrong.find("divide"), 6, "power2"); write(path, wrong);
  rejects([&] { (void)p::read_roles(path, true); }, "reader policy");
  wrong = text; wrong.insert(line_end + 1, std::string(p::fp8_policy_line) + '\n'); write(path, wrong);
  rejects([&] { (void)p::read_roles(path, true); }, "reader policy");
}
void ct_mixed_and_cache_policy(const fs::path &directory) {
  auto roles = dense();
  const std::string channel = "model.language_model.layers.3.self_attn.k_proj";
  auto &w = roles.at(channel + ".weight"); w.dtype = "F8_E4M3"; w.bytes /= 2;
  add(roles, channel + ".weight_scale", "BF16", {w.shape[0], 1});
  const std::string local = "model.language_model.layers.0.mlp.gate_proj";
  const auto shape = roles.at(local + ".weight").shape; roles.erase(local + ".weight");
  add(roles, local + ".weight_packed", "U8", {shape[0], shape[1] / 2});
  add(roles, local + ".weight_scale", "F8_E4M3", {shape[0], shape[1] / 16});
  add(roles, local + ".weight_global_scale", "F32", {1});
  add(roles, local + ".input_global_scale", "F32", {1});
  const std::string cache = "model.language_model.layers.3.self_attn.";
  add(roles, cache + "k_scale", "BF16", {1}); add(roles, cache + "v_scale", "BF16", {1});
  const auto all = p::select_roles(roles, identity, true, geometry, {}, {true}, {true});
  const auto text = p::select_roles(roles, identity, false, geometry, {}, {true}, {true});
  require(all.selected.size() == 339 && text.selected.size() == 324 && all.omitted_cache_calibration.size() == 2,
      "CT mixed selected roles must exclude only explicit BF16 cache calibration");
  require(std::ranges::count(all.selected, p::ProjectionCodec::kCtFloat8ChannelToken, &p::Role::projection_codec) == 1 &&
      std::ranges::count(all.selected, p::ProjectionCodec::kCtLocalNvfp4, &p::Role::projection_codec) == 1,
      "CT mixed codecs must follow the original complete groups");
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "fixed reader policy");
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry, {}, {true}); }, "explicit CT BF16 KV policy");
  auto invalid = roles; invalid.erase(local + ".input_global_scale");
  rejects([&] { (void)p::select_roles(invalid, identity, true, geometry, {}, {true}, {true}); }, "missing prepared role");
  invalid = roles; invalid.at(channel + ".weight_scale").shape[1] = 2; invalid.at(channel + ".weight_scale").bytes *= 2;
  rejects([&] { (void)p::select_roles(invalid, identity, true, geometry, {}, {true}, {true}); }, "type/shape differs");
  invalid = roles; add(invalid, "model.language_model.layers.0.self_attn.k_scale", "BF16", {1});
  rejects([&] { (void)p::select_roles(invalid, identity, true, geometry, {}, {true}, {true}); }, "unclaimed");
  invalid = roles; invalid.at(cache + "v_scale").shape = {2}; invalid.at(cache + "v_scale").bytes = 4;
  rejects([&] { (void)p::select_roles(invalid, identity, true, geometry, {}, {true}, {true}); }, "calibration pair");
  auto serialized = manifest(roles); const auto at = serialized.find('\n', serialized.find('\n', serialized.find('\n') + 1) + 1);
  serialized.insert(at + 1, std::string(p::ct_mixed_policy_line) + '\n' + std::string(p::bf16_cache_policy_line) + '\n');
  const auto path = directory / "ct-mixed.tsv"; write(path, serialized);
  require(p::read_roles(path, true).omitted_cache_calibration.size() == 2, "explicit CT/cache policy must roundtrip");
}
void check_selected(const p::Roles &result, std::size_t count, std::size_t bytes, bool mtp) {
  if (result.identity != identity || result.selected.size() != count || result.bytes != bytes)
    throw std::runtime_error("selected role count/bytes differs: " + std::to_string(result.selected.size()) +
        "/" + std::to_string(result.bytes) + ", expected " + std::to_string(count) + "/" + std::to_string(bytes));
  ++checks;
  require(mtp || std::ranges::none_of(result.selected, [](const auto &r) { return r.name.starts_with("mtp."); }),
      "ordinary execution selected MTP payloads");
}
void full_inventory_and_selection(const fs::path &directory) {
  const auto path = directory / "complete.tsv";
  const auto all = dense();
  require(all.size() == 335 && dense(false).size() == 320, "canonical dense inventory count differs");
  write(path, manifest(all));
  check_selected(p::read_roles(path, true), 335, 3885306496ull, true);
  check_selected(p::read_roles(path, false), 320, 3763650176ull, false);
  const auto first = p::read_roles(path, true).selected.front();
  require(first.source == directory / all.begin()->second.source && first.offset == all.begin()->second.offset,
      "quoted relative source or original source offset was lost");
  write(path, manifest(dense(false)));
  check_selected(p::read_roles(path, false), 320, 3763650176ull, false);
  rejects([&] { (void)p::read_roles(path, true); }, "MTP requires");

  auto f32 = all;
  for (auto &[name, role] : f32) {
    if (name.ends_with(".A_log") || name.ends_with(".dt_bias") || name.ends_with(".linear_attn.norm.weight")) {
      role.dtype = "F32";
      role.bytes *= 2;
    }
  }
  check_selected(p::select_roles(f32, identity, true, geometry), 335, 3885312256ull, true);

  // Freeze the complete physical metadata independently against the two actual
  // artifact plans, not only the inventory helper that builds these fixtures.
  // Q01: Vishva007/Qwen3.5-2B-W4A16-AutoRound-GPTQ @8ec79a3b08636bc68fd40d69df80e97ff1263995
  // Q03: AxionML/Qwen3.5-2B-NVFP4 @6f2ad4febe92be529de499b08f0020f5ff06bf25
  // FNV-1a covers sorted name, dtype, shape and byte size of every physical role.
  for (const bool nvfp4 : {false, true}) {
    const auto roles = quantized(nvfp4);
    require(metadata_fingerprint(roles) == (nvfp4 ? 0x76432bb180990b15ull : 0x69e2f17d3339b8b5ull),
        "physical metadata differs from the pinned original artifact");
    const auto quant_path = directory / (nvfp4 ? "Q03.tsv" : "Q01.tsv");
    write(quant_path, manifest(roles));
    check_selected(p::read_roles(quant_path, true), nvfp4 ? 893 : 785,
        nvfp4 ? 1912027216ull : 1936773760ull, true);
    check_selected(p::read_roles(quant_path, false), nvfp4 ? 878 : 770,
        nvfp4 ? 1790370896ull : 1815117440ull, false);
  }
  auto scalars = quantized(true);
  for (auto &[name, role] : scalars)
    if (name.ends_with(".input_scale") || name.ends_with(".weight_scale_2")) role.shape = {1};
  check_selected(p::select_roles(scalars, identity, true, geometry), 893, 1912027216ull, true);

  std::string crlf;
  for (const auto ch : manifest(all)) {
    if (ch == '\n') crlf += '\r';
    crlf += ch;
  }
  write(path, crlf);
  check_selected(p::read_roles(path, true), 335, 3885306496ull, true);
}
void complete_validation_also_applies_when_mtp_is_off(const fs::path &directory) {
  const auto path = directory / "invalid-inventory.tsv";
  const auto check = [&](const Inventory &roles, std::string_view diagnostic) {
    write(path, manifest(roles));
    for (const bool mtp : {false, true}) rejects([&] { (void)p::read_roles(path, mtp); }, diagnostic);
  };
  auto roles = dense();
  roles.erase("model.language_model.embed_tokens.weight");
  check(roles, "missing prepared role");
  roles = dense(); roles.erase("mtp.norm.weight");
  check(roles, "missing prepared role");
  roles = dense(); roles.at("mtp.norm.weight").shape = {1024, 2};
  check(roles, "type/shape differs");
  roles = dense(); roles.at("mtp.norm.weight").dtype = "F16";
  check(roles, "type/shape differs");
  roles = dense(); roles.at("mtp.norm.weight").offset = UINT64_MAX;
  check(roles, "byte extent");
  roles = dense(); add(roles, "undeclared.weight", "BF16", {1});
  check(roles, "unclaimed prepared physical role");
  roles = dense(); add(roles, "mtp.undeclared.weight", "BF16", {1});
  check(roles, "unclaimed prepared physical role");
  auto text = manifest(dense()); text += row(dense().begin()->second);
  write(path, text);
  rejects([&] { (void)p::read_roles(path, false); }, "duplicate prepared role");
}
void quantized_auxiliaries_are_complete_and_typed(const fs::path &directory) {
  const auto path = directory / "invalid-quant.tsv";
  const std::string base = "model.language_model.layers.0.mlp.gate_proj";
  for (const auto *suffix : {".qweight", ".qzeros", ".scales", ".g_idx"}) {
    auto roles = quantized(false); roles.erase(base + suffix);
    write(path, manifest(roles));
    rejects([&] { (void)p::read_roles(path, false); }, suffix == std::string_view(".scales") ? "group geometry" : "missing prepared role");
  }
  for (const auto *suffix : {".weight", ".weight_scale", ".weight_scale_2", ".input_scale"}) {
    auto roles = quantized(true); roles.erase(base + suffix);
    write(path, manifest(roles));
    const auto diagnostic = suffix == std::string_view(".weight") ? "missing prepared role" :
        suffix == std::string_view(".weight_scale") ? "type/shape differs" : "NVFP4 scalar";
    rejects([&] { (void)p::read_roles(path, false); }, diagnostic);
  }
  auto roles = quantized(false);
  roles.at(base + ".scales").shape = {63, 6144};
  roles.at(base + ".scales").bytes = p::physical_bytes("F16", {63, 6144});
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "group geometry");
  roles = quantized(false); roles.at(base + ".g_idx").dtype = "F32";
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "type/shape differs");
  roles = quantized(false); add(roles, base + ".weight", "BF16", {6144, 2048});
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "unclaimed prepared physical role");
  roles = quantized(true); roles.at(base + ".weight_scale").dtype = "U8";
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "type/shape differs");
  roles = quantized(true); roles.at(base + ".input_scale").shape = {1, 1};
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "NVFP4 scalar");
  roles = quantized(true); roles.at(base + ".input_scale").dtype = "I32";
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "type/shape differs");
  roles = quantized(true); add(roles, base + ".g_idx", "I32", {2048});
  rejects([&] { (void)p::select_roles(roles, identity, true, geometry); }, "unclaimed prepared physical role");
}
void packed_axes_are_complete_before_weight_io() {
  const auto check = [&](qwen35::ModelGeometry model, std::string base, bool nvfp4,
                         std::int64_t groups) {
    Inventory roles;
    const auto inventory = p::parameter_inventory(model.mtp_layers != 0, model);
    for (const auto &[name, parameter] : inventory) add(roles, name, "BF16", parameter.shape);
    const auto shape = inventory.at(base + ".weight").shape;
    const auto n = shape[0], k = shape[1];
    roles.erase(base + ".weight");
    if (nvfp4) {
      add(roles, base + ".weight", "U8", {n, k / 2});
      add(roles, base + ".weight_scale", "F8_E4M3", {n, k / 16});
      add(roles, base + ".weight_scale_2", "F32", {});
      add(roles, base + ".input_scale", "F32", {});
    } else {
      add(roles, base + ".qweight", "I32", {k / 8, n});
      add(roles, base + ".qzeros", "I32", {groups, n / 8});
      add(roles, base + ".scales", "F16", {groups, n});
      add(roles, base + ".g_idx", "I32", {k});
    }
    for (const bool mtp : {false, true}) {
      if (mtp && !model.mtp_layers) continue;
      rejects([&] { (void)p::select_roles(roles, identity, mtp, model); },
          nvfp4 ? "complete K groups" : "GPTQ group geometry");
    }
  };
  auto model = geometry; model.hidden = 2049;
  check(model, "mtp.fc", false, 1); // Truncated K/8 and N/8 cannot identify original GPTQ.
  model = geometry; model.intermediate = 6145;
  check(model, "model.language_model.layers.0.mlp.gate_proj", false, 64);
  model = geometry; model.hidden = 2052;
  check(model, "mtp.fc", true, 0); // Even packed K still needs complete group16, including MTP-off.
  model = geometry; model.hidden = std::int64_t{1} << 32; model.mtp_layers = 0;
  check(model, "model.language_model.layers.0.mlp.gate_proj", false, 1); // No truncated U32 group size.
}
void ct_storage_is_distinct_and_original_shape_is_checked(const fs::path &directory) {
  const auto all = ct_quantized();
  // cyankiwi/Qwen3.5-2B-AWQ-4bit @718fd9b52c5ca16f1f856f6c4e4209129802a6a2
  // Full117112-byte original header, SHA25643cad8e5...4b2e91d99. This is CT
  // signed symmetric INT4, not the different legacy AWQ/GPTQ/NVFP4 layouts.
  require(metadata_fingerprint(all) == 0x38c2b43dd19168ecull,
      "CT physical metadata differs from the pinned original artifact");
  const auto path = directory / "CT.tsv";
  write(path, manifest(all));
  const auto complete = p::read_roles(path, true), text = p::read_roles(path, false);
  check_selected(complete, 649, 1838357584ull, true);
  check_selected(text, 620, 1792067552ull, false);
  std::size_t ct = 0;
  for (const auto &role : complete.selected) {
    if (role.name.ends_with(".weight_packed")) {
      require(role.projection_codec == p::ProjectionCodec::kCtSymmetric4 && role.projection_group_size == 32,
          "CT storage/group must be inferred from the complete original role geometry");
      ++ct;
    }
  }
  require(ct == 157, "all150 main and7 MTP CT projections must be present");
  const std::string base = "mtp.layers.0.self_attn.q_proj";
  for (const auto *suffix : {".weight_packed", ".weight_scale", ".weight_shape"}) {
    auto invalid = all; invalid.erase(base + suffix);
    for (const bool mtp : {false, true})
      rejects([&] { (void)p::select_roles(invalid, identity, mtp, geometry); },
          suffix == std::string_view(".weight_scale") ? "CT symmetric group geometry" : "missing prepared role");
  }
  for (const auto kind : {0,1,2,3}) {
    auto invalid = all;
    auto &codes = invalid.at(base + ".weight_packed");
    auto &scales = invalid.at(base + ".weight_scale");
    auto &shape = invalid.at(base + ".weight_shape");
    if (kind == 0) { codes.shape={256,4096}; }
    if (kind == 1) { scales.shape={4096,63}; scales.bytes=p::physical_bytes(scales.dtype,scales.shape); }
    if (kind == 2) { shape.dtype="I32"; shape.bytes=8; }
    if (kind == 3) { add(invalid, base + ".weight", "BF16", {4096,2048}); }
    rejects([&] { (void)p::select_roles(invalid, identity, false, geometry); },
        kind == 1 ? "CT symmetric group geometry" : kind == 3 ? "unclaimed" : "type/shape differs");
  }
  for (const auto *dtype : {"F16", "BF16", "F32"}) {
    auto variable = all;
    auto &scale = variable.at(base + ".weight_scale");
    scale.dtype=dtype;scale.shape={4096,32};scale.bytes=p::physical_bytes(dtype,scale.shape);
    const auto selected=p::select_roles(variable,identity,true, geometry);
    const auto found=std::ranges::find_if(selected.selected,[&](const auto &r){return r.name==base+".weight_packed";});
    require(found!=selected.selected.end() && found->projection_group_size==64,
        "CT group size must come from actual K/scale groups, not a model constant");
  }
  const auto codes=std::ranges::find_if(complete.selected,[&](const auto &r){return r.name==base+".weight_packed";});
  std::array<std::byte,16> raw{};
  const auto encode=[&](std::size_t first,std::uint64_t value){
    for(std::size_t i=0;i<8;++i) raw[first+i]=static_cast<std::byte>((value>>(8*i))&255);
  };
  encode(0,4096);encode(8,2048);
  p::validate_ct_shape_values(*codes,raw);++checks;
  raw[8]^=std::byte{1};
  rejects([&]{p::validate_ct_shape_values(*codes,raw);},"weight_shape values differ");
  rejects([&]{p::validate_ct_shape_values(*codes,std::span(raw).first(15));},"shape value input");
  encode(8,UINT64_MAX);
  rejects([&]{p::validate_ct_shape_values(*codes,raw);},"weight_shape values differ");
}
void malformed_input_and_shared_entrypoint_are_bounded(const fs::path &directory) {
  const auto path = directory / "malformed.tsv";
  const auto all = dense();
  const auto prefix = std::string(p::role_signature) + '\n' + identity + '\n' + qwen_test::geometry_line(geometry) + '\n';
  const auto invalid_row = [&](std::string_view line, std::string_view diagnostic) {
    write(path, prefix + std::string(line) + '\n');
    rejects([&] { (void)p::read_roles(path, false); }, diagnostic);
  };
  write(path, "wrong\n" + identity + '\n');
  rejects([&] { (void)p::read_roles(path, false); }, "signature");
  for (const auto &bad : {std::string(63, 'a'), std::string(64, 'A'), std::string(64, 'g')}) {
    write(path, std::string(p::role_signature) + '\n' + bad + '\n');
    rejects([&] { (void)p::read_roles(path, false); }, "identity");
  }
  invalid_row("x BF16 4", "rank exceeds");
  invalid_row("x BF16 1 0", "extent");
  invalid_row("x BF16 1 -1", "integer");
  invalid_row("x BF16 1 9223372036854775808", "extent");
  invalid_row("x F32 2 9223372036854775807 2", "extent");
  invalid_row("x BF16 18446744073709551616", "integer");
  invalid_row("x BF16 +1", "integer");
  invalid_row("x F64 0", "unsupported prepared role dtype");
  invalid_row("x BF16 0 \"x.bin\" 18446744073709551615 2", "byte extent");
  invalid_row("x BF16 0 \"x.bin\" 0 4", "byte extent");
  invalid_row("x BF16 0 \"x.bin\" 0 2 extra", "byte extent");
  invalid_row("x BF16 0 x.bin 0 2", "quoted source file");
  invalid_row("x BF16 0 \"\" 0 2", "quoted source file");
  invalid_row("x BF16 0 \"unterminated.bin 0 2", "quoted source file");
  invalid_row(std::string("x BF16 0 \"bad") + '\0' + "name\" 0 2", "quoted source file");
  invalid_row("x BF16 0 \"x.bin\" -1 2", "integer");
  invalid_row("x BF16 0 \"x.bin\" 0", "missing prepared role integer");
  rejects([&] { (void)p::read_roles(directory / "does-not-exist.tsv", false); }, "signature");

  auto roles = all; roles.begin()->second.bytes += 2;
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "byte extent");
  roles = all; roles.begin()->second.offset = UINT64_MAX;
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "byte extent");
  roles = all; roles.begin()->second.name = "different.key";
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "name/key differs");
  roles = all; roles.begin()->second.source.clear();
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "source file");
  roles = all; roles.begin()->second.shape = {-1};
  rejects([&] { (void)p::select_roles(roles, identity, false, geometry); }, "extent");
  rejects([&] { (void)p::select_roles(all, "bad", false, geometry); }, "identity");

  // Reject oversized local metadata while reading, before constructing a
  // filesystem path or accumulating an unbounded physical-role inventory.
  write(path, prefix + std::string(p::maximum_manifest_line_bytes + 1, 'x'));
  rejects([&] { (void)p::read_roles(path, false); }, "line length limit");
  write(path, prefix + "x BF16 0 \"" + std::string(p::maximum_manifest_path_bytes + 1, 'x') + "\" 0 2\n");
  rejects([&] { (void)p::read_roles(path, false); }, "source path length limit");
  std::string oversized = prefix;
  for (std::size_t i = 0; i <= (p::parameter_inventory(true, geometry).size()*p::maximum_physical_components); ++i)
    oversized += "extra" + std::to_string(i) + " BF16 0 \"never-opened.bin\" 0 2\n";
  write(path, oversized);
  rejects([&] { (void)p::read_roles(path, false); }, "role count exceeds model limit");
}
void nf4_original_state_and_six_role_contract(const fs::path &directory) {
  // Original171B: unsloth/Qwen3.8-27B-unsloth-bnb-4bit
  // @8aa5f05d26b7205477066e1449e0af13f762a299/model.safetensors,
  // bytes15264695436..15264695606; SHA2569210c25c...ed9d6c1.
  const auto fixture = fs::path(__FILE__).parent_path().parent_path() / "tooling/fixtures/qwen_nf4_quant_state.json";
  std::ifstream original(fixture, std::ios::binary);
  const std::string raw((std::istreambuf_iterator<char>(original)), {});
  auto actual = p::parse_nf4_state(raw);
  require(raw.size() == 171 && actual.logical_shape == std::array<std::int64_t,2>{1024,5120} &&
      actual.offset_f32_bits == 0x3d2fd550 && actual.decode_dtype == "F16" && actual.compute_dtype == "F16",
      "original NF4 JSON must preserve source decode type, original shape and F32 offset bits");
  p::validate_nf4_state_values(raw, actual);
  actual.logical_shape[0] = 3;
  rejects([&]{p::validate_nf4_state_values(raw,actual);},"values differ");
  for (const auto &bad : {raw + std::string(512,' '), raw + "x", std::string("{}"),
      std::string("{\"quant_type\":\"nf4\",\"quant_type\":\"nf4\"}")})
    rejects([&]{(void)p::parse_nf4_state(bad);},"invalid original NF4");
  for (const auto replacement : {"float32", "bfloat16"}) {
    auto bad = raw; bad.replace(bad.find("float16"),7,replacement);
    rejects([&]{(void)p::parse_nf4_state(bad);},"invalid original NF4");
  }
  auto all = dense();
  for (auto &[name,role] : all) { (void)name; role.dtype = "F16"; }
  const std::array<std::string,2> names{"model.language_model.layers.3.self_attn.k_proj.weight",
                                      "mtp.layers.0.self_attn.k_proj.weight"};
  std::map<std::string,p::Nf4State> states;
  for (const auto &name : names) {
    const auto shape = all.at(name).shape; all.erase(name);
    const auto count = shape[0]*shape[1], groups = count/64;
    add(all,name,"U8",{count/2,1}); add(all,name+".absmax","U8",{groups});
    add(all,name+".nested_absmax","F32",{groups/256+(groups%256!=0)});
    add(all,name+".nested_quant_map","F32",{256}); add(all,name+".quant_map","F32",{16});
    add(all,name+std::string(p::nf4_state_suffix),"U8",{171});
    states.emplace(name,p::Nf4State{{shape[0],shape[1]},0x3d2fd550,std::string(64,'b')});
  }
  const auto select = [&](const Inventory &inventory,const auto &metadata,bool mtp=false) {
    return p::select_roles(inventory,identity,mtp,geometry,{},{},{},{true},metadata);
  };
  const auto text=select(all,states),full=select(all,states,true);
  require(text.nf4_policy.enabled && text.selected.size()==325 && full.selected.size()==345,
      "NF4 ordinary/MTP selection must validate both complete six-role groups");
  const auto codes=std::ranges::find_if(text.selected,[&](const auto &r){return r.name==names[0];});
  require(codes!=text.selected.end() && codes->projection_codec==p::ProjectionCodec::kNf4DoubleQuant &&
      codes->projection_group_size==64 && codes->nf4_state.logical_shape==std::array<std::int64_t,2>{512,2048},
      "NF4 principal must keep source physical axes separate from logical decode shape");
  for (const auto suffix : {"", ".absmax", ".nested_absmax", ".nested_quant_map", ".quant_map", ".quant_state.bitsandbytes__nf4"}) {
    auto bad=all;bad.erase(names[1]+suffix);
    rejects([&]{(void)select(bad,states);},suffix==std::string_view(".quant_state.bitsandbytes__nf4")?"byte limit":"missing prepared role");
  }
  auto bad=all;bad.at("model.language_model.norm.weight").dtype="BF16";
  rejects([&]{(void)select(bad,states);},"type/shape differs");
  auto fewer=states;fewer.erase(names[1]);
  rejects([&]{(void)select(all,fewer);},"per-group original state");
  auto wrong_shape=states;wrong_shape.at(names[1]).logical_shape[0]=3;
  rejects([&]{(void)select(all,wrong_shape);},"typed logical shape differs");
  auto extra=states;extra.emplace("unknown.weight",actual);
  rejects([&]{(void)select(all,extra);},"per-group state inventory");
  const auto path=directory/"nf4.tsv";
  auto serialized=std::string(p::role_signature)+'\n'+identity+'\n'+qwen_test::geometry_line(geometry)+'\n'+std::string(p::nf4_policy_line)+'\n';
  for (const auto &[name,state]:states) serialized+="nf4_state\t"+name+'\t'+state.original_json_sha256+"\t3d2fd550\n";
  for (const auto &[name,role]:all) { (void)name;serialized+=row(role); }
  write(path,serialized);
  require(p::read_roles(path,false).selected.size()==325,"NF4 V2 policy and per-group state must roundtrip");
  write(path,serialized+"nf4_state\tunknown.weight\t"+std::string(64,'a')+"\t3d2fd550\n");
  rejects([&]{(void)p::read_roles(path,false);},"misplaced prepared NF4 state");
}

void gguf_inventory_and_source_semantics(const fs::path &directory) {
  Inventory all;
  std::uint64_t cursor=4096;
  for (const auto &[name,parameter]:p::parameter_inventory(true, geometry)) {
    const auto source=p::gguf_parameter(name, gguf_geometry);
    p::Role role{name,"F32",parameter.shape,"borrowed payload.bin",0,0};
    role.physical_name=source.physical;role.meaning=source.meaning;
    if (source.meaning==p::ParameterMeaning::kEmbeddingRows) role.dtype="F16";
    if (name=="mtp.fc.weight") role.dtype="Q8_0";
    role.bytes=p::gguf_physical_bytes(role.dtype,role.shape);
    cursor=(cursor+31)/32*32;role.original_offset=cursor;role.offset=cursor-4096;
    cursor+=role.bytes;all.emplace(name,std::move(role));
  }
  const p::OriginalSource original{"test/Qwen3.5-2B-MTP",std::string(40,'b'),"original.gguf",
      std::string(64,'c'),std::string(64,'d'),std::string(64,'e'),cursor,4096};
  const auto complete=p::select_gguf_roles(all,identity,original,true, gguf_geometry);
  const auto text=p::select_gguf_roles(all,identity,original,false, gguf_geometry);
  require(complete.representation==p::SourceRepresentation::kGgufConverted && complete.selected.size()==335 &&
      text.selected.size()==320 && text.original.lfs_sha256==original.lfs_sha256 &&
      text.original.xet_hash==original.xet_hash && text.original.header_sha256==original.header_sha256,
      "GGUF must retain complete source identity and select MTP payload independently");
  require(all.at("model.language_model.norm.weight").meaning==p::ParameterMeaning::kRmsEffectiveGamma &&
      all.at("model.language_model.layers.0.linear_attn.A_log").meaning==p::ParameterMeaning::kGdnNegativeExpDecay &&
      all.at("mtp.pre_fc_norm_embedding.weight").physical_name=="blk.24.nextn.enorm.weight" &&
      all.at("mtp.fc.weight").bytes==8912896 &&
      all.at("model.language_model.embed_tokens.weight").bytes==1017118720,
      "GGUF effective gamma/negative decay/F16/Q8 roles must not become HF parameters");
  const auto serialize=[&](const Inventory &values) {
    std::ostringstream out;
    out<<p::gguf_role_signature<<'\n'<<identity<<'\n'<<qwen_test::geometry_line(gguf_geometry)<<"\nsource "<<std::quoted(original.repository)<<' '
       <<original.revision<<' '<<std::quoted(original.file)<<' '<<original.bytes<<' '<<original.lfs_sha256
       <<' '<<std::quoted(original.xet_hash)<<' '<<original.header_sha256<<' '<<original.header_bytes<<'\n';
    for (const auto &[name,role]:values) {
      out<<name<<' '<<role.physical_name<<' '<<role.dtype<<' '<<p::meaning_name(role.meaning)<<' '<<role.shape.size();
      for (const auto dim:role.shape) out<<' '<<dim;
      out<<' '<<std::quoted(role.source.string())<<' '<<role.offset<<' '<<role.bytes<<' '<<role.original_offset<<'\n';
    }
    return out.str();
  };
  const auto path=directory/"gguf.tsv";
  write(path,serialize(all));
  const auto parsed=p::read_roles(path,true);
  require(parsed.selected.size()==335 && parsed.bytes==complete.bytes &&
      parsed.selected.front().source==directory/all.begin()->second.source &&
      p::read_roles(path,false).selected.size()==320 && p::is_role_signature(p::gguf_role_signature),
      "distinct GGUF format must roundtrip quoted borrowed sources without reading bodies");
  auto ordinary_geometry = gguf_geometry; ordinary_geometry.mtp_layers = 0;
  auto ordinary_roles = all;
  std::erase_if(ordinary_roles, [](const auto &entry) { return entry.first.starts_with("mtp."); });
  require(p::select_gguf_roles(ordinary_roles,identity,original,false,ordinary_geometry).selected.size()==320,
      "a source without NextN must remain valid for ordinary GGUF execution");
  rejects([&]{(void)p::select_gguf_roles(ordinary_roles,identity,original,true,ordinary_geometry);},
      "MTP requires");
  auto ordinary_manifest = serialize(ordinary_roles);
  const auto source_geometry_line = qwen_test::geometry_line(gguf_geometry);
  ordinary_manifest.replace(ordinary_manifest.find(source_geometry_line), source_geometry_line.size(),
      qwen_test::geometry_line(ordinary_geometry));
  write(path,ordinary_manifest);
  require(p::read_roles(path,false).selected.size()==320,
      "ordinary GGUF metadata without NextN must roundtrip through the shared reader");
  rejects([&]{(void)p::read_roles(path,true);},"MTP requires");
  const auto complete_text=serialize(all);
  write(path,complete_text+std::string(p::maximum_manifest_line_bytes+1,'x'));
  rejects([&]{(void)p::read_roles(path,false);},"line length limit");
  const auto last_row=complete_text.rfind('\n',complete_text.size()-2)+1;
  write(path,complete_text+complete_text.substr(last_row));
  rejects([&]{(void)p::read_roles(path,false);},"role count exceeds model limit");
  auto f16=all;
  auto &norm=f16.at("model.language_model.norm.weight");norm.dtype="F16";norm.bytes/=2;
  require(p::select_gguf_roles(f16,identity,original,true, gguf_geometry).selected.size()==335,
      "F16 effective gamma must keep its source type and meaning");
  rejects([&]{(void)p::select_roles(all,identity,true, geometry);},"HF prepared roles");
  for (const auto kind:{0,1,2,3,4,5}) {
    auto invalid=all;auto &role=invalid.at("mtp.fc.weight");
    if (kind==0) role.physical_name="blk.23.nextn.eh_proj.weight";
    else if (kind==1) role.meaning=p::ParameterMeaning::kHfRaw;
    else if (kind==2) role.bytes+=1;
    else if (kind==3) role.original_offset=original.bytes;
    else if (kind==4) role.original_offset=all.begin()->second.original_offset;
    else invalid.erase("mtp.norm.weight");
    rejects([&]{(void)p::select_gguf_roles(invalid,identity,original,false, gguf_geometry);},
        kind<2?"name/shape/meaning":kind<4?"byte range":kind==4?"overlap":"complete source-role");
  }
  auto source=original;source.header_sha256="bad";
  rejects([&]{(void)p::select_gguf_roles(all,identity,source,true, gguf_geometry);},"identity");
  rejects([&]{(void)p::gguf_physical_bytes("Q8_0",{2048,4095});},"K blocks");
  rejects([&]{(void)p::gguf_physical_bytes("Q8_0",{INT64_MAX,INT64_MAX-31});},"overflows");
  auto invalid=serialize(all);
  const auto at=invalid.find("gdn_negative_exp_A_log");
  invalid.replace(at,std::string("gdn_negative_exp_A_log").size(),"raw_A_log");
  write(path,invalid);
  rejects([&]{(void)p::read_roles(path,false);},"parameter meaning");
}
void source_geometry_drives_role_layout() {
  const auto larger = qwen_test::q27();
  const auto inventory = p::parameter_inventory(true, larger);
  require(inventory.size() == 866 && larger.attention_layers() == 16 && larger.recurrent_layers() == 48 &&
      larger.hidden == 5120 && larger.attention_width() == 6144 && larger.gdn_value_width() == 6144 &&
      larger.gdn_qkv_width() == 10240, "Q27 derived geometry must preserve three distinct model widths");
  require(inventory.at("lm_head.weight").shape == std::vector<std::int64_t>{248320,5120} &&
      inventory.at("model.language_model.layers.0.linear_attn.out_proj.weight").shape == std::vector<std::int64_t>{5120,6144} &&
      inventory.at("mtp.layers.0.self_attn.q_proj.weight").shape == std::vector<std::int64_t>{12288,5120} &&
      p::gguf_parameter("mtp.fc.weight", larger).physical == "blk.64.nextn.eh_proj.weight" &&
      p::gguf_parameter("lm_head.weight", larger).physical == "output.weight",
      "source inventory must bind the independent head and naturally located NextN roles");
  require(qwen35::read_model_geometry(qwen_test::geometry_line(larger)) == larger,
      "source geometry must roundtrip all independent dimensions and layout meaning");
  auto invalid = larger; invalid.gdn_value_heads = 47;
  rejects([&] { invalid.validate(); }, "geometry");
  invalid = larger; invalid.query_heads = INT64_MAX;
  rejects([&] { invalid.validate(); }, "geometry");
  invalid = larger; invalid.gdn_key_heads = invalid.gdn_value_heads = static_cast<std::int64_t>(INT32_MAX) + 1;
  rejects([&] { invalid.validate(); }, "geometry");
  invalid = larger; invalid.gdn_value_heads = static_cast<std::int64_t>(INT32_MAX) + 1;
  invalid.gdn_key_heads = 1;
  rejects([&] { invalid.validate(); }, "geometry");
  invalid = larger; invalid.layers[0] = 'X';
  rejects([&] { invalid.validate(); }, "geometry");
  invalid = larger; invalid.rope_theta = std::numeric_limits<float>::infinity();
  rejects([&] { invalid.validate(); }, "geometry");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("pass the scratch fixture directory");
    const fs::path directory = argv[1];
    fs::create_directories(directory);
    full_inventory_and_selection(directory);
    complete_validation_also_applies_when_mtp_is_off(directory);
    quantized_auxiliaries_are_complete_and_typed(directory);
    packed_axes_are_complete_before_weight_io();
    ct_storage_is_distinct_and_original_shape_is_checked(directory);
    malformed_input_and_shared_entrypoint_are_bounded(directory);
    gguf_inventory_and_source_semantics(directory);
    source_geometry_drives_role_layout();
    fp8_blocks_and_policy(directory);
    ct_mixed_and_cache_policy(directory);
    nf4_original_state_and_six_role_contract(directory);
    for (int i=2;i<argc;++i) {
      const auto all=p::read_roles(argv[i],true),text=p::read_roles(argv[i],false);
      require(all.geometry == text.geometry && !all.geometry.layers.empty() &&
          all.selected.size() >= p::parameter_inventory(true, all.geometry).size() &&
          text.selected.size() < all.selected.size(), "external actual V2 metadata contract differs");
    }
    std::printf("PASS %zu checks: dense335, pinned Q01/Q03/CT physical metadata, MTP selection, bounded inventory and malformed ranges\n", checks);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
