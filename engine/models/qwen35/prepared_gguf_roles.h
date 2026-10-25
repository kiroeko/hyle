#pragma once
#include "prepared_roles.h"

// Engine-private source contract. These format names describe original
// storage; they do not choose a Provider, kernel or activation precision.
namespace qwen35::prepared {
struct StorageBlock final { std::size_t elements, bytes; };
inline StorageBlock gguf_storage_block(std::string_view dtype) {
  if (dtype == "F32") return {1,4};
  if (dtype == "F16") return {1,2};
  if (dtype == "Q8_0") return {32,34};
  if (dtype == "NVFP4") return {64,36};
  if (dtype == "Q3_K" || dtype == "IQ3_S") return {256,110};
  if (dtype == "IQ4_XS") return {256,136};
  if (dtype == "IQ4_NL") return {32,18};
  if (dtype == "Q4_K") return {256,144};
  if (dtype == "Q5_K") return {256,176};
  if (dtype == "Q6_K") return {256,210};
  throw std::runtime_error("unsupported prepared GGUF storage: " + std::string(dtype));
}
inline std::size_t gguf_physical_bytes(std::string_view dtype, const std::vector<std::int64_t> &shape) {
  const auto block = gguf_storage_block(dtype);
  if (block.elements == 1) return physical_bytes(dtype,shape);
  if (shape.size() != 2 || shape[1] <= 0 || std::uint64_t(shape[1]) % block.elements)
    throw std::runtime_error("prepared GGUF weight needs complete K blocks in logical[N,K]");
  const auto blocks = std::uint64_t(shape[1]) / block.elements;
  const auto maximum = (std::numeric_limits<std::size_t>::max)();
  if (shape[0] <= 0 || blocks > maximum / block.bytes ||
      std::uint64_t(shape[0]) > maximum / (blocks * block.bytes))
    throw std::runtime_error("prepared GGUF physical extent overflows");
  return static_cast<std::size_t>(std::uint64_t(shape[0]) * blocks * block.bytes);
}
struct GgufParameter final { std::string physical; ParameterMeaning meaning; };
inline GgufParameter gguf_parameter(std::string_view name, const ModelGeometry &geometry) {
  const auto draft_prefix = "blk." + std::to_string(geometry.layers.size()) + ".";
  if (name == "model.language_model.embed_tokens.weight") return {"token_embd.weight",ParameterMeaning::kEmbeddingRows};
  if (name == "lm_head.weight" && !geometry.tied_head) return {"output.weight",ParameterMeaning::kLinearWeight};
  if (name == "model.language_model.norm.weight") return {"output_norm.weight",ParameterMeaning::kRmsEffectiveGamma};
  for (const auto &[logical,physical] : std::initializer_list<std::pair<std::string_view,std::string_view>>{
      {"mtp.fc.weight","eh_proj"},{"mtp.norm.weight","shared_head_norm"},
      {"mtp.pre_fc_norm_embedding.weight","enorm"},{"mtp.pre_fc_norm_hidden.weight","hnorm"}})
    if (name == logical) return {draft_prefix+"nextn."+std::string(physical)+".weight",
        name == "mtp.fc.weight" ? ParameterMeaning::kLinearWeight : ParameterMeaning::kRmsEffectiveGamma};
  std::string prefix;
  if (name.starts_with("mtp.layers.0.")) { prefix=draft_prefix;name.remove_prefix(13); }
  else if (name.starts_with("model.language_model.layers.")) {
    name.remove_prefix(28);
    const auto dot=name.find('.');
    if (dot == std::string_view::npos) throw std::runtime_error("invalid prepared GGUF layer name");
    prefix="blk."+std::string(name.substr(0,dot))+".";name.remove_prefix(dot+1);
  } else throw std::runtime_error("unknown prepared GGUF logical parameter");
  for (const auto &[logical,physical] : std::initializer_list<std::pair<std::string_view,std::string_view>>{
      {"input_layernorm.weight","attn_norm.weight"},{"post_attention_layernorm.weight","post_attention_norm.weight"},
      {"self_attn.q_norm.weight","attn_q_norm.weight"},{"self_attn.k_norm.weight","attn_k_norm.weight"}})
    if (name == logical) return {prefix+std::string(physical),ParameterMeaning::kRmsEffectiveGamma};
  for (const auto &[logical,physical] : std::initializer_list<std::pair<std::string_view,std::string_view>>{
      {"mlp.gate_proj.weight","ffn_gate.weight"},{"mlp.up_proj.weight","ffn_up.weight"},{"mlp.down_proj.weight","ffn_down.weight"},
      {"self_attn.q_proj.weight","attn_q.weight"},{"self_attn.k_proj.weight","attn_k.weight"},
      {"self_attn.v_proj.weight","attn_v.weight"},{"self_attn.o_proj.weight","attn_output.weight"},
      {"linear_attn.in_proj_qkv.weight","attn_qkv.weight"},{"linear_attn.in_proj_z.weight","attn_gate.weight"},
      {"linear_attn.in_proj_a.weight","ssm_alpha.weight"},{"linear_attn.in_proj_b.weight","ssm_beta.weight"},
      {"linear_attn.out_proj.weight","ssm_out.weight"}})
    if (name == logical) return {prefix+std::string(physical),ParameterMeaning::kLinearWeight};
  if (name == "linear_attn.A_log") return {prefix+"ssm_a",ParameterMeaning::kGdnNegativeExpDecay};
  if (name == "linear_attn.dt_bias") return {prefix+"ssm_dt.bias",ParameterMeaning::kGdnDtBias};
  if (name == "linear_attn.norm.weight") return {prefix+"ssm_norm.weight",ParameterMeaning::kGdnEffectiveGamma};
  if (name == "linear_attn.conv1d.weight") return {prefix+"ssm_conv1d.weight",ParameterMeaning::kGdnConvolution};
  throw std::runtime_error("unknown prepared GGUF logical parameter");
}
inline std::string_view meaning_name(ParameterMeaning meaning) {
  switch (meaning) {
    case ParameterMeaning::kLinearWeight:return "linear_weight";
    case ParameterMeaning::kEmbeddingRows:return "embedding_rows";
    case ParameterMeaning::kRmsEffectiveGamma:return "rms_effective_gamma";
    case ParameterMeaning::kGdnEffectiveGamma:return "gdn_effective_gamma";
    case ParameterMeaning::kGdnNegativeExpDecay:return "gdn_negative_exp_A_log";
    case ParameterMeaning::kGdnDtBias:return "gdn_dt_bias";
    case ParameterMeaning::kGdnConvolution:return "gdn_convolution_kernel";
    case ParameterMeaning::kHfRaw:break;
  }
  throw std::runtime_error("invalid converted GGUF parameter meaning");
}
inline void validate_original_source(const OriginalSource &source) {
  validate_identity(source.lfs_sha256);validate_identity(source.header_sha256);
  if (!source.xet_hash.empty()) validate_identity(source.xet_hash);
  if (source.revision.size()!=40 || !std::ranges::all_of(source.revision,[](char ch){
        return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');}) || source.repository.empty() ||
      !source.file.ends_with(".gguf") || source.header_bytes<24 || source.header_bytes>(16u<<20) ||
      source.header_bytes>=source.bytes)
    throw std::runtime_error("invalid prepared GGUF original source identity");
}
inline Roles select_gguf_roles(const std::map<std::string,Role> &all,std::string identity,
                              OriginalSource source,bool include_mtp,const ModelGeometry &geometry) {
  geometry.validate();
  if (include_mtp && geometry.mtp_layers != 1)
    throw std::runtime_error("MTP requires one NextN layer in the prepared GGUF source geometry");
  if (geometry.gdn_layout != GdnHeadLayout::kCyclic) throw std::runtime_error("converted GGUF requires its cyclic GDN source layout");
  validate_identity(identity);validate_original_source(source);
  const auto inventory=parameter_inventory(geometry.mtp_layers != 0, geometry);
  if (all.size()!=inventory.size()) throw std::runtime_error("prepared GGUF requires its complete source-role inventory");
  Roles result{std::move(identity),{},0,SourceRepresentation::kGgufConverted,std::move(source),geometry};
  std::set<std::string> physical;
  std::vector<std::pair<std::uint64_t,std::uint64_t>> extents;
  for (const auto &[name,parameter] : inventory) {
    const auto found=all.find(name);
    if (found==all.end()) throw std::runtime_error("missing prepared GGUF role: "+name);
    const auto &role=found->second;const auto expected=gguf_parameter(name,geometry);
    if (role.name!=name || role.shape!=parameter.shape || role.physical_name!=expected.physical ||
        role.meaning!=expected.meaning || !physical.insert(role.physical_name).second)
      throw std::runtime_error("prepared GGUF physical name/shape/meaning differs: "+name);
    const auto block=gguf_storage_block(role.dtype);
    const bool matrix=role.meaning==ParameterMeaning::kLinearWeight||role.meaning==ParameterMeaning::kEmbeddingRows;
    if ((!matrix && block.elements!=1) || role.source.empty() ||
        role.source.native().find(std::filesystem::path::value_type{})!=std::filesystem::path::string_type::npos)
      throw std::runtime_error("prepared GGUF scalar/norm/source representation differs: "+name);
    if (role.bytes!=gguf_physical_bytes(role.dtype,role.shape) || role.offset>UINT64_MAX-role.bytes ||
        role.original_offset<result.original.header_bytes || role.original_offset>result.original.bytes ||
        role.bytes>result.original.bytes-role.original_offset)
      throw std::runtime_error("prepared GGUF physical byte range differs: "+name);
    extents.emplace_back(role.original_offset,role.original_offset+role.bytes);
    if (include_mtp || !name.starts_with("mtp.")) {
      if (role.bytes>(std::numeric_limits<std::size_t>::max)()-result.bytes)
        throw std::runtime_error("prepared GGUF model byte extent overflows");
      result.bytes+=role.bytes;result.selected.push_back(role);
    }
  }
  std::sort(extents.begin(),extents.end());
  for (std::size_t i=1;i<extents.size();++i)
    if (extents[i-1].second>extents[i].first) throw std::runtime_error("prepared GGUF original ranges overlap");
  return result;
}
inline Roles read_gguf_roles(const std::filesystem::path &path,bool include_mtp) {
  std::ifstream input(path);std::string signature,identity,line;
  read_manifest_line(input,signature,128);read_manifest_line(input,identity,65);read_manifest_line(input,line);
  for (auto *value:{&signature,&identity,&line}) if (value->ends_with('\r')) value->pop_back();
  if (signature!=gguf_role_signature) throw std::runtime_error("invalid prepared GGUF signature");
  const auto geometry = read_model_geometry(line);
  read_manifest_line(input,line);
  OriginalSource source;
  std::istringstream header(line);std::string tag,extra;
  if (!(header>>tag>>std::quoted(source.repository)>>source.revision>>std::quoted(source.file))||tag!="source")
    throw std::runtime_error("invalid prepared GGUF original source header");
  source.bytes=unsigned_field(header);
  if (!(header>>source.lfs_sha256>>std::quoted(source.xet_hash)>>source.header_sha256))
    throw std::runtime_error("missing prepared GGUF original source hashes");
  source.header_bytes=unsigned_field(header);
  if (header>>extra) throw std::runtime_error("extra prepared GGUF source field");
  const auto maximum_roles = parameter_inventory(geometry.mtp_layers != 0,geometry).size();
  std::map<std::string,Role> all;
  while (read_manifest_line(input,line)) {
    if (line.find_first_not_of(" \t\r")==std::string::npos) continue;
    if (all.size()==maximum_roles) throw std::runtime_error("prepared GGUF physical role count exceeds model limit");
    std::istringstream fields(line);Role role;std::string meaning;
    if (!(fields>>role.name>>role.physical_name>>role.dtype>>meaning))
      throw std::runtime_error("invalid prepared GGUF role");
    role.meaning=gguf_parameter(role.name,geometry).meaning;
    if (meaning!=meaning_name(role.meaning)) throw std::runtime_error("prepared GGUF parameter meaning differs");
    const auto rank=unsigned_field(fields);
    if (!rank||rank>3) throw std::runtime_error("prepared GGUF role rank exceeds model geometry");
    for (std::size_t i=0;i<rank;++i) {
      const auto dim=unsigned_field(fields);
      if (!dim||dim>std::uint64_t((std::numeric_limits<std::int64_t>::max)()))
        throw std::runtime_error("prepared GGUF shape extent overflows");
      role.shape.push_back(static_cast<std::int64_t>(dim));
    }
    std::string file;fields>>std::ws;
    if (fields.peek()!='"'||!(fields>>std::quoted(file))||file.empty()||file.find('\0')!=std::string::npos)
      throw std::runtime_error("prepared GGUF role requires quoted source file");
    if (file.size()>maximum_manifest_path_bytes)
      throw std::runtime_error("prepared source path length limit exceeded");
    role.source=path.parent_path()/std::filesystem::path(file);role.offset=unsigned_field(fields);
    const auto bytes=unsigned_field(fields);role.original_offset=unsigned_field(fields);
    if (bytes>(std::numeric_limits<std::size_t>::max)()||(fields>>extra))
      throw std::runtime_error("prepared GGUF physical byte extent overflows");
    role.bytes=static_cast<std::size_t>(bytes);
    if (!all.emplace(role.name,std::move(role)).second) throw std::runtime_error("duplicate prepared GGUF role");
  }
  if (!input.eof()) throw std::runtime_error("failed reading prepared GGUF roles");
  return select_gguf_roles(all,std::move(identity),std::move(source),include_mtp,geometry);
}
} // namespace qwen35::prepared
