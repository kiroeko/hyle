#pragma once

#include "qwen_packed_moe_roles.h"
#include "qwen_gguf_selected_experts.h"

// B07 exact frozen census schema, not payload pins or invented source offsets.
// Files must be generated only from completed original-range receipts.
namespace qwen_gguf_q24_artifacts {
namespace named=original_named_role_source;
namespace original=qwen_original_mtp_roles;
namespace nextn=qwen24_original_nextn_roles;
namespace target=qwen24_original_target_roles;
namespace c=hyle::compute;
inline constexpr original::Identity kIdentity{"unsloth/Qwen3.8-2.4T-A95B-GGUF",
    "567d3e6ac26c5474b18311e619c04350fb9a5556","Q8_0","Q8_0/Qwen3.8-2.4T-A95B-Q8_0-00001-of-00056.gguf"};
inline constexpr std::size_t kStaticRoles=33,kStaticBytes=5497491456ULL;
struct Schema final {std::string_view name;ggml_type native;std::int64_t k,n;bool embedding;};
inline constexpr std::array<Schema,kStaticRoles> kSchema{
    Schema{"token_embd.weight",GGML_TYPE_Q8_0,8192,248320,true},
    Schema{"output_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.92.nextn.eh_proj.weight",GGML_TYPE_Q8_0,16384,8192,false},
    Schema{"blk.92.attn_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.92.ffn_gate_inp.weight",GGML_TYPE_F32,8192,512,false},
    Schema{"blk.92.ffn_down_shexp.weight",GGML_TYPE_Q8_0,2048,8192,false},
    Schema{"blk.92.ffn_gate_shexp.weight",GGML_TYPE_Q8_0,8192,2048,false},
    Schema{"blk.92.ffn_up_shexp.weight",GGML_TYPE_Q8_0,8192,2048,false},
    Schema{"blk.92.ffn_gate_inp_shexp.weight",GGML_TYPE_F32,8192,1,false},
    Schema{"blk.92.post_attention_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.92.attn_k_norm.weight",GGML_TYPE_F32,0,256,false},
    Schema{"blk.92.attn_k.weight",GGML_TYPE_Q8_0,8192,1024,false},
    Schema{"blk.92.attn_output.weight",GGML_TYPE_Q8_0,16384,8192,false},
    Schema{"blk.92.attn_q_norm.weight",GGML_TYPE_F32,0,256,false},
    Schema{"blk.92.attn_q.weight",GGML_TYPE_Q8_0,8192,32768,false},
    Schema{"blk.92.attn_v.weight",GGML_TYPE_Q8_0,8192,1024,false},
    Schema{"blk.92.nextn.shared_head_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.92.nextn.enorm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.92.nextn.hnorm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.91.attn_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.91.ffn_gate_inp.weight",GGML_TYPE_F32,8192,512,false},
    Schema{"blk.91.ffn_down_shexp.weight",GGML_TYPE_Q8_0,2048,8192,false},
    Schema{"blk.91.ffn_gate_shexp.weight",GGML_TYPE_Q8_0,8192,2048,false},
    Schema{"blk.91.ffn_up_shexp.weight",GGML_TYPE_Q8_0,8192,2048,false},
    Schema{"blk.91.ffn_gate_inp_shexp.weight",GGML_TYPE_F32,8192,1,false},
    Schema{"blk.91.post_attention_norm.weight",GGML_TYPE_F32,0,8192,false},
    Schema{"blk.91.attn_k_norm.weight",GGML_TYPE_F32,0,256,false},
    Schema{"blk.91.attn_output.weight",GGML_TYPE_Q8_0,16384,8192,false},
    Schema{"output.weight",GGML_TYPE_Q8_0,8192,248320,false},
    Schema{"blk.91.attn_k.weight",GGML_TYPE_Q8_0,8192,1024,false},
    Schema{"blk.91.attn_q_norm.weight",GGML_TYPE_F32,0,256,false},
    Schema{"blk.91.attn_q.weight",GGML_TYPE_Q8_0,8192,32768,false},
    Schema{"blk.91.attn_v.weight",GGML_TYPE_Q8_0,8192,1024,false},
};
inline c::TensorDesc description(const Schema &s) {
  if(!s.k)return original::detail::dense({s.n});
  if(s.native==GGML_TYPE_F32)return original::detail::dense({s.n,s.k}); // Physical original rows, bound once.
  auto d=original::detail::take(c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,s.k,s.n),"B07 actual Q8_0 role");
  if(s.embedding)d=original::detail::take(d.transposed(),"B07 actual Gather row bank");return d;
}
inline std::array<named::ManifestSpec,kStaticRoles> manifest(const std::array<named::PinnedFile,kStaticRoles> &files) {
  std::array<named::ManifestSpec,kStaticRoles> result;
  for(std::size_t i=0;i<result.size();++i) {
    const auto &s=kSchema[i];result[i]={std::string(s.name),files[i].file,files[i].sha256,s.native,description(s)};
  }
  return result;
}
inline original::q24::Data kv_data(std::shared_ptr<const named::Data> source) {
  if(!source||source->size()!=kStaticRoles)throw std::runtime_error("B07 requires its33 actually acquired original static roles");
  const auto &id=source->identity();
  if(id.repository!=kIdentity.repository||id.revision!=kIdentity.revision||id.variant!=kIdentity.variant)
    throw std::runtime_error("B07 static source identity differs from its frozen census");
  std::size_t bytes=0;
  for(const auto &schema:kSchema) {
    const auto &s=source->spec(schema.name);
    if(s.native!=schema.native||s.description!=description(schema))throw std::runtime_error("B07 actual static descriptor differs from its fixed census");
    bytes+=s.bytes;
  }
  if(bytes!=kStaticBytes)throw std::runtime_error("B07 actual33-role byte extent differs");
  return {.codec=c::GgufQuantizedType::kQ8_0,.native=GGML_TYPE_Q8_0,.identity=&source->identity(),.original_owner=source,.key_source=source->bytes("blk.92.attn_k.weight"),
      .value_source=source->bytes("blk.92.attn_v.weight"),.norm_source=source->norm("blk.92.attn_k_norm.weight")};
}
inline original::q24::Roles input_roles(const named::Roles &actual,const c::StorageOwner &owner) {
  const auto frequencies=original::q24::frequencies();
  return {actual.at("blk.92.attn_k.weight"),actual.at("blk.92.attn_v.weight"),actual.at("blk.92.attn_k_norm.weight"),
      original::detail::bind("text.rope.frequencies",original::detail::dense({32}),owner,std::as_bytes(std::span(frequencies)),&actual.source->identity())};
}
} // namespace qwen_gguf_q24_artifacts
