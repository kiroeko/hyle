#pragma once
#include "../../engine/models/glm/glm_named_parameters.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ggml.h>
#include "glm_gguf_projection_fragment.h"
#include "original_named_role_source.h"

// Pure B09 named-role adapter. No files, mapping, SHA calculation, upload or
// model execution occurs here. NamedSource is the EXISTING authenticated
// loose-role source after its model-neutral extraction (see proposal notes).
// Actual PinnedFile values must come from a complete acquisition; this file
// contains geometry/codec metadata only and deliberately has no payload pins.
namespace glm_gguf_named_parameters {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
using Value=sem::ValueId;
inline constexpr std::string_view kRepository="unsloth/GLM-5.3-GGUF";
inline constexpr std::string_view kRevision="346b3591c7f28d1a23716f97a065ecf12ec14771";
inline constexpr std::string_view kVariant="UD-Q4_K_XL";
inline constexpr std::string_view kFirstFile="UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf";
inline constexpr std::int64_t kHidden=6144,kQueryRank=2048,kHeads=64;
inline constexpr std::int64_t kNope=192,kRope=64,kLatent=512,kValue=256,kVocabulary=154880;
inline constexpr std::int64_t kTargetLayer=77,kNextnLayer=78;
inline constexpr std::size_t kStaticRoles=47;
inline constexpr std::uint64_t kStaticBytes=2567086080ULL;
// B09-specific source facts; root must reconcile these with the fresh actual
// header before constructing a runtime. Shared compute has no family defaults.
inline constexpr glm_gguf_model_types::Geometry kGeometry{
    6144,2048,64,192,64,512,256,154880,128,32,1,2048,1.e-5F,1.e-6F,1.F/16.F};
struct Spec final {
  std::string_view name;
  std::uint32_t native;
  std::array<std::int64_t,3> gguf_dimensions;
  std::size_t rank,bytes;
};
inline constexpr std::array<Spec,kStaticRoles> kStaticSpecs{
    Spec{"output.weight", 8, {6144,154880,0}, 2, 1011056640},
    Spec{"output_norm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"token_embd.weight", 8, {6144,154880,0}, 2, 1011056640},
    Spec{"blk.77.attn_k_b.weight", 8, {192,512,64}, 3, 6684672},
    Spec{"blk.77.attn_kv_a_mqa.weight", 8, {6144,576,0}, 2, 3760128},
    Spec{"blk.77.attn_kv_a_norm.weight", 0, {512,0,0}, 1, 2048},
    Spec{"blk.77.attn_norm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.77.attn_output.weight", 8, {16384,6144,0}, 2, 106954752},
    Spec{"blk.77.attn_q_a.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.77.attn_q_a_norm.weight", 0, {2048,0,0}, 1, 8192},
    Spec{"blk.77.attn_q_b.weight", 8, {2048,16384,0}, 2, 35651584},
    Spec{"blk.77.attn_v_b.weight", 8, {512,256,64}, 3, 8912896},
    Spec{"blk.77.exp_probs_b.bias", 0, {256,0,0}, 1, 1024},
    Spec{"blk.77.ffn_down_shexp.weight", 8, {2048,6144,0}, 2, 13369344},
    Spec{"blk.77.ffn_gate_inp.weight", 0, {6144,256,0}, 2, 6291456},
    Spec{"blk.77.ffn_gate_shexp.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.77.ffn_norm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.77.ffn_up_shexp.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.77.indexer.attn_k.weight", 8, {6144,128,0}, 2, 835584},
    Spec{"blk.77.indexer.attn_q_b.weight", 8, {2048,4096,0}, 2, 8912896},
    Spec{"blk.77.indexer.k_norm.bias", 0, {128,0,0}, 1, 512},
    Spec{"blk.77.indexer.k_norm.weight", 0, {128,0,0}, 1, 512},
    Spec{"blk.77.indexer.proj.weight", 0, {6144,32,0}, 2, 786432},
    Spec{"blk.78.attn_k_b.weight", 8, {192,512,64}, 3, 6684672},
    Spec{"blk.78.attn_kv_a_mqa.weight", 8, {6144,576,0}, 2, 3760128},
    Spec{"blk.78.attn_kv_a_norm.weight", 0, {512,0,0}, 1, 2048},
    Spec{"blk.78.attn_norm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.78.attn_output.weight", 8, {16384,6144,0}, 2, 106954752},
    Spec{"blk.78.attn_q_a.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.78.attn_q_a_norm.weight", 0, {2048,0,0}, 1, 8192},
    Spec{"blk.78.attn_q_b.weight", 8, {2048,16384,0}, 2, 35651584},
    Spec{"blk.78.attn_v_b.weight", 8, {512,256,64}, 3, 8912896},
    Spec{"blk.78.exp_probs_b.bias", 0, {256,0,0}, 1, 1024},
    Spec{"blk.78.ffn_down_shexp.weight", 8, {2048,6144,0}, 2, 13369344},
    Spec{"blk.78.ffn_gate_inp.weight", 0, {6144,256,0}, 2, 6291456},
    Spec{"blk.78.ffn_gate_shexp.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.78.ffn_norm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.78.ffn_up_shexp.weight", 8, {6144,2048,0}, 2, 13369344},
    Spec{"blk.78.indexer.attn_k.weight", 8, {6144,128,0}, 2, 835584},
    Spec{"blk.78.indexer.attn_q_b.weight", 8, {2048,4096,0}, 2, 8912896},
    Spec{"blk.78.indexer.k_norm.bias", 0, {128,0,0}, 1, 512},
    Spec{"blk.78.indexer.k_norm.weight", 0, {128,0,0}, 1, 512},
    Spec{"blk.78.indexer.proj.weight", 0, {6144,32,0}, 2, 786432},
    Spec{"blk.78.nextn.eh_proj.weight", 8, {12288,6144,0}, 2, 80216064},
    Spec{"blk.78.nextn.enorm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.78.nextn.hnorm.weight", 0, {6144,0,0}, 1, 24576},
    Spec{"blk.78.nextn.shared_head_norm.weight", 0, {6144,0,0}, 1, 24576},
};
inline c::TensorDesc description(const Spec &s) {
  if(s.native==0) {
    if(s.rank==1)return sem::dense({s.gguf_dimensions[0]});
    if(s.rank==2)return sem::dense({s.gguf_dimensions[1],s.gguf_dimensions[0]});
    if(s.rank==3)return sem::dense({s.gguf_dimensions[2],s.gguf_dimensions[1],s.gguf_dimensions[0]});
    throw std::runtime_error("converted plain source rank is not supported");
  }
  const auto codec=s.native==8?c::GgufQuantizedType::kQ8_0:
      s.native==12?c::GgufQuantizedType::kQ4K:s.native==13?c::GgufQuantizedType::kQ5K:
      s.native==14?c::GgufQuantizedType::kQ6K:throw std::runtime_error("converted source native codec is not implemented");
  if(s.rank!=2&&s.rank!=3)throw std::runtime_error("converted packed source rank differs");
  auto d=s.rank==2?c::gguf_weight_description(codec,
      c::DataType::kFloat32,s.gguf_dimensions[0],s.gguf_dimensions[1]):
      c::gguf_grouped_weight_description(codec,
      c::DataType::kFloat32,s.gguf_dimensions[2],s.gguf_dimensions[1],s.gguf_dimensions[0]);
  if(!d)throw std::runtime_error("B09 canonical packed source geometry is invalid");
  if(s.name=="token_embd.weight") {
    auto rows=d->transposed();if(!rows)throw std::runtime_error("B09 embedding orientation is invalid");
    return *rows;
  }
  return *d;
}
inline bool actual_pin(std::string_view pin) {
  if(pin.size()!=64)return false;
  for(const char ch:pin)if(!((ch>='0'&&ch<='9')||(ch>='a'&&ch<='f')))return false;
  return true;
}
// ManifestSpec/PinnedFile are the common named source's existing structures,
// not a second loader or a model-specific input protocol. Physical order is
// exactly roles-schema.json filtered to its 47 non-expert sources.
inline std::vector<original_named_role_source::ManifestSpec> manifest(
    std::span<const Spec> specs,std::span<const original_named_role_source::PinnedFile> files,std::uint64_t expected_bytes) {
  if(specs.empty()||files.size()!=specs.size())throw std::runtime_error("converted manifest needs every distinct actual static pin");
  std::vector<original_named_role_source::ManifestSpec> result;result.reserve(specs.size());
  std::uint64_t extent=0;
  for(std::size_t i=0;i<specs.size();++i) {
    const auto &s=specs[i];auto d=description(s);
    const auto bytes=d.required_bytes();
    if(!bytes||*bytes!=s.bytes||!actual_pin(files[i].sha256)||
        files[i].file.filename()!=std::filesystem::path(std::string(s.name)+".bin"))
      throw std::runtime_error("B09 actual pin/file/order differs from its original named role");
    result.push_back({std::string(s.name),files[i].file,std::string(files[i].sha256),static_cast<ggml_type>(s.native),d});
    if(s.bytes>std::numeric_limits<std::uint64_t>::max()-extent)throw std::runtime_error("converted static extent overflows");
    extent+=s.bytes;
  }
  if(extent==0||extent!=expected_bytes)throw std::runtime_error("converted static role byte accounting differs from its explicit manifest");
  return result;
}
inline constexpr std::array<Spec,6> kExpertSpecs{
    Spec{"blk.77.ffn_down_exps.weight", 14, {2048,6144,256}, 3, 2642411520ULL},
    Spec{"blk.77.ffn_gate_exps.weight", 12, {6144,2048,256}, 3, 1811939328ULL},
    Spec{"blk.77.ffn_up_exps.weight", 12, {6144,2048,256}, 3, 1811939328ULL},
    Spec{"blk.78.ffn_down_exps.weight", 13, {2048,6144,256}, 3, 2214592512ULL},
    Spec{"blk.78.ffn_gate_exps.weight", 12, {6144,2048,256}, 3, 1811939328ULL},
    Spec{"blk.78.ffn_up_exps.weight", 12, {6144,2048,256}, 3, 1811939328ULL},
};
} // namespace glm_gguf_named_parameters
