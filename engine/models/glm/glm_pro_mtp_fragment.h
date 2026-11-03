#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "glm_official_index_cache.h"
#include "glm_original_moe_fragment.h"

// Model-local original GLM Pro NextN construction. Values name operands in
// this same ordinary Fragment; no loading, execution, provider or acceptance
// policy is hidden here. The selected official non-fused indexer keeps its
// Hadamard/UE8M0 boundary. Main cache rows contain BOTH CKV512 and rotated KPE64.
namespace glm_pro_mtp_fragment {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
namespace core=hyle::inference::model_definition::ops;
namespace codec=glm_official_index_cache;
namespace moe=glm_original_moe_fragment;
using Value=sem::ValueId;
inline constexpr std::int64_t kHidden=6144,kQueryRank=2048,kHeads=64;
inline constexpr std::int64_t kNope=192,kRope=64,kLatent=512,kValue=256;
inline constexpr std::int64_t kCacheDepth=kLatent+kRope,kIndexDepth=128,kIndexHeads=32;
inline constexpr std::int64_t kVocabulary=154880;
inline constexpr float kEpsilon=1.0e-5F,kAttentionScale=1.0F/16.0F;

struct Parameters final {
  Value eh,embedding_norm,hidden_norm,input_norm,post_norm;
  Value index_bias,index_norm,index_weights,index_key,index_query;
  Value kv_norm,kv_a,output,query_norm,query_a,query_b;
  Value shared_norm,head,prepared_key,prepared_value;
};
// Shared MLA operands. Target and NextN callers supply their own trained
// parameters and explicit residual; this layer knows neither EH nor LM policy.
struct MlaParameters final {
  Value kv_norm,kv_a,output,post_norm,query_b,prepared_key,prepared_value;
};
MlaParameters mla_parameters(const Parameters &p);
struct NormPorts final {
  Value input,raw,stored,visible;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored,&visible})visitor(*value);
  }
};
struct ProjectionPorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct RopePorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct Prefix final {
  Value hidden,embedding,joined;
  NormPorts hidden_norm,embedding_norm,input_norm,query_norm;
  ProjectionPorts eh,query;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&hidden,&embedding,&joined})visitor(*value);
    hidden_norm.visit_values(visitor);embedding_norm.visit_values(visitor);
    input_norm.visit_values(visitor);query_norm.visit_values(visitor);
    eh.visit_values(visitor);query.visit_values(visitor);
  }
};
struct IndexHistory final {
  Value input,positions,codes,scales;
  ProjectionPorts key;
  NormPorts key_norm;
  RopePorts rope;
  codec::Quantized encoded;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&positions,&codes,&scales})visitor(*value);
    key.visit_values(visitor);key_norm.visit_values(visitor);rope.visit_values(visitor);
    for(auto *value:{&encoded.codes,&encoded.scale,&encoded.rotated,&encoded.transformed,&encoded.input})visitor(*value);
  }
};
struct IndexPorts final {
  IndexHistory history;
  ProjectionPorts query;
  RopePorts rope;
  codec::Quantized encoded;
  Value weights_input,weights,scores,ids;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    history.visit_values(visitor);query.visit_values(visitor);rope.visit_values(visitor);
    for(auto *value:{&encoded.codes,&encoded.scale,&encoded.rotated,&encoded.transformed,&encoded.input,
        &weights_input,&weights,&scores,&ids})visitor(*value);
  }
};
struct AttentionPorts final {
  ProjectionPorts query,kv,output;
  NormPorts kv_norm,post_norm;
  RopePorts query_rope,key_rope;
  Value absorbed_input,absorbed_stored,query_full,row_stored;
  Value selected_full,selected_value,attention_raw,attention_stored,value_stored;
  Value residual_sum,residual_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    query.visit_values(visitor);kv.visit_values(visitor);output.visit_values(visitor);
    kv_norm.visit_values(visitor);post_norm.visit_values(visitor);
    query_rope.visit_values(visitor);key_rope.visit_values(visitor);
    for(auto *value:{&absorbed_input,&absorbed_stored,&query_full,&row_stored,
        &selected_full,&selected_value,&attention_raw,&attention_stored,&value_stored,
        &residual_sum,&residual_stored})visitor(*value);
  }
};
struct State final {
  Value main,codes,scales,selection,positions;
};
struct Options final {
  std::int64_t visible=0,selected=2048;
  bool fresh=true;
};
struct Ports final {
  Prefix prefix;
  AttentionPorts attention;
  std::optional<IndexPorts> index;
  Value positions,combined,logits,top_values,pending;
  NormPorts shared_norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    prefix.visit_values(visitor);attention.visit_values(visitor);
    if(index)index->visit_values(visitor);
    shared_norm.visit_values(visitor);
    for(auto *value:{&positions,&combined,&logits,&top_values,&pending})visitor(*value);
  }
};
struct Result final {
  State state;
  Value hidden,pending;
  Ports ports;
  moe::Result feed_forward;
};
struct PreparedKv final {Value decoded,key,value;};

void require(sem::Fragment &f,Value value,const c::TensorDesc &description);
Value cast(sem::Fragment &f,Value value,c::DataType type);
Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> dimensions);
Value permute(sem::Fragment &f,Value value,std::initializer_list<std::int32_t> axes);
Value slice(sem::Fragment &f,Value value,std::int32_t axis,std::int64_t first,std::int64_t count);
Value contiguous(sem::Fragment &f,Value value);
NormPorts norm(sem::Fragment &f,Value actual_f32,Value gamma,
                      std::int64_t width,bool layer=false,Value bias={});
ProjectionPorts fp8(sem::Fragment &f,Value actual_bf16,Value weight);
RopePorts rotate(sem::Fragment &f,Value actual_bf16,Value positions);

// Cold only. Decode the ONE original macroblock grid before the per-head
// split:448 rows/head is not divisible by128, so scales must not reset at a
// head boundary. Outputs are compact BF16, never a persistent F32 weight bank.
PreparedKv append_prepare_kv(sem::Fragment &f,Value encoded);

Prefix append_prefix(sem::Fragment &f,Value hidden,Value embedding,const Parameters &p);

// Fresh selection and accepted-feature catchup share this actual unpooled
// history producer. A skip step never calls it. Caller ensures positions are
// the corresponding absolute append rows of its finite declared cache.
IndexHistory append_index_history(sem::Fragment &f,Value input,Value positions,
    Value old_codes,Value old_scales,const Parameters &p);

// Prefix and suffix are split at the optional learned-indexer frontier.
// The shared-index target77 legitimately supplies upstream selection directly.
Value append_mla_prefix(sem::Fragment &f,Value input_normalized,Value query_normalized,
    Value old_main,Value positions,const MlaParameters &p,AttentionPorts &a);
void append_mla_suffix(sem::Fragment &f,Value main,Value selection,Value residual,
    const MlaParameters &p,AttentionPorts &a);

Result append_step(sem::Fragment &f,const Prefix &prefix,const State &state,
    const Parameters &p,const moe::Parameters &moe_parameters,const Options &options);
} // namespace glm_pro_mtp_fragment
