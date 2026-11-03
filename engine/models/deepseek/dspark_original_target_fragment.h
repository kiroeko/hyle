#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "dspark_component_math.h"
#include "dspark_target_geometry.h"
#include "dspark_formal_fragment.h"
#include "hyle/compute/low_precision.h"

// Shared original Flash0731/Pro0813 target-tail mathematics. Every parameter is an actual
// caller-owned immutable Value; every cache is a typed ordinary Graph input.
// Decode plans are cold frontier facts, checked by the consumer before submit.
// No parameter loading, routing-provider choice, executor or acceptance policy.
namespace dspark_original_target_fragment {
namespace ds=dspark_component_math;
namespace core=hyle::inference::model_definition::ops;
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
using Value=sem::ValueId;
namespace geometry=dspark_target_geometry;
using Geometry=geometry::Geometry;
// Default Flash dimensions retained for callers preparing that exact fixture.
// Model-dependent construction below uses its explicit cold Plan geometry.
inline constexpr auto kHidden=geometry::kFlash.hidden,kHeads=geometry::kFlash.heads,kQueryRank=geometry::kFlash.query_rank;
inline constexpr auto kGroups=geometry::kFlash.groups,kGroupWidth=geometry::kFlash.group_width();
inline constexpr auto kDepth=geometry::kDepth,kOutputRank=geometry::kOutputRank,kWindow=geometry::kWindow;
inline constexpr auto kIndexDepth=geometry::kIndexDepth,kIndexHeads=geometry::kIndexHeads,kIndexTop=geometry::kFlash.index_top;
inline constexpr auto kExperts=geometry::kFlash.experts,kSlots=geometry::kSlots,kIntermediate=geometry::kFlash.intermediate;
inline constexpr auto kMainEncoding=c::RowCacheEncoding::float8_block64_tail64();
inline constexpr auto kIndexEncoding=c::RowCacheEncoding::float4_block32();

using dspark_formal_fragment::require;
using dspark_formal_fragment::cast;
using dspark_formal_fragment::stored;
using dspark_formal_fragment::visible;
using dspark_formal_fragment::copy;
using dspark_formal_fragment::reshape;
using dspark_formal_fragment::indices;
using dspark_formal_fragment::index;
using dspark_formal_fragment::range;
using dspark_formal_fragment::scatter;
using dspark_formal_fragment::gather;
using dspark_formal_fragment::decode;
using dspark_formal_fragment::encode;
using dspark_formal_fragment::binary;
using dspark_formal_fragment::sum;

struct Plan final {
  std::int64_t position,capacity;
  Geometry geometry=::dspark_target_geometry::kFlash;
  static Plan make(std::int64_t position,std::int64_t capacity,Geometry model=::dspark_target_geometry::kFlash) {
    model.validate();
    // A short prefix may have no compressed rows. Its cold graph omits that
    // reader/selection; the physical state carriers remain positively sized.
    if(position<0||capacity<128||capacity<=position||capacity>std::numeric_limits<std::int32_t>::max()-1)
      throw std::runtime_error("target decode plan needs a bounded actual prefix frontier");
    return {position,capacity,model};
  }
  std::int64_t pool_capacity(std::int64_t ratio) const {return capacity/ratio;}
  std::int64_t visible(std::int64_t ratio) const {return (position+1)/ratio;}
  bool commits(std::int64_t ratio) const {return (position+1)%ratio==0;}
};
void require_ratio(std::int64_t ratio);
struct IndexState final {
  Value raw,score,pool;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(raw);visitor(score);visitor(pool);}
};
struct LayerState final {
  Value window,raw,score,pool;
  std::optional<IndexState> index;
  Value cursor;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(window);visitor(raw);visitor(score);visitor(pool);
    if(index)index->visit_values(visitor);visitor(cursor);
  }
};
std::vector<c::TensorDesc> descriptions(std::int64_t ratio,std::int64_t capacity,
    const Geometry &model=geometry::kFlash);
LayerState state_inputs(sem::Fragment &f,std::int64_t ratio,std::int64_t capacity,
    const Geometry &model=geometry::kFlash);
void require_state(const sem::Fragment &f,LayerState state,std::int64_t ratio,const Plan &plan);

using dspark_formal_fragment::NormPorts;
using dspark_formal_fragment::norm;
using dspark_formal_fragment::ProjectionPorts;
using dspark_formal_fragment::fp8;

struct CompressorParameters final {Value key,gate,ape,norm;};
struct IndexerParameters final {Value query,weights;CompressorParameters compressor;};
struct AttentionParameters final {
  Value query_a,query_norm,query_b,kv,kv_norm,output_a,output_b,sinks;
  CompressorParameters compressor;
  std::optional<IndexerParameters> indexer;
};
struct PooledPorts final {
  Value values,scores,probabilities,weighted,pooled_raw,pooled_stored;
  NormPorts norm;
  ds::RopePorts rotary;
  std::optional<ProjectionPorts> hadamard;
  Value encoded;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&values,&scores,&probabilities,&weighted,&pooled_raw,&pooled_stored})visitor(*port);
    norm.visit_values(visitor);rotary.visit_values(visitor);if(hadamard)hadamard->visit_values(visitor);visitor(encoded);
  }
};
struct CompressorPorts final {
  Value input,key_raw,gate_raw,ape,score,raw_updated,score_updated;
  std::optional<PooledPorts> pooled;
  Value raw_next,score_next,pool_next;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&key_raw,&gate_raw,&ape,&score,&raw_updated,&score_updated})visitor(*port);
    if(pooled)pooled->visit_values(visitor);
    for(auto *port:{&raw_next,&score_next,&pool_next})visitor(*port);
  }
};
CompressorPorts append_compressor(sem::Fragment &f,Value input,Value old_raw,Value old_score,Value old_pool,
    const CompressorParameters &parameters,std::int64_t ratio,bool indexer,const Plan &plan);

struct IndexSelectionPorts final {
  ProjectionPorts query,weights;
  ds::RopePorts rotary;
  Value hadamard_raw,hadamard_stored,query_qat,scaled_weights_raw,scaled_weights_stored,scores,ids;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    query.visit_values(visitor);weights.visit_values(visitor);rotary.visit_values(visitor);
    for(auto *port:{&hadamard_raw,&hadamard_stored,&query_qat,&scaled_weights_raw,&scaled_weights_stored,&scores,&ids})visitor(*port);
  }
};
struct IndexerPorts final {
  CompressorPorts compressor;
  std::optional<IndexSelectionPorts> selection;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    if(selection)selection->visit_values(visitor);
    compressor.visit_values(visitor);
  }
};
struct CompressedReadPorts final {
  Value ids,decoded;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(ids);visitor(decoded);}
};
struct AttentionPorts final {
  Value input;
  ProjectionPorts query_a,query_b,kv,output_b;
  NormPorts query_norm,kv_norm;
  ds::QueryNormPorts query_head_norm;
  ds::RopePorts query_rotary,kv_rotary,inverse_rotary;
  Value window_encoded,window_next,window_rows,window_decoded;
  std::optional<CompressedReadPorts> compressed;
  Value chosen,mask,context_raw,context_stored,output_a_input,output_a_stored;
  CompressorPorts compressor;
  std::optional<IndexerPorts> indexer;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(input);query_a.visit_values(visitor);query_b.visit_values(visitor);kv.visit_values(visitor);output_b.visit_values(visitor);
    query_norm.visit_values(visitor);kv_norm.visit_values(visitor);query_head_norm.visit_values(visitor);
    query_rotary.visit_values(visitor);kv_rotary.visit_values(visitor);inverse_rotary.visit_values(visitor);
    for(auto *port:{&window_encoded,&window_next,&window_rows,&window_decoded})visitor(*port);
    if(compressed)compressed->visit_values(visitor);
    for(auto *port:{&chosen,&mask,&context_raw,&context_stored,&output_a_input,&output_a_stored})visitor(*port);
    compressor.visit_values(visitor);if(indexer)indexer->visit_values(visitor);
  }
};
struct AttentionResult final {Value output;LayerState state;AttentionPorts ports;};
AttentionResult append_attention(sem::Fragment &f,Value input,LayerState state,
    const AttentionParameters &parameters,std::int64_t ratio,const Plan &plan);

using dspark_formal_fragment::HcParameters;
using dspark_formal_fragment::MoeParameters;
using dspark_formal_fragment::Boundary;
using dspark_formal_fragment::MoeProjectionPorts;
using dspark_formal_fragment::MoePorts;
using dspark_formal_fragment::MoeResult;
using dspark_formal_fragment::append_moe;

inline void visit_hc(ds::HyperPrePorts &ports,auto &&visitor) {
  for(auto *port:{&ports.state,&ports.flat,&ports.projection,&ports.inverse,&ports.mixes,&ports.pre,&ports.post,
      &ports.combination,&ports.collapsed_raw,&ports.collapsed})visitor(*port);
}
inline void visit_post(ds::HyperPostPorts &ports,auto &&visitor) {
  for(auto *port:{&ports.branch,&ports.residual,&ports.expanded,&ports.stored})visitor(*port);
}
struct LayerParameters final {HcParameters attention_hc,ffn_hc;AttentionParameters attention;MoeParameters moe;};
struct LayerPorts final {
  ds::HyperPrePorts attention_hc,ffn_hc;
  ds::HyperPostPorts attention_post,ffn_post;
  NormPorts attention_norm,ffn_norm;
  AttentionPorts attention;
  MoePorts moe;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visit_hc(attention_hc,visitor);visit_hc(ffn_hc,visitor);visit_post(attention_post,visitor);visit_post(ffn_post,visitor);
    attention_norm.visit_values(visitor);ffn_norm.visit_values(visitor);attention.visit_values(visitor);moe.visit_values(visitor);
  }
};
struct LayerResult final {Value output,capture;LayerState state;LayerPorts ports;Boundary boundary;};
LayerResult append_layer(sem::Fragment &f,Value input,LayerState state,const LayerParameters &parameters,
    std::int64_t ratio,const Plan &plan);

using dspark_formal_fragment::HeadParameters;
using dspark_formal_fragment::HeadPorts;
using dspark_formal_fragment::HeadResult;
using dspark_formal_fragment::append_head;

} // namespace dspark_original_target_fragment
