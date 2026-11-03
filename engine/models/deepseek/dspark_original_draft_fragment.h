#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "dspark_formal_fragment.h"
#include "dspark_vocabulary_fragment.h"

// The fixed formal DSpark forward_embed/three stages/forward_head expressed
// as ordinary Values. Accepted main histories are inputs; proposal KV remains
// episode-local. Loading, resident uploads, execution and StateHead policy
// belong to the existing consumer and are deliberately absent here.
namespace dspark_original_draft_fragment {
namespace math=dspark_formal_fragment;
namespace ds=math::ds;
namespace core=math::core;
namespace sem=math::sem;
namespace c=math::c;
using Value=math::Value;
using Geometry=math::Geometry;
inline constexpr std::int64_t kRows=5,kVocabulary=129280,kNoiseToken=128799;
inline constexpr auto kWindow=math::geometry::kWindow,kDepth=math::geometry::kDepth;
inline constexpr auto kEncoding=c::RowCacheEncoding::float8_block64_tail64();

struct AttentionParameters final {Value query_a,query_norm,query_b,kv,kv_norm,output_a,output_b,sinks;};
struct HistoryPorts final {
  math::ProjectionPorts projection;
  math::NormPorts norm;
  ds::RopePorts rotary;
  Value encoded,combined,decoded;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    projection.visit_values(visitor);norm.visit_values(visitor);rotary.visit_values(visitor);
    for(auto *port:{&encoded,&combined,&decoded})visitor(*port);
  }
};
HistoryPorts proposal_keys(sem::Fragment &f,Value input,Value history,Value positions,
    const AttentionParameters &parameters,const Geometry &model);
struct AttentionPorts final {
  Value input,positions,mask,keys;
  math::ProjectionPorts query_a,query_b,output_b;
  math::NormPorts query_norm;
  ds::QueryNormPorts head_norm;
  ds::RopePorts rotary,inverse;
  HistoryPorts history;
  Value context_raw,context_stored;
  math::OutputAPorts output_a;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&positions,&mask,&keys})visitor(*port);
    query_a.visit_values(visitor);query_b.visit_values(visitor);output_b.visit_values(visitor);
    query_norm.visit_values(visitor);head_norm.visit_values(visitor);rotary.visit_values(visitor);
    inverse.visit_values(visitor);history.visit_values(visitor);
    visitor(context_raw);visitor(context_stored);output_a.visit_values(visitor);
  }
};
struct AttentionResult final {Value output;AttentionPorts ports;};
AttentionResult attention(sem::Fragment &f,Value input,Value history,Value positions,Value mask,
    const AttentionParameters &parameters,const Geometry &model);

struct StageParameters final {math::HcParameters attention_hc,ffn_hc;AttentionParameters attention;math::MoeParameters moe;};
struct Parameters final {
  Value embedding,head,markov_w1,markov_w2,confidence;
  std::array<StageParameters,3> stages;
  math::HeadParameters head_hc;
  std::int64_t markov_rank;
};
struct Inputs final {Value pending,positions,mask;std::array<Value,3> histories;};
inline void visit_hc(ds::HyperPrePorts &p,auto &&visitor) {
  for(auto *port:{&p.state,&p.flat,&p.projection,&p.inverse,&p.mixes,&p.pre,&p.post,&p.combination,&p.collapsed_raw,&p.collapsed})visitor(*port);
}
inline void visit_post(ds::HyperPostPorts &p,auto &&visitor) {
  for(auto *port:{&p.branch,&p.residual,&p.expanded,&p.stored})visitor(*port);
}
struct StagePorts final {
  ds::HyperPrePorts attention_hc,ffn_hc;
  ds::HyperPostPorts attention_post,ffn_post;
  math::NormPorts attention_norm,ffn_norm;
  AttentionPorts attention;
  math::MoePorts moe;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visit_hc(attention_hc,visitor);visit_hc(ffn_hc,visitor);visit_post(attention_post,visitor);visit_post(ffn_post,visitor);
    attention_norm.visit_values(visitor);ffn_norm.visit_values(visitor);attention.visit_values(visitor);moe.visit_values(visitor);
  }
};
struct HeadPorts final {
  math::HeadResult hc;
  Value base_logits;
  std::array<Value,kRows> markov,bias,adjusted,top_scores,top_ids;
  Value confidence_input;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    hc.ports.visit_values(visitor);visitor(hc.normalized);visitor(base_logits);visitor(confidence_input);
    for(std::size_t row=0;row<std::size_t(kRows);++row)
      for(auto *port:{&markov[row],&bias[row],&adjusted[row],&top_scores[row],&top_ids[row]})visitor(*port);
  }
};
struct Result final {
  Inputs inputs;
  Value state,hidden,normalized,ids,markov,logits,confidence;
  Value embedding_ids,embedding;
  std::array<StagePorts,3> stages;
  HeadPorts head;
  std::array<math::Boundary,3> routers;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&state,&hidden,&normalized,&ids,&markov,&logits,&confidence,&embedding_ids,&embedding})visitor(*port);
    for(auto *port:{&inputs.pending,&inputs.positions,&inputs.mask,&inputs.histories[0],&inputs.histories[1],&inputs.histories[2]})visitor(*port);
    for(auto &stage:stages)stage.visit_values(visitor);head.visit_values(visitor);
  }
};
Result append(sem::Fragment &f,const Inputs &input,const Parameters &parameters,const Geometry &model);

struct MainParameters final {Value projection,norm;};
struct MainPorts final {
  std::array<Value,3> captures,mean_raw,mean_stored;
  Value joined;
  math::ProjectionPorts projection;
  math::NormPorts norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(std::size_t stage=0;stage<3;++stage)for(auto *port:{&captures[stage],&mean_raw[stage],&mean_stored[stage]})visitor(*port);
    visitor(joined);projection.visit_values(visitor);norm.visit_values(visitor);
  }
};
MainPorts main_projection(sem::Fragment &f,std::array<Value,3> captures,
    const MainParameters &parameters,const Geometry &model);
struct MainHistoryParameters final {Value projection,norm;};
struct MainHistoryPorts final {
  Value main,positions,write_rows;
  std::array<Value,3> histories;
  std::array<math::ProjectionPorts,3> projection;
  std::array<math::NormPorts,3> norm;
  std::array<ds::RopePorts,3> rotary;
  std::array<Value,3> encoded;
  std::vector<std::array<Value,4>> checkpoints;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(main);visitor(positions);visitor(write_rows);
    for(auto &history:histories)visitor(history);
    for(std::size_t stage=0;stage<3;++stage) {
      projection[stage].visit_values(visitor);norm[stage].visit_values(visitor);
      rotary[stage].visit_values(visitor);visitor(encoded[stage]);
    }
    for(auto &snapshot:checkpoints)for(auto &value:snapshot)visitor(value);
  }
};
// positions/write_rows are ordinary validated metadata inputs. Every snapshot
// owns only actual native outputs; no proposed row is persisted here.
MainHistoryPorts main_history(sem::Fragment &f,Value main,Value positions,Value write_rows,
    std::array<Value,3> histories,const std::array<MainHistoryParameters,3> &parameters,const Geometry &model);
} // namespace dspark_original_draft_fragment
