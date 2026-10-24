#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dspark_original_draft_program.h"
#include "dspark_original_target_reference.h"
#include "glm_shared_vocabulary_reference.h"

// Actual-predecessor references for the formal draft. Original E4/UE8 and
// FP4/group32 certificates, HC and row-cache codecs are shared with the
// trained target. BF16 LM/Markov/confidence dots use the existing independent
// full-vocabulary certificate. No reference tensor becomes a Graph input.
namespace dspark_original_draft_reference {
namespace graph=dspark_original_draft_fragment;
namespace common=dspark_formal_fragment;
namespace roles=dspark_original_formal_roles;
namespace target=dspark_original_target_reference;
namespace fp8=glm_original_attention_reference;
namespace hc=dspark_original_hc_reference;
namespace lm=glm_shared_vocabulary_reference;
using Value=graph::Value;
using F32=std::span<const float>;
using I32=std::span<const std::int32_t>;
using BF16=std::span<const std::uint16_t>;

inline std::string name(const roles::Data &data,std::size_t stage,std::string_view suffix) {
  return roles::layer_name(data.model(),roles::Section::kDraft,stage,suffix);
}
inline void projection(const roles::Data &data,std::string_view role,F32 actual,F32 input,bool full) {
  const auto matrix=data.matrix(role);const auto source=target::bits(input);
  const auto n=std::size_t(matrix.outputs),k=std::size_t(matrix.reduction);
  target::need(k&&source.size()%k==0&&actual.size()==source.size()/k*n,"formal draft FP8 projection extent");
  const std::string label(role);
  for(std::size_t row=0;row<source.size()/k;++row) {
    SCOPED_TRACE(::testing::Message()<<label<<" token="<<row);
    EXPECT_TRUE(fp8::expect_fp8_matrix(actual.subspan(row*n,n),BF16(source).subspan(row*k,k),
        matrix.codes,std::as_bytes(matrix.scales),n,k,label.c_str(),full,
        graph::c::model_ops::Float8ScalePolicy::kPowerOfTwo,1.e-4F,true));
  }
}
inline void bfloat_projection(F32 actual,F32 input,std::span<const std::byte> weight,
    std::size_t outputs,bool full,I32 selected={}) {
  target::need(outputs>0&&!input.empty()&&weight.size()%(outputs*2)==0,"formal BF16 dot source extent");
  const auto width=weight.size()/(outputs*2);const auto source=target::bits(input);
  EXPECT_TRUE(lm::expect_logits(actual,source,weight,outputs,width,full,selected));
}
template<class Get> void norm(const roles::Data &data,std::string_view role,const common::NormPorts &p,Get get) {
  target::norm(get(p.input),get(p.raw),get(p.stored),data.norm(role),"formal draft actual RMS/BF16");
}
template<class Get> void hc_pre(const roles::Data &data,std::size_t stage,std::string_view branch,
    const graph::ds::HyperPrePorts &p,Get get) {
  const std::string prefix="hc_"+std::string(branch);
  target::hc_pre(p,data.floats(name(data,stage,prefix+"_fn")),data.floats(name(data,stage,prefix+"_scale")),
      data.floats(name(data,stage,prefix+"_base")),get);
}

template<class Get,class GetBytes> void attention(const roles::Data &data,std::size_t stage,
    const graph::AttentionPorts &p,Value old_history,Get get,GetBytes get_bytes,bool full) {
  const auto &model=data.geometry();const auto positions=target::ids(get_bytes(p.positions));
  target::need(positions.size()==std::size_t(graph::kRows),"formal draft actual rotary position rows");
  for(std::size_t row=0;row<positions.size();++row)
    ASSERT_EQ(positions[row],positions[0]+std::int32_t(row))<<"draft absolute position sequence";
  projection(data,name(data,stage,"attn.wq_a.weight"),get(p.query_a.stored),get(p.query_a.input),full);
  target::exact(get(p.query_a.input),get(p.input),"formal draft QA actual input");
  target::exact(get(p.query_norm.input),get(p.query_a.stored),"formal draft QA to learned norm");
  norm(data,name(data,stage,"attn.q_norm.weight"),p.query_norm,get);
  projection(data,name(data,stage,"attn.wq_b.weight"),get(p.query_b.stored),get(p.query_b.input),full);
  target::exact(get(p.query_b.input),get(p.query_norm.stored),"formal draft QB actual input");
  target::exact(get(p.head_norm.input),get(p.query_b.stored),"formal draft per-head norm actual input");
  target::query_norm(p.head_norm,get);target::exact(get(p.rotary.input),get(p.head_norm.output),"formal draft query rotary input");
  target::rope_rows(p.rotary,graph::kDepth,positions,false,false,get);
  const auto &h=p.history;
  projection(data,name(data,stage,"attn.wkv.weight"),get(h.projection.stored),get(h.projection.input),full);
  target::exact(get(h.projection.input),get(p.input),"formal draft same KV activation");
  target::exact(get(h.norm.input),get(h.projection.stored),"formal draft KV learned norm input");
  norm(data,name(data,stage,"attn.kv_norm.weight"),h.norm,get);
  target::exact(get(h.rotary.input),get(h.norm.stored),"formal draft KV rotary input");
  target::rope_rows(h.rotary,graph::kDepth,positions,false,false,get);
  const auto proposed=get_bytes(h.encoded),old=get_bytes(old_history),combined=get_bytes(h.combined);
  target::encoded_equal(proposed,target::encoded(graph::kEncoding,get(h.rotary.stored)));
  ASSERT_EQ(old.size(),std::size_t(graph::kWindow)*graph::kEncoding.row_bytes());
  ASSERT_EQ(combined.size(),old.size()+proposed.size());
  EXPECT_TRUE(std::equal(old.begin(),old.end(),combined.begin()));
  EXPECT_TRUE(std::equal(proposed.begin(),proposed.end(),combined.begin()+std::ptrdiff_t(old.size())));
  std::vector<std::int32_t> rows(std::size_t(graph::kWindow+graph::kRows));std::iota(rows.begin(),rows.end(),0);
  target::exact(get(h.decoded),target::decoded(graph::kEncoding,combined,rows),"formal actual main plus temporary proposal reader");
  const auto keys=get(p.keys),mask=get(p.mask);
  target::exact(keys,get(h.decoded),"formal draft SAME decoded K/V rows");
  const auto context=get(p.context_raw);
  target::close(context,graph::ds::old::sink_reference(get(p.rotary.stored),keys,keys,mask,
      data.floats(name(data,stage,"attn.attn_sink")),graph::kRows,graph::kWindow+graph::kRows,
      1.F/std::sqrt(float(graph::kDepth))),"formal draft sink attention actual operands");
  target::store(get(p.context_stored),context,"formal draft attention BF16 return");
  target::exact(get(p.inverse.input),get(p.context_stored),"formal draft inverse rotary predecessor");
  target::rope_rows(p.inverse,graph::kDepth,positions,false,true,get);
  const auto inverse=get(p.inverse.stored),lhs=get(p.output_a.input);
  const auto groups=std::size_t(model.groups),width=std::size_t(model.group_width()),tokens=std::size_t(graph::kRows);
  std::vector<float> reordered(inverse.size());
  target::need(inverse.size()==tokens*groups*width&&lhs.size()==inverse.size(),"formal draft OA actual batch grouping");
  for(std::size_t group=0;group<groups;++group)for(std::size_t row=0;row<tokens;++row)
    std::copy_n(inverse.begin()+std::ptrdiff_t((row*groups+group)*width),width,
        reordered.begin()+std::ptrdiff_t((group*tokens+row)*width));
  target::exact(lhs,reordered,"formal draft OA actual group-major activation");
  target::output_a_matrix(data.matrix(name(data,stage,"attn.wo_a.weight")),get(p.output_a.stored),lhs,full,model);
  target::exact(get(p.output_b.input),get(p.output_a.stored),"formal draft OB actual token-major predecessor");
  projection(data,name(data,stage,"attn.wo_b.weight"),get(p.output_b.stored),get(p.output_b.input),full);
}

template<class Get,class GetBytes> void moe(const roles::Data &data,std::size_t stage,
    const common::MoePorts &p,Get get,GetBytes get_bytes,bool full) {
  const auto &model=data.geometry();const auto hidden=std::size_t(model.hidden),intermediate=std::size_t(model.intermediate);
  const auto input=get(p.input),logits=get(p.logits);
  const auto global=target::ids(get_bytes(p.ids));
  bfloat_projection(logits,input,data.raw(name(data,stage,"ffn.gate.weight")),std::size_t(model.experts),true);
  dspark_original_expert_reference::expect_routing({logits,get(p.softplus),get(p.scores),get(p.selection),
      get(p.selected),get(p.sum),get(p.normalized),get(p.weights),global},data.floats(name(data,stage,"ffn.gate.bias")),
      model.route_scale,"formal Pro draft Top6",std::size_t(model.experts),std::size_t(common::geometry::kSlots));
  const auto mapped=target::ids(get_bytes(p.map)),local=target::ids(get_bytes(p.local_ids));
  target::need(mapped.size()==std::size_t(model.experts)&&global.size()==std::size_t(graph::kRows*6)&&local.size()==global.size(),"formal draft actual global/map extent");
  auto unique=global;std::ranges::sort(unique);unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
  for(std::size_t id=0;id<mapped.size();++id) {
    const auto at=std::lower_bound(unique.begin(),unique.end(),std::int32_t(id));
    EXPECT_EQ(mapped[id],at!=unique.end()&&*at==std::int32_t(id)?std::int32_t(at-unique.begin()):-1);
  }
  for(std::size_t slot=0;slot<global.size();++slot)EXPECT_EQ(local[slot],mapped.at(std::size_t(global[slot])));
  const auto gate=get(p.routed.gate_stored),up=get(p.routed.up_stored),down=get(p.routed.down_stored),weighted=get(p.weighted_stored);
  target::need(gate.size()==global.size()*intermediate&&up.size()==gate.size()&&down.size()==global.size()*hidden&&
      weighted.size()==gate.size(),"formal draft routed projections actual extent");
  for(std::size_t slot=0;slot<global.size();++slot)for(std::size_t leg=0;leg<3;++leg) {
    const auto n=leg==2?hidden:intermediate,k=leg==2?intermediate:hidden;
    const auto part=static_cast<roles::Leg>(leg);
    target::fp4_projection(F32(leg==0?gate:(leg==1?up:down)).subspan(slot*n,n),
        leg==2?F32(weighted).subspan(slot*k,k):F32(input).subspan((slot/6)*hidden,hidden),
        data.expert_codes(roles::Section::kDraft,stage,part,global[slot]),
        data.expert_scales(roles::Section::kDraft,stage,part,global[slot]),n,k,full);
  }
  const auto activate=[&](const common::MoeProjectionPorts &branch) {
    const auto g=get(branch.gate_stored),u=get(branch.up_stored),clipped=get(branch.gate_clipped),uc=get(branch.up_clipped),actual=get(branch.activation_raw);
    target::need(g.size()==u.size()&&g.size()==clipped.size()&&g.size()==uc.size()&&g.size()==actual.size(),"formal draft gated branch extent");
    std::vector<float> cg(g.size()),cu(g.size()),expected(g.size());
    for(std::size_t i=0;i<g.size();++i) {
      cg[i]=std::min(g[i],10.F);cu[i]=std::clamp(u[i],-10.F,10.F);
      expected[i]=float(double(clipped[i])*hc::detail::sigmoid(clipped[i])*double(uc[i]));
    }
    target::exact(clipped,cg,"formal draft gate clip10");target::exact(uc,cu,"formal draft up clip10");
    target::close(actual,expected,"formal draft clipped SwiGLU actual predecessors");
  };
  activate(p.routed);activate(p.shared);
  const auto activation=get(p.routed.activation_raw),weights=get(p.weights),weighted_raw=get(p.weighted_raw);
  std::vector<float> expected_weighted(activation.size());
  target::need(weights.size()==global.size()&&weighted_raw.size()==activation.size(),"formal draft down weight boundary extent");
  for(std::size_t i=0;i<activation.size();++i)expected_weighted[i]=float(double(activation[i])*double(weights[i/intermediate]));
  target::exact(weighted_raw,expected_weighted,"formal draft weights BEFORE down-input BF16");
  target::store(weighted,weighted_raw,"formal draft weighted down-input BF16");
  target::exact(get(p.shared_gate.input),input,"formal draft shared gate SAME actual input");
  target::exact(get(p.shared_up.input),input,"formal draft shared up SAME actual input");
  projection(data,name(data,stage,"ffn.shared_experts.w1.weight"),get(p.shared_gate.stored),get(p.shared_gate.input),full);
  projection(data,name(data,stage,"ffn.shared_experts.w3.weight"),get(p.shared_up.stored),get(p.shared_up.input),full);
  target::store(get(p.shared_activation_stored),get(p.shared.activation_raw),"formal draft shared activation BF16");
  target::exact(get(p.shared_down.input),get(p.shared_activation_stored),"formal draft shared down actual BF16 predecessor");
  projection(data,name(data,stage,"ffn.shared_experts.w2.weight"),get(p.shared_down.stored),get(p.shared_down.input),full);
  const auto sum=get(p.routed_sum),shared=get(p.shared.down_stored),merged=get(p.merged_raw);
  target::reduction(sum,6,[&](std::size_t output,std::size_t slot) {return down[((output/hidden)*6+slot)*hidden+output%hidden];},"formal draft actual six expert sum F32");
  std::vector<float> expected_merge(sum.size());target::need(shared.size()==sum.size()&&merged.size()==sum.size(),"formal draft branch merge extent");
  for(std::size_t i=0;i<sum.size();++i)expected_merge[i]=float(double(sum[i])+double(shared[i]));
  target::exact(merged,expected_merge,"formal draft routed plus unscaled shared F32");target::store(get(p.merged_stored),merged,"formal draft MoE BF16 return");
}

template<class Get> void head_hc(const roles::Data &data,const common::HeadResult &head,Get get) {
  const auto &p=head.ports;const auto flat=get(p.flat),input=get(p.input),projected=get(p.projection),inverse=get(p.inverse),mixes=get(p.mixes);
  target::exact(flat,input,"formal draft final HC flattened BF16 state");
  EXPECT_TRUE(hc::expect_linear(projected,hc::linear(flat,data.floats(name(data,2,"hc_head_fn")),4),"formal draft final HC F32 fn"));
  target::need(inverse.size()==std::size_t(graph::kRows)&&mixes.size()==4*inverse.size()&&projected.size()==mixes.size(),"formal draft final HC row extent");
  target::close(inverse,hc::inverse_rms(flat,4*std::size_t(data.geometry().hidden)),"formal draft final HC inverse RMS");
  std::vector<float> expected(mixes.size());for(std::size_t i=0;i<expected.size();++i)expected[i]=float(double(projected[i])*double(inverse[i/4]));
  target::exact(mixes,expected,"formal draft final HC actual normalized mix");
  target::close(get(p.gate),hc::head_gate(mixes,data.floats(name(data,2,"hc_head_scale")),data.floats(name(data,2,"hc_head_base"))),"formal draft final four sigmoid gates");
  target::close(get(p.collapsed_raw),hc::collapse(input,get(p.gate)),"formal draft actual final HC collapse");
  target::store(get(p.collapsed_stored),get(p.collapsed_raw),"formal draft final HC BF16");
  target::exact(get(p.norm.input),get(p.collapsed_stored),"formal draft final norm predecessor");
  norm(data,name(data,2,"norm.weight"),p.norm,get);target::exact(get(head.normalized),get(p.norm.stored),"formal draft normalized fullV input");
}

template<class Get,class GetBytes> void observe(const roles::Data &data,const graph::Result &result,
    Get get,GetBytes get_bytes,bool full) {
  const auto &model=data.geometry();const auto pending=target::ids(get_bytes(result.inputs.pending));
  const auto embed_ids=target::ids(get_bytes(result.embedding_ids));
  ASSERT_EQ(pending.size(),1U);ASSERT_EQ(embed_ids.size(),std::size_t(graph::kRows));
  EXPECT_EQ(embed_ids[0],pending[0]);for(std::size_t row=1;row<embed_ids.size();++row)EXPECT_EQ(embed_ids[row],graph::kNoiseToken);
  EXPECT_TRUE(lm::expect_embedding(target::bits(get(result.embedding)),data.raw("embed.weight"),embed_ids,
      std::size_t(graph::kVocabulary),std::size_t(model.hidden)));
  std::vector<float> state(std::size_t(graph::kRows*4*model.hidden));const auto embedded=get(result.embedding);
  for(std::size_t row=0;row<std::size_t(graph::kRows);++row)for(std::size_t stream=0;stream<4;++stream)
    std::copy_n(embedded.begin()+std::ptrdiff_t(row*std::size_t(model.hidden)),std::size_t(model.hidden),
        state.begin()+std::ptrdiff_t((row*4+stream)*std::size_t(model.hidden)));
  target::exact(get(result.stages[0].attention_hc.state),state,"formal original embedding four-stream expansion");
  for(std::size_t stage=0;stage<3;++stage) {
    SCOPED_TRACE(::testing::Message()<<"formal original draft stage="<<stage);const auto &p=result.stages[stage];
    if(stage)target::exact(get(p.attention_hc.state),get(result.stages[stage-1].ffn_post.stored),"formal draft SAME previous stage BF16 output");
    hc_pre(data,stage,"attn",p.attention_hc,get);norm(data,name(data,stage,"attn_norm.weight"),p.attention_norm,get);
    target::exact(get(p.attention_norm.input),get(p.attention_hc.collapsed),"formal draft HC to attention learned norm");
    target::exact(get(p.attention.input),get(p.attention_norm.stored),"formal draft learned norm to attention");
    attention(data,stage,p.attention,result.inputs.histories[stage],get,get_bytes,full);
    target::exact(get(p.attention_post.branch),get(p.attention.output_b.stored),"formal draft attention to actual HC branch");
    target::hc_post(p.attention_post,p.attention_hc,get);
    target::exact(get(p.ffn_hc.state),get(p.attention_post.stored),"formal draft FFN actual residual");
    hc_pre(data,stage,"ffn",p.ffn_hc,get);norm(data,name(data,stage,"ffn_norm.weight"),p.ffn_norm,get);
    target::exact(get(p.ffn_norm.input),get(p.ffn_hc.collapsed),"formal draft HC to FFN learned norm");
    target::exact(get(p.moe.input),get(p.ffn_norm.stored),"formal draft learned norm to original MoE");
    moe(data,stage,p.moe,get,get_bytes,full);target::exact(get(p.ffn_post.branch),get(p.moe.merged_stored),"formal draft MoE to HC branch");
    target::hc_post(p.ffn_post,p.ffn_hc,get);
  }
  target::exact(get(result.state),get(result.stages.back().ffn_post.stored),"formal draft last four-stream state");
  target::exact(get(result.head.hc.ports.input),get(result.state),"formal draft final HC SAME actual state");
  head_hc(data,result.head.hc,get);
  target::exact(get(result.hidden),get(result.head.hc.ports.collapsed_stored),"formal draft raw HC hidden for confidence");
  target::exact(get(result.normalized),get(result.head.hc.normalized),"formal draft normalized LM hidden");
  const auto output_ids=target::ids(get_bytes(result.ids));ASSERT_EQ(output_ids.size(),std::size_t(graph::kRows+1));EXPECT_EQ(output_ids[0],pending[0]);
  const auto base=get(result.head.base_logits),hidden=get(result.hidden),normalized=get(result.normalized);
  const auto rank=std::size_t(data.spec(name(data,2,"markov_head.markov_w1.weight")).columns);
  bfloat_projection(base,normalized,data.raw("head.weight"),std::size_t(graph::kVocabulary),full,I32(output_ids).subspan(1));
  std::vector<float> scores,markov,adjusted;
  for(std::size_t row=0;row<std::size_t(graph::kRows);++row) {
    const auto actual_markov=get(result.head.markov[row]),bias=get(result.head.bias[row]),combined=get(result.head.adjusted[row]);
    const std::array<std::int32_t,1> prior{output_ids[row]};
    EXPECT_TRUE(lm::expect_embedding(target::bits(actual_markov),data.raw(name(data,2,"markov_head.markov_w1.weight")),prior,std::size_t(graph::kVocabulary),rank));
    bfloat_projection(bias,actual_markov,data.raw(name(data,2,"markov_head.markov_w2.weight")),std::size_t(graph::kVocabulary),full,I32(output_ids).subspan(row+1,1));
    ASSERT_EQ(combined.size(),std::size_t(graph::kVocabulary));std::vector<float> expected(combined.size());
    for(std::size_t token=0;token<combined.size();++token)expected[token]=float(double(base[row*combined.size()+token])+double(bias[token]));
    target::exact(combined,expected,"formal draft actual LM plus actual Markov cancellation");
    const auto actual_id=target::ids(get_bytes(result.head.top_ids[row]));ASSERT_EQ(actual_id.size(),1U);EXPECT_EQ(actual_id[0],output_ids[row+1]);
    const auto score=get(result.head.top_scores[row]);scores.insert(scores.end(),score.begin(),score.end());
    markov.insert(markov.end(),actual_markov.begin(),actual_markov.end());adjusted.insert(adjusted.end(),combined.begin(),combined.end());
  }
  target::exact(get(result.markov),markov,"formal draft actual preceding-ID Markov rows");target::exact(get(result.logits),adjusted,"formal draft complete adjusted logits");
  EXPECT_TRUE(lm::expect_greedy(adjusted,I32(output_ids).subspan(1),graph::kVocabulary,scores));
  std::vector<float> confidence_input;confidence_input.reserve(std::size_t(graph::kRows)*(std::size_t(model.hidden)+rank));
  for(std::size_t row=0;row<std::size_t(graph::kRows);++row) {
    confidence_input.insert(confidence_input.end(),hidden.begin()+std::ptrdiff_t(row*std::size_t(model.hidden)),hidden.begin()+std::ptrdiff_t((row+1)*std::size_t(model.hidden)));
    confidence_input.insert(confidence_input.end(),markov.begin()+std::ptrdiff_t(row*rank),markov.begin()+std::ptrdiff_t((row+1)*rank));
  }
  target::exact(get(result.head.confidence_input),confidence_input,"formal confidence uses raw HC hidden plus actual previous-ID Markov");
  bfloat_projection(get(result.confidence),confidence_input,data.raw(name(data,2,"confidence_head.proj.weight")),1,true);
}

template<class Get> void observe_main(const roles::Data &data,const graph::MainPorts &p,Get get,bool full) {
  const auto hidden=std::size_t(data.geometry().hidden);std::vector<float> joined;
  const auto first=get(p.captures[0]);target::need(!first.empty()&&first.size()%(4*hidden)==0,"formal capture rows extent");
  const auto rows=first.size()/(4*hidden);
  std::array<std::vector<float>,3> means;
  for(std::size_t stage=0;stage<3;++stage) {
    const auto capture=get(p.captures[stage]);(void)target::bits(capture);
    ASSERT_EQ(capture.size(),rows*4*hidden);
    target::reduction(get(p.mean_raw[stage]),4,[&](std::size_t output,std::size_t stream) {
      return capture[((output/hidden)*4+stream)*hidden+output%hidden]/4.F;
    },"formal captured BF16 four-stream mean");
    means[stage]=get(p.mean_stored[stage]);target::store(means[stage],get(p.mean_raw[stage]),"formal capture mean BF16");
  }
  joined.reserve(rows*3*hidden);
  for(std::size_t row=0;row<rows;++row)for(std::size_t stage=0;stage<3;++stage)
    joined.insert(joined.end(),means[stage].begin()+std::ptrdiff_t(row*hidden),means[stage].begin()+std::ptrdiff_t((row+1)*hidden));
  target::exact(get(p.joined),joined,"formal main projection stage-ordered actual target captures");
  target::exact(get(p.projection.input),joined,"formal main FP8 actual input");
  projection(data,name(data,0,"main_proj.weight"),get(p.projection.stored),get(p.projection.input),full);
  target::exact(get(p.norm.input),get(p.projection.stored),"formal main norm actual input");norm(data,name(data,0,"main_norm.weight"),p.norm,get);
}
template<class Get,class GetBytes> void observe_history(const roles::Data &data,const graph::MainHistoryPorts &p,
    Get get,GetBytes get_bytes,bool full) {
  const auto main=get(p.main);
  const auto positions=target::ids(get_bytes(p.positions)),writes=target::ids(get_bytes(p.write_rows));
  const auto rows=positions.size(),hidden=std::size_t(data.geometry().hidden),stride=std::size_t(graph::kEncoding.row_bytes());
  target::need(rows&&main.size()==rows*hidden&&writes.size()==rows&&p.checkpoints.size()==rows,"formal main history actual frontier extent");
  for(std::size_t row=0;row<rows;++row) {
    ASSERT_GE(positions[row],0);ASSERT_EQ(writes[row],positions[row]%graph::kWindow);
    if(row)ASSERT_EQ(positions[row],positions[row-1]+1);
  }
  for(std::size_t stage=0;stage<3;++stage) {
    const auto &projection_ports=p.projection[stage];
    target::exact(get(projection_ports.input),main,"formal main-history actual input");
    projection(data,name(data,stage,"attn.wkv.weight"),get(projection_ports.stored),get(projection_ports.input),full);
    target::exact(get(p.norm[stage].input),get(projection_ports.stored),"formal main-history learned norm predecessor");
    norm(data,name(data,stage,"attn.kv_norm.weight"),p.norm[stage],get);
    target::exact(get(p.rotary[stage].input),get(p.norm[stage].stored),"formal main-history rotary input");
    target::rope_rows(p.rotary[stage],graph::kDepth,positions,false,false,get);
    const auto encoded=get_bytes(p.encoded[stage]);target::encoded_equal(encoded,target::encoded(graph::kEncoding,get(p.rotary[stage].stored)));
    auto previous=get_bytes(p.histories[stage]);ASSERT_EQ(previous.size(),std::size_t(graph::kWindow)*stride);
    for(std::size_t row=0;row<rows;++row) {
      const auto actual=get_bytes(p.checkpoints[row][stage]);
      target::scatter_bytes(actual,previous,std::span(encoded).subspan(row*stride,stride),std::size_t(writes[row]),stride);
      previous=actual;
    }
  }
  for(std::size_t row=0;row<rows;++row)
    target::exact(get(p.checkpoints[row][3]),F32(main).subspan(row*hidden,hidden),"formal main-history actual last accepted carry");
}
} // namespace dspark_original_draft_reference
