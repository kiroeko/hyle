#include "dspark_original_draft_fragment.h"

namespace dspark_original_draft_fragment {
HistoryPorts proposal_keys(sem::Fragment &f,Value input,Value history,Value positions,
    const AttentionParameters &parameters,const Geometry &model) {
  math::require(f,input,sem::dense({kRows,model.hidden},c::DataType::kBFloat16));
  math::require(f,history,c::row_cache_carrier_description(kEncoding,kWindow).value());
  math::require(f,positions,sem::dense({kRows},c::DataType::kInt32));
  HistoryPorts result;
  result.projection=math::fp8(f,input,parameters.kv,kDepth,model.hidden);
  result.norm=math::norm(f,result.projection.stored,parameters.kv_norm);
  const auto rotated=ds::rope(f,math::reshape(f,math::visible(f,result.norm.stored),{1,kRows,1,kDepth}),
      positions,false,false,&result.rotary);
  result.encoded=math::encode(f,math::reshape(f,rotated,{kRows,kDepth}),kEncoding);
  result.combined=ds::concat(f,history,result.encoded,0);
  result.decoded=math::decode(f,result.combined,math::range(f,kWindow+kRows),kEncoding);
  return result;
}
}

namespace dspark_original_draft_fragment {
AttentionResult attention(sem::Fragment &f,Value input,Value history,Value positions,Value mask,
    const AttentionParameters &parameters,const Geometry &model) {
  model.validate();math::require(f,input,sem::dense({kRows,model.hidden},c::DataType::kBFloat16));
  math::require(f,mask,sem::dense({kRows,kWindow+kRows}));
  math::require(f,parameters.sinks,sem::dense({model.heads}));
  AttentionPorts p;p.input=input;p.positions=positions;p.mask=mask;
  p.query_a=math::fp8(f,input,parameters.query_a,model.query_rank,model.hidden);
  p.query_norm=math::norm(f,p.query_a.stored,parameters.query_norm);
  p.query_b=math::fp8(f,p.query_norm.stored,parameters.query_b,model.heads*kDepth,model.query_rank);
  const auto q=ds::query_norm(f,math::reshape(f,math::visible(f,p.query_b.stored),{1,kRows,model.heads,kDepth}),&p.head_norm);
  const auto rotated=ds::rope(f,q,positions,false,false,&p.rotary);
  p.history=proposal_keys(f,input,history,positions,parameters,model);
  p.keys=math::reshape(f,math::visible(f,p.history.decoded),{1,kWindow+kRows,1,kDepth});
  p.context_raw=f.one(c::model_ops::sink_attention(1.F/std::sqrt(float(kDepth))),
      {rotated,p.keys,p.keys,mask,parameters.sinks});
  p.context_stored=math::stored(f,p.context_raw);
  const auto inverse=ds::rope(f,math::visible(f,p.context_stored),positions,false,true,&p.inverse);
  p.output_a=math::output_a(f,math::stored(f,inverse),parameters.output_a,kRows,model);
  p.output_b=math::fp8(f,math::reshape(f,p.output_a.stored,{kRows,model.groups*math::geometry::kOutputRank}),
      parameters.output_b,model.hidden,model.groups*math::geometry::kOutputRank);
  return {p.output_b.stored,p};
}
}

namespace dspark_original_draft_fragment {
Result append(sem::Fragment &f,const Inputs &input,const Parameters &parameters,const Geometry &model) {
  model.validate();
  if(parameters.markov_rank<=0||parameters.markov_rank>std::numeric_limits<std::int32_t>::max()-model.hidden)
    throw std::runtime_error("formal DSpark Markov/confidence geometry exceeds I32");
  math::require(f,input.pending,sem::dense({1},c::DataType::kInt32));
  math::require(f,input.positions,sem::dense({kRows},c::DataType::kInt32));
  math::require(f,input.mask,sem::dense({kRows,kWindow+kRows}));
  for(auto history:input.histories)math::require(f,history,c::row_cache_carrier_description(kEncoding,kWindow).value());
  dspark_vocabulary_fragment::require_embedding(f.description(parameters.embedding),kVocabulary,model.hidden);
  dspark_vocabulary_fragment::require_head(f.description(parameters.head),kVocabulary,model.hidden);
  math::require(f,parameters.markov_w1,sem::dense({kVocabulary,parameters.markov_rank},c::DataType::kBFloat16));
  math::require(f,parameters.markov_w2,sem::dense({kVocabulary,parameters.markov_rank},c::DataType::kBFloat16));
  math::require(f,parameters.confidence,sem::dense({1,model.hidden+parameters.markov_rank},c::DataType::kBFloat16));
  Result result;result.inputs=input;
  const auto noise=math::indices(f,std::array<std::int32_t,kRows-1>{kNoiseToken,kNoiseToken,kNoiseToken,kNoiseToken});
  result.embedding_ids=ds::concat(f,input.pending,noise,0);
  result.embedding=dspark_vocabulary_fragment::append_embedding(f,parameters.embedding,result.embedding_ids,kVocabulary,model.hidden).stored;
  auto state=math::stored(f,math::binary(f,c::BinaryKind::kMultiply,
      math::reshape(f,math::visible(f,result.embedding),{kRows,1,model.hidden}),core::filled(f,sem::dense({1,4,1}),1.F)));
  for(std::size_t stage=0;stage<3;++stage) {
    const auto &parameter=parameters.stages[stage];auto &p=result.stages[stage];
    const auto actual=math::visible(f,state);
    const auto pre=ds::hyper_pre(f,actual,model.hidden,parameter.attention_hc.mixing,true);
    p.attention_hc=pre.ports;p.attention_norm=math::norm(f,pre.collapsed,parameter.attention_hc.norm);
    const auto attended=attention(f,p.attention_norm.stored,input.histories[stage],input.positions,input.mask,parameter.attention,model);
    p.attention=attended.ports;
    const auto after=ds::hyper_post(f,pre,actual,math::visible(f,attended.output),&p.attention_post);
    const auto ffn=ds::hyper_pre(f,after,model.hidden,parameter.ffn_hc.mixing,true);
    p.ffn_hc=ffn.ports;p.ffn_norm=math::norm(f,ffn.collapsed,parameter.ffn_hc.norm);
    const auto routed=math::append_moe(f,p.ffn_norm.stored,parameter.moe,model);
    p.moe=routed.ports;result.routers[stage]=routed.boundary;
    state=math::stored(f,ds::hyper_post(f,ffn,after,math::visible(f,routed.output),&p.ffn_post));
  }
  result.state=state;result.head.hc=math::append_head(f,state,parameters.head_hc);
  result.hidden=result.head.hc.ports.collapsed_stored;result.normalized=result.head.hc.normalized;
  result.head.base_logits=dspark_vocabulary_fragment::append_head(f,result.normalized,parameters.head,kVocabulary,model.hidden);
  std::array<Value,kRows+1> ids;ids[0]=input.pending;
  for(std::size_t row=0;row<std::size_t(kRows);++row) {
    result.head.markov[row]=math::gather(f,parameters.markov_w1,ids[row]);
    result.head.bias[row]=f.one(c::model_ops::dense_linear_f32(),{result.head.markov[row],parameters.markov_w2});
    result.head.adjusted[row]=math::binary(f,c::BinaryKind::kAdd,
        ds::slice(f,result.head.base_logits,0,std::int64_t(row),1),result.head.bias[row]);
    const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{result.head.adjusted[row]},2);
    result.head.top_scores[row]=top[0];result.head.top_ids[row]=math::reshape(f,top[1],{1});ids[row+1]=result.head.top_ids[row];
  }
  const auto concatenate=[&](const auto &values) {auto value=values[0];for(std::size_t i=1;i<values.size();++i)value=ds::concat(f,value,values[i],0);return value;};
  result.ids=concatenate(ids);result.markov=concatenate(result.head.markov);result.logits=concatenate(result.head.adjusted);
  result.head.confidence_input=ds::concat(f,result.hidden,result.markov,1);
  result.confidence=f.one(c::model_ops::dense_linear_f32(),{result.head.confidence_input,parameters.confidence});
  return result;
}
}

namespace dspark_original_draft_fragment {
MainPorts main_projection(sem::Fragment &f,std::array<Value,3> captures,
    const MainParameters &parameters,const Geometry &model) {
  model.validate();
  if(captures[0].ordinal()>=f.descriptions.size()||f.description(captures[0]).rank()!=3)
    throw std::runtime_error("formal DSpark main projection needs actual captured rows");
  const auto rows=f.description(captures[0]).dimensions()[0];MainPorts result;result.captures=captures;
  for(std::size_t stage=0;stage<3;++stage) {
    math::require(f,captures[stage],sem::dense({rows,4,model.hidden}));
    const auto actual=math::visible(f,math::stored(f,captures[stage]));
    result.mean_raw[stage]=f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kMean,
        .axes=c::Axes::from(std::array<std::int32_t,1>{1}).value(),.keep_dimensions=false,
        .accumulation_type=c::DataType::kFloat32}),{actual});
    result.mean_stored[stage]=math::stored(f,result.mean_raw[stage]);
  }
  result.joined=ds::concat(f,ds::concat(f,result.mean_stored[0],result.mean_stored[1],1),result.mean_stored[2],1);
  result.projection=math::fp8(f,result.joined,parameters.projection,model.hidden,3*model.hidden);
  result.norm=math::norm(f,result.projection.stored,parameters.norm);return result;
}
}

namespace dspark_original_draft_fragment {
MainHistoryPorts main_history(sem::Fragment &f,Value main,Value positions,Value write_rows,
    std::array<Value,3> histories,const std::array<MainHistoryParameters,3> &parameters,const Geometry &model) {
  model.validate();
  if(main.ordinal()>=f.descriptions.size()||f.description(main).rank()!=2)
    throw std::runtime_error("formal DSpark main history needs actual main rows");
  const auto rows=f.description(main).dimensions()[0];
  math::require(f,main,sem::dense({rows,model.hidden},c::DataType::kBFloat16));
  math::require(f,positions,sem::dense({rows},c::DataType::kInt32));
  math::require(f,write_rows,sem::dense({rows},c::DataType::kInt32));
  MainHistoryPorts result;result.main=main;result.positions=positions;result.write_rows=write_rows;result.histories=histories;
  for(std::size_t stage=0;stage<3;++stage) {
    math::require(f,histories[stage],c::row_cache_carrier_description(kEncoding,kWindow).value());
    result.projection[stage]=math::fp8(f,main,parameters[stage].projection,kDepth,model.hidden);
    result.norm[stage]=math::norm(f,result.projection[stage].stored,parameters[stage].norm);
    const auto rotated=ds::rope(f,math::reshape(f,math::visible(f,result.norm[stage].stored),{1,rows,1,kDepth}),positions,false,false,&result.rotary[stage]);
    result.encoded[stage]=math::encode(f,math::reshape(f,rotated,{rows,kDepth}),kEncoding);
  }
  for(std::int64_t row=0;row<rows;++row) {
    const auto destination=ds::slice(f,write_rows,0,row,1);
    for(std::size_t stage=0;stage<3;++stage)
      histories[stage]=math::scatter(f,histories[stage],destination,ds::slice(f,result.encoded[stage],0,row,1));
    result.checkpoints.push_back({histories[0],histories[1],histories[2],math::visible(f,ds::slice(f,main,0,row,1))});
  }
  return result;
}
}
