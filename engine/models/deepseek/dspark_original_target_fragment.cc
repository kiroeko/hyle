#include "dspark_original_target_fragment.h"

namespace dspark_original_target_fragment {
void require_ratio(std::int64_t ratio) {
  if(ratio!=4&&ratio!=128)throw std::runtime_error("target layer must retain its actual4/128 compression ratio");
}
}

namespace dspark_original_target_fragment {
std::vector<c::TensorDesc> descriptions(std::int64_t ratio,std::int64_t capacity,
    const Geometry &model) {
  model.validate();require_ratio(ratio);
  if(capacity<128||capacity>std::numeric_limits<std::int32_t>::max()-1)
    throw std::runtime_error("target state capacity is outside the finite I32 domain");
  const auto rows=ratio==4?8:128,channels=ratio==4?1024:512;
  std::vector<c::TensorDesc> result{c::row_cache_carrier_description(kMainEncoding,kWindow).value(),
      sem::dense({rows,channels}),sem::dense({rows,channels}),
      c::row_cache_carrier_description(kMainEncoding,capacity/ratio).value()};
  if(ratio==4) {
    result.push_back(sem::dense({8,256}));result.push_back(sem::dense({8,256}));
    result.push_back(c::row_cache_carrier_description(kIndexEncoding,capacity/4).value());
  }
  result.push_back(sem::dense({1},c::DataType::kInt32));return result;
}
}

namespace dspark_original_target_fragment {
LayerState state_inputs(sem::Fragment &f,std::int64_t ratio,std::int64_t capacity,
    const Geometry &model) {
  const auto types=descriptions(ratio,capacity,model);std::size_t at=0;
  LayerState result{f.input(types[at++]),f.input(types[at++]),f.input(types[at++]),f.input(types[at++]),{}, {}};
  if(ratio==4)result.index=IndexState{f.input(types[at++]),f.input(types[at++]),f.input(types[at++])};
  result.cursor=f.input(types[at]);return result;
}
}

namespace dspark_original_target_fragment {
void require_state(const sem::Fragment &f,LayerState state,std::int64_t ratio,const Plan &plan) {
  require_ratio(ratio);(void)Plan::make(plan.position,plan.capacity,plan.geometry);
  if(state.index.has_value()!=(ratio==4))throw std::runtime_error("target state cannot omit or invent indexer state");
  const auto types=descriptions(ratio,plan.capacity,plan.geometry);std::size_t at=0;
  state.visit_values([&](Value &value){require(f,value,types.at(at++));});
}
}

namespace dspark_original_target_fragment {
CompressorPorts append_compressor(sem::Fragment &f,Value input,Value old_raw,Value old_score,Value old_pool,
    const CompressorParameters &parameters,std::int64_t ratio,bool indexer,const Plan &plan) {
  const auto &model=plan.geometry;model.validate();
  require_ratio(ratio);if(indexer&&ratio!=4)throw std::runtime_error("ratio128 has no indexer compressor");
  const auto depth=indexer?kIndexDepth:kDepth,channels=ratio==4?2*depth:depth;
  const std::int64_t rows=ratio==4?8:128;
  require(f,old_raw,sem::dense({rows,channels}));require(f,old_score,sem::dense({rows,channels}));
  require(f,parameters.key,sem::dense({channels,model.hidden}));require(f,parameters.gate,sem::dense({channels,model.hidden}));
  require(f,parameters.ape,sem::dense({ratio,channels}));
  const auto encoding=indexer?kIndexEncoding:kMainEncoding;
  require(f,old_pool,c::row_cache_carrier_description(encoding,plan.pool_capacity(ratio)).value());
  CompressorPorts result;result.input=visible(f,input);require(f,result.input,sem::dense({1,model.hidden}));
  const auto transpose=[&](Value value){return f.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{value});};
  result.key_raw=f.one(c::Operation(c::MatMul{}),{result.input,transpose(parameters.key)});
  result.gate_raw=f.one(c::Operation(c::MatMul{}),{result.input,transpose(parameters.gate)});
  result.ape=gather(f,parameters.ape,index(f,plan.position%ratio));
  result.score=binary(f,c::BinaryKind::kAdd,result.gate_raw,result.ape);
  const auto row=index(f,(ratio==4?ratio:0)+plan.position%ratio);
  result.raw_updated=scatter(f,old_raw,row,result.key_raw);
  result.score_updated=scatter(f,old_score,row,result.score);
  result.raw_next=result.raw_updated;result.score_next=result.score_updated;result.pool_next=old_pool;
  if(plan.commits(ratio)) {
    PooledPorts pooled;
    if(ratio==4) {
      const auto values=ds::slice(f,result.raw_updated,0,0,4),scores=ds::slice(f,result.score_updated,0,0,4);
      const auto current=ds::slice(f,result.raw_updated,0,4,4),current_scores=ds::slice(f,result.score_updated,0,4,4);
      pooled.values=ds::concat(f,ds::slice(f,values,1,0,depth),ds::slice(f,current,1,depth,depth),0);
      pooled.scores=ds::concat(f,ds::slice(f,scores,1,0,depth),ds::slice(f,current_scores,1,depth,depth),0);
      // Source decode copies the complete current double-channel block to
      // the old overlap half only AFTER pooling its old/current channels.
      result.raw_next=ds::concat(f,current,current,0);result.score_next=ds::concat(f,current_scores,current_scores,0);
    } else {pooled.values=result.raw_updated;pooled.scores=result.score_updated;}
    const auto transpose_scores=ds::permute(f,pooled.scores,{1,0});
    const auto probability=f.one(c::Operation(c::Softmax{.axis=-1}),{transpose_scores});
    pooled.probabilities=ds::permute(f,probability,{1,0});
    pooled.weighted=binary(f,c::BinaryKind::kMultiply,pooled.values,pooled.probabilities);
    pooled.pooled_raw=reshape(f,sum(f,pooled.weighted,0),{1,depth});pooled.pooled_stored=stored(f,pooled.pooled_raw);
    pooled.norm=norm(f,pooled.pooled_stored,parameters.norm);
    const auto compressed_position=index(f,plan.position+1-ratio);
    const auto rotated=ds::rope(f,reshape(f,visible(f,pooled.norm.stored),{1,1,1,depth}),compressed_position,true,false,&pooled.rotary);
    auto entry=reshape(f,rotated,{1,depth});
    if(indexer) {
      const auto raw=f.one(c::Operation(c::Hadamard{}),{entry});const auto rounded=stored(f,raw);
      pooled.hadamard=ProjectionPorts{entry,raw,rounded};entry=rounded;
    }
    pooled.encoded=encode(f,entry,encoding);
    result.pool_next=scatter(f,old_pool,index(f,plan.position/ratio),pooled.encoded);
    result.pooled=std::move(pooled);
  }
  return result;
}
}

namespace dspark_original_target_fragment {
AttentionResult append_attention(sem::Fragment &f,Value input,LayerState state,
    const AttentionParameters &parameters,std::int64_t ratio,const Plan &plan) {
  const auto &model=plan.geometry;
  require_state(f,state,ratio,plan);require(f,input,sem::dense({1,model.hidden},c::DataType::kBFloat16));
  if(parameters.indexer.has_value()!=(ratio==4))throw std::runtime_error("target ratio4 requires its distinct indexer parameters");
  require(f,parameters.sinks,sem::dense({model.heads}));
  // Prepared from the original FP8 wo_a ONCE, retaining BF16 only. Ordinary
  // batched MatMul avoids a grouped-provider F32 shadow of this large bank.
  require(f,parameters.output_a,sem::dense({model.groups,model.group_width(),kOutputRank},c::DataType::kBFloat16));
  AttentionResult result;result.state=state;auto &ports=result.ports;ports.input=input;
  ports.query_a=fp8(f,input,parameters.query_a,model.query_rank,model.hidden);
  ports.query_norm=norm(f,ports.query_a.stored,parameters.query_norm);
  ports.query_b=fp8(f,ports.query_norm.stored,parameters.query_b,model.heads*kDepth,model.query_rank);
  const auto q=ds::query_norm(f,reshape(f,visible(f,ports.query_b.stored),{1,1,model.heads,kDepth}),&ports.query_head_norm);
  const auto rotated_query=ds::rope(f,q,state.cursor,true,false,&ports.query_rotary);
  ports.kv=fp8(f,input,parameters.kv,kDepth,model.hidden);
  ports.kv_norm=norm(f,ports.kv.stored,parameters.kv_norm);
  const auto rotated_kv=ds::rope(f,reshape(f,visible(f,ports.kv_norm.stored),{1,1,1,kDepth}),state.cursor,true,false,&ports.kv_rotary);
  ports.window_encoded=encode(f,reshape(f,rotated_kv,{1,kDepth}),kMainEncoding);
  ports.window_next=scatter(f,state.window,index(f,plan.position%kWindow),ports.window_encoded);
  result.state.window=ports.window_next;
  const auto window_count=std::min(plan.position+1,kWindow);
  const auto window_first=plan.position+1-window_count;
  std::vector<std::int32_t> chronological(std::size_t(window_count),0);
  for(std::int64_t row=0;row<window_count;++row)chronological[std::size_t(row)]=std::int32_t((window_first+row)%kWindow);
  ports.window_rows=indices(f,chronological);
  ports.window_decoded=decode(f,ports.window_next,ports.window_rows,kMainEncoding);

  ports.compressor=append_compressor(f,input,state.raw,state.score,state.pool,parameters.compressor,ratio,false,plan);
  result.state.raw=ports.compressor.raw_next;result.state.score=ports.compressor.score_next;result.state.pool=ports.compressor.pool_next;
  const auto visible_count=plan.visible(ratio);
  if(visible_count)ports.compressed.emplace();
  if(ratio==4) {
    auto &ip=ports.indexer.emplace();const auto &parameter=*parameters.indexer;const auto old_index=*state.index;
    const auto append_index=[&] {
      ip.compressor=append_compressor(f,input,old_index.raw,old_index.score,old_index.pool,parameter.compressor,4,true,plan);
      result.state.index=IndexState{ip.compressor.raw_next,ip.compressor.score_next,ip.compressor.pool_next};
    };
    if(visible_count) {
      auto &selection=ip.selection.emplace();
      selection.query=fp8(f,ports.query_norm.stored,parameter.query,kIndexHeads*kIndexDepth,model.query_rank);
      const auto rotated=ds::rope(f,reshape(f,visible(f,selection.query.stored),{1,1,kIndexHeads,kIndexDepth}),state.cursor,true,false,&selection.rotary);
      const auto query_rows=reshape(f,rotated,{kIndexHeads,kIndexDepth});
      selection.hadamard_raw=f.one(c::Operation(c::Hadamard{}),{query_rows});selection.hadamard_stored=stored(f,selection.hadamard_raw);
      selection.query_qat=f.one(c::model_ops::block_quantization_roundtrip(c::model_ops::BlockQuantizationFormat::kFloat4E2M1Block32),{selection.hadamard_stored});
      append_index();
      require(f,parameter.weights,sem::dense({kIndexHeads,model.hidden},c::DataType::kBFloat16));
      const auto weight_raw=f.one(c::model_ops::dense_linear_f32(),{input,parameter.weights});
      selection.weights={input,weight_raw,stored(f,weight_raw)};
      selection.scaled_weights_raw=binary(f,c::BinaryKind::kMultiply,visible(f,selection.weights.stored),
          core::filled(f,sem::dense({1}),1.F/std::sqrt(float(kIndexHeads*kIndexDepth))));
      selection.scaled_weights_stored=stored(f,selection.scaled_weights_raw);
      const std::array<std::int32_t,2> ends{0,std::int32_t(visible_count)};
      const auto ranges=reshape(f,indices(f,ends),{1,2});
      selection.scores=f.one(c::model_ops::row_cache_indexer_scores(),{reshape(f,selection.query_qat,{1,kIndexHeads,kIndexDepth}),
          ip.compressor.pool_next,selection.scaled_weights_stored,ranges});
      const auto selected=f.append(c::Operation(c::TopK{.count=std::size_t(std::min(visible_count,model.index_top)),.stable=true}),{selection.scores},2);
      selection.ids=selected[1];ports.compressed->ids=reshape(f,selection.ids,{std::min(visible_count,model.index_top)});
    } else append_index();
  } else if(visible_count)ports.compressed->ids=range(f,visible_count);
  const auto length=window_count+(ratio==4?std::min(visible_count,model.index_top):visible_count);
  if(ports.compressed) {
    ports.compressed->decoded=decode(f,ports.compressor.pool_next,ports.compressed->ids,kMainEncoding);
    ports.chosen=reshape(f,ds::concat(f,visible(f,ports.window_decoded),visible(f,ports.compressed->decoded),0),{1,length,1,kDepth});
  } else ports.chosen=reshape(f,visible(f,ports.window_decoded),{1,length,1,kDepth});
  ports.mask=core::filled(f,sem::dense({1,length}),0.F);
  ports.context_raw=f.one(c::model_ops::sink_attention(1.F/std::sqrt(float(kDepth))),
      {rotated_query,ports.chosen,ports.chosen,ports.mask,parameters.sinks});
  ports.context_stored=stored(f,ports.context_raw);
  const auto inverse=ds::rope(f,visible(f,ports.context_stored),state.cursor,true,true,&ports.inverse_rotary);
  const auto output_a=dspark_formal_fragment::output_a(f,stored(f,inverse),parameters.output_a,1,model);
  ports.output_a_input=output_a.input;ports.output_a_stored=output_a.stored;
  ports.output_b=fp8(f,reshape(f,ports.output_a_stored,{1,model.groups*kOutputRank}),parameters.output_b,model.hidden,model.groups*kOutputRank);
  result.output=ports.output_b.stored;
  // The actual accepted frontier input selects its next I32 value in Graph;
  // no host-computed output or oracle tensor substitutes for cursor state.
  std::vector<std::int32_t> next(std::size_t(plan.capacity+1),0);
  for(std::size_t at=0;at<next.size();++at)next[at]=std::int32_t(at+1);
  result.state.cursor=gather(f,indices(f,next),state.cursor);return result;
}
}

namespace dspark_original_target_fragment {
LayerResult append_layer(sem::Fragment &f,Value input,LayerState state,const LayerParameters &parameters,
    std::int64_t ratio,const Plan &plan) {
  const auto &model=plan.geometry;
  require(f,input,sem::dense({1,4,model.hidden},c::DataType::kBFloat16));require_state(f,state,ratio,plan);
  LayerResult result;auto &ports=result.ports;
  const auto actual=visible(f,input);
  const auto attention_hc=ds::hyper_pre(f,actual,model.hidden,parameters.attention_hc.mixing,true);
  ports.attention_hc=attention_hc.ports;
  ports.attention_norm=norm(f,attention_hc.collapsed,parameters.attention_hc.norm);
  const auto attention=append_attention(f,ports.attention_norm.stored,state,parameters.attention,ratio,plan);
  result.state=attention.state;ports.attention=attention.ports;
  const auto after_attention=ds::hyper_post(f,attention_hc,actual,visible(f,attention.output),&ports.attention_post);
  const auto ffn_hc=ds::hyper_pre(f,after_attention,model.hidden,parameters.ffn_hc.mixing,true);
  ports.ffn_hc=ffn_hc.ports;ports.ffn_norm=norm(f,ffn_hc.collapsed,parameters.ffn_hc.norm);
  const auto moe=append_moe(f,ports.ffn_norm.stored,parameters.moe,model);ports.moe=moe.ports;result.boundary=moe.boundary;
  const auto output=ds::hyper_post(f,ffn_hc,after_attention,visible(f,moe.output),&ports.ffn_post);
  result.output=stored(f,output);result.capture=visible(f,result.output);return result;
}
}
