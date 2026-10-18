// Included after the public consumer's backend adapters. This exercises state
// feedback through public APIs, not a model loader or an acceptance policy.
template<class Route>Status exercise_mtp_prefixes(Route &route,const char *name) {
  constexpr std::size_t n=4,d=128,channels=128,k=4,capacity=n+2;
  HYLE_ASSIGN_OR_RETURN(auto sd,dense({1,1,d,d}));
  HYLE_ASSIGN_OR_RETURN(auto hd,dense({1,k-1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto cache_desc,dense({1,capacity,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto all_desc,dense({1,k+n,channels}));
  HYLE_ASSIGN_OR_RETURN(auto verify_window,dense({1,k+n-1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto weights_desc,dense({channels,k}));
  HYLE_ASSIGN_OR_RETURN(auto conv_desc,dense({1,n,channels}));
  HYLE_ASSIGN_OR_RETURN(auto one_conv,dense({1,1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto full_window,dense({1,k,channels}));
  HYLE_ASSIGN_OR_RETURN(auto qkv_desc,dense({1,n,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto gate_desc,dense({1,n,1,1}));
  HYLE_ASSIGN_OR_RETURN(auto token_desc,dense({1,1,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto scalar_desc,dense({1,1,1,1}));
  HYLE_ASSIGN_OR_RETURN(auto snapshots_desc,dense({1,n,1,d,d}));
  std::vector<float> initial(d*d),window((k+n)*channels),weights(channels*k);
  std::array<std::vector<float>,5> values;
  for(std::size_t i=0;i<initial.size();++i)initial[i]=float(int(i%23)-11)*0.001f;
  for(std::size_t i=0;i<window.size();++i)window[i]=float(int(i%31)-15)*0.0078125f;
  for(std::size_t i=0;i<weights.size();++i)weights[i]=float(1+i%7)/64.f;
  std::array<c::ReadLease,5> fixed;
  for(std::size_t port=0;port<5;++port) {
    values[port].resize((n+1)*(port<3?d:1));
    for(std::size_t i=0;i<values[port].size();++i)
      values[port][i]=port==3?-0.01f*float(i+1):port==4?0.15f+0.03f*float(i):float(int((i*3+port)%17)-8)*0.015625f;
    HYLE_ASSIGN_OR_RETURN(auto desc,dense({1,n+1,1,static_cast<std::int64_t>(port<3?d:1)}));
    HYLE_ASSIGN_OR_RETURN(fixed[port],route.load(desc,values[port]));
  }
  HYLE_ASSIGN_OR_RETURN(auto s0,route.load(sd,initial));
  HYLE_ASSIGN_OR_RETURN(auto all_window,route.load(all_desc,window));
  HYLE_ASSIGN_OR_RETURN(auto history,all_window.subview(0,(k-1)*channels*sizeof(float)));
  HYLE_ASSIGN_OR_RETURN(auto filter,route.load(weights_desc,weights));
  std::vector<float> keys(capacity*d),vals(capacity*d),base_keys(capacity*d),base_vals(capacity*d);
  for(std::size_t i=0;i<keys.size();++i) {
    keys[i]=float(int(i%29)-14)*0.03125f;vals[i]=float(int(i%37)-18)*0.0625f;
    if(i<d){base_keys[i]=keys[i];base_vals[i]=vals[i];}
  }
  HYLE_ASSIGN_OR_RETURN(auto base_k,route.load(cache_desc,base_keys));
  HYLE_ASSIGN_OR_RETURN(auto base_v,route.load(cache_desc,base_vals));
  HYLE_ASSIGN_OR_RETURN(auto verified_k,route.load(cache_desc,keys));
  HYLE_ASSIGN_OR_RETURN(auto verified_v,route.load(cache_desc,vals));
  const std::array specs{c::StateEntrySpec{.description=sd},c::StateEntrySpec{.description=hd},
      c::StateEntrySpec{.description=cache_desc},c::StateEntrySpec{.description=cache_desc},
      c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(std::uint64_t)}};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::StateSchema::create(specs));
  HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(schema));
  std::array<c::StateEntryId,5> ids;
  for(std::size_t i=0;i<ids.size();++i){HYLE_ASSIGN_OR_RETURN(ids[i],schema.entry(i));}
  const std::uint64_t original_length=1;
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{ids[0],s0},c::StateMember{ids[1],history},
      c::StateMember{ids[2],base_k},c::StateMember{ids[3],base_v},c::StateMember{ids[4],{},std::as_bytes(std::span(&original_length,1))}}));
  // Independent sequential formula, with a nonzero starting state. These values
  // are only assertions; every continuation below consumes the library output.
  std::vector<std::vector<float>> states{initial};std::vector<float> expected_y((n+1)*d);
  for(std::size_t t=0;t<=n;++t) {
    auto next=states.back();const double decay=std::exp(double(values[3][t]));
    for(std::size_t row=0;row<d;++row) {
      double dot=0;for(std::size_t col=0;col<d;++col)dot+=double(states.back()[row*d+col])*decay*values[1][t*d+col];
      const double delta=values[4][t]*(values[2][t*d+row]-dot);double y=0;
      for(std::size_t col=0;col<d;++col){const double s=double(states.back()[row*d+col])*decay+delta*values[1][t*d+col];next[row*d+col]=float(s);y+=s*values[0][t*d+col];}
      expected_y[t*d+row]=float(y/std::sqrt(double(d)));
    }
    states.push_back(std::move(next));
  }
  Work verify{c::Operation(c::GatedDeltaNet{.snapshot_count=n}),
      {qkv_desc,qkv_desc,qkv_desc,gate_desc,gate_desc,sd},{qkv_desc,snapshots_desc}};
  Work decode{c::Operation(c::GatedDeltaNet{}),
      {token_desc,token_desc,token_desc,scalar_desc,scalar_desc,sd},{token_desc,sd}};
  Work conv_verify{c::model_ops::causal_conv1d(),{verify_window,weights_desc},{conv_desc}};
  Work conv_next{c::model_ops::causal_conv1d(),{full_window,weights_desc},{one_conv}};
  Work append_history{c::Operation(c::Concat{.axis=1}),{hd,one_conv},{full_window}};
  std::array<c::ReadLease,6> verify_inputs;
  for(std::size_t port=0;port<5;++port){HYLE_ASSIGN_OR_RETURN(verify_inputs[port],fixed[port].subview(0,n*(port<3?d:1)*sizeof(float)));}
  verify_inputs[5]=s0;
  HYLE_ASSIGN_OR_RETURN(auto verification_window,all_window.subview(0,(k+n-1)*channels*sizeof(float)));
  for(bool graph:{false,true}) {
    HYLE_ASSIGN_OR_RETURN(auto verified,route.math(verify,verify_inputs,graph));
    HYLE_ASSIGN_OR_RETURN(auto verified_y,route.read(verified[0],qkv_desc));
    HYLE_RETURN_IF_ERROR(close_values(verified_y,std::span(expected_y).first(n*d)));
    HYLE_ASSIGN_OR_RETURN(auto convolution,route.math(conv_verify,std::array{verification_window,filter},graph));
    std::vector<float> conv_expected(n*channels);
    for(std::size_t t=0;t<n;++t)for(std::size_t channel=0;channel<channels;++channel)
      for(std::size_t tap=0;tap<k;++tap)conv_expected[t*channels+channel]+=window[(t+tap)*channels+channel]*weights[channel*k+tap];
    HYLE_ASSIGN_OR_RETURN(auto conv_actual,route.read(convolution[0],conv_desc));
    HYLE_RETURN_IF_ERROR(close_values(conv_actual,conv_expected));
    for(std::size_t accepted=0;accepted<=n;++accepted) {
      HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));
      if(accepted) {
        // B=1 makes both accepted state and the next conv history contiguous.
        // ReadLease subviews retain the original allocation; no host readback,
        // per-token re-execution, suffix erase or mutable alias is necessary.
        HYLE_ASSIGN_OR_RETURN(auto selected,verified[1].subview((accepted-1)*d*d*sizeof(float),d*d*sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto selected_history,verification_window.subview(accepted*channels*sizeof(float),(k-1)*channels*sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
        HYLE_RETURN_IF_ERROR(draft.replace(ids[0],selected));HYLE_RETURN_IF_ERROR(draft.replace(ids[1],selected_history));
        HYLE_RETURN_IF_ERROR(draft.replace(ids[2],verified_k));HYLE_RETURN_IF_ERROR(draft.replace(ids[3],verified_v));
        const auto length=std::uint64_t(1+accepted);
        HYLE_RETURN_IF_ERROR(draft.replace_metadata(ids[4],std::as_bytes(std::span(&length,1))));
        HYLE_ASSIGN_OR_RETURN(auto candidate,draft.finalize());HYLE_RETURN_IF_ERROR(head.publish(base.identity(),candidate));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto rejected,store.begin(base));
        HYLE_RETURN_IF_ERROR(rejected.replace(ids[2],verified_k));HYLE_RETURN_IF_ERROR(rejected.cancel());
      }
      HYLE_ASSIGN_OR_RETURN(auto current,head.current());
      HYLE_ASSIGN_OR_RETURN(auto accepted_state,current.member(ids[0]));
      HYLE_ASSIGN_OR_RETURN(auto accepted_values,route.read(accepted_state,sd));HYLE_RETURN_IF_ERROR(close_values(accepted_values,states[accepted]));
      HYLE_ASSIGN_OR_RETURN(auto metadata,current.member(ids[4]));std::uint64_t length=0;
      HYLE_RETURN_IF_ERROR(metadata.view()->read(std::as_writable_bytes(std::span(&length,1))));
      HYLE_RETURN_IF_ERROR(require(length==1+accepted,"MTP KV length was not committed with the state"));
      std::array<c::ReadLease,6> next_inputs;
      for(std::size_t port=0;port<5;++port){const auto bytes=(port<3?d:1)*sizeof(float);HYLE_ASSIGN_OR_RETURN(next_inputs[port],fixed[port].subview(accepted*bytes,bytes));}
      next_inputs[5]=accepted_state;
      HYLE_ASSIGN_OR_RETURN(auto next,route.math(decode,next_inputs,graph));
      HYLE_ASSIGN_OR_RETURN(auto next_y,route.read(next[0],token_desc));
      HYLE_RETURN_IF_ERROR(close_values(next_y,std::span(expected_y).subspan(accepted*d,d)));
      HYLE_ASSIGN_OR_RETURN(auto next_s,route.read(next[1],sd));HYLE_RETURN_IF_ERROR(close_values(next_s,states[accepted+1]));
      HYLE_ASSIGN_OR_RETURN(auto accepted_history,current.member(ids[1]));
      HYLE_ASSIGN_OR_RETURN(auto new_sample,all_window.subview((k-1+accepted)*channels*sizeof(float),channels*sizeof(float)));
      HYLE_ASSIGN_OR_RETURN(auto appended,route.math(append_history,std::array{accepted_history,new_sample},graph));
      HYLE_ASSIGN_OR_RETURN(auto next_conv,route.math(conv_next,std::array{appended[0],filter},graph));
      std::vector<float> expected_conv(channels);
      for(std::size_t channel=0;channel<channels;++channel)for(std::size_t tap=0;tap<k;++tap)
        expected_conv[channel]+=window[(accepted+tap)*channels+channel]*weights[channel*k+tap];
      HYLE_ASSIGN_OR_RETURN(auto actual_conv,route.read(next_conv[0],one_conv));HYLE_RETURN_IF_ERROR(close_values(actual_conv,expected_conv));
      // Only the active KV prefix reaches attention, even though verified
      // suffix bytes still exist. Add the current token on device, then use it.
      HYLE_ASSIGN_OR_RETURN(auto prefix_desc,dense({1,static_cast<std::int64_t>(length),1,d}));
      HYLE_ASSIGN_OR_RETURN(auto next_cache_desc,dense({1,static_cast<std::int64_t>(length+1),1,d}));
      std::array<c::ReadLease,2> caches;
      Work append_cache{c::Operation(c::Concat{.axis=1}),{prefix_desc,token_desc},{next_cache_desc}};
      for(std::size_t port=0;port<2;++port) {
        HYLE_ASSIGN_OR_RETURN(auto cache,current.member(ids[2+port]));
        HYLE_ASSIGN_OR_RETURN(auto prefix,cache.subview(0,length*d*sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto new_kv,(port?verified_v:verified_k).subview(length*d*sizeof(float),d*sizeof(float)));
        HYLE_ASSIGN_OR_RETURN(auto joined,route.math(append_cache,std::array{prefix,new_kv},graph));caches[port]=joined[0];
      }
      Work attention{c::Operation(c::ScaledDotProductAttention{.layout=c::AttentionLayout::kBatchSequenceHeadDepth,.scale=0.125f}),
          {token_desc,next_cache_desc,next_cache_desc},{token_desc}};
      HYLE_ASSIGN_OR_RETURN(auto attended,route.math(attention,std::array{next_inputs[0],caches[0],caches[1]},graph));
      std::vector<double> probabilities(length+1);double total=0;
      for(std::size_t t=0;t<=length;++t){double score=0;for(std::size_t col=0;col<d;++col)score+=double(values[0][accepted*d+col])*keys[t*d+col];probabilities[t]=std::exp(score*0.125);total+=probabilities[t];}
      std::vector<float> expected_attention(d);
      for(std::size_t col=0;col<d;++col){double value=0;for(std::size_t t=0;t<=length;++t)value+=probabilities[t]*vals[t*d+col]/total;expected_attention[col]=float(value);}
      HYLE_ASSIGN_OR_RETURN(auto actual_attention,route.read(attended[0],token_desc));HYLE_RETURN_IF_ERROR(close_values(actual_attention,expected_attention));
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto old_state,base.member(ids[0]));HYLE_ASSIGN_OR_RETURN(auto old_values,route.read(old_state,sd));HYLE_RETURN_IF_ERROR(close_values(old_values,initial));
  HYLE_ASSIGN_OR_RETURN(auto old_k,base.member(ids[2]));HYLE_ASSIGN_OR_RETURN(auto old_keys,route.read(old_k,cache_desc));HYLE_RETURN_IF_ERROR(close_values(old_keys,base_keys));
  HYLE_RETURN_IF_ERROR(store.drain());
  std::printf("MTP %s: accept 0..4, GDN/conv/KV atomic prefix + own-state continuation Direct/Graph PASS\n",name);
  return Status::Ok();
}
