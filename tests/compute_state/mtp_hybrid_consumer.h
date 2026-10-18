// Include after CommonRoute, only when both CPU and CUDA are enabled. The CPU
// route uses pinned host storage owned by the same CUDA Context as the GPU
// route. This is real CPU math on host-accessible storage, not a CUDA kernel.
template<class VerifyRoute,class ContinueRoute>
Status exercise_mtp_hybrid(VerifyRoute &verify,ContinueRoute &continuation,const char *name) {
  constexpr std::size_t n=4,accepted=2,d=128,channels=128,k=4;
  constexpr std::size_t state_bytes=d*d*sizeof(float),history_bytes=(k-1)*channels*sizeof(float);
  const bool source_cuda=verify.context.device().backend_id()==c::cuda::kBackendId;
  const bool target_cuda=continuation.context.device().backend_id()==c::cuda::kBackendId;
  HYLE_RETURN_IF_ERROR(require(source_cuda!=target_cuda,"MTP hybrid requires one CPU and one CUDA route"));
  HYLE_ASSIGN_OR_RETURN(auto state_desc,dense({1,1,d,d}));
  HYLE_ASSIGN_OR_RETURN(auto history_desc,dense({1,k-1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto qkv_desc,dense({1,n,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto gate_desc,dense({1,n,1,1}));
  HYLE_ASSIGN_OR_RETURN(auto snapshots_desc,dense({1,n,1,d,d}));
  HYLE_ASSIGN_OR_RETURN(auto token_desc,dense({1,1,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto scalar_desc,dense({1,1,1,1}));
  HYLE_ASSIGN_OR_RETURN(auto window_desc,dense({1,k+n,channels}));
  HYLE_ASSIGN_OR_RETURN(auto verify_window_desc,dense({1,k+n-1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto weights_desc,dense({channels,k}));
  HYLE_ASSIGN_OR_RETURN(auto conv_desc,dense({1,n,channels}));
  HYLE_ASSIGN_OR_RETURN(auto sample_desc,dense({1,1,channels}));
  HYLE_ASSIGN_OR_RETURN(auto next_window_desc,dense({1,k,channels}));
  std::vector<float> initial(d*d),window((k+n)*channels),weights(channels*k);
  for(std::size_t i=0;i<initial.size();++i)initial[i]=float(int(i%23)-11)*0.001f;
  for(std::size_t i=0;i<window.size();++i)window[i]=float(int(i%31)-15)*0.0078125f;
  for(std::size_t i=0;i<weights.size();++i)weights[i]=float(1+i%7)/64.f;
  std::array<std::vector<float>,5> values;
  std::array<c::ReadLease,5> inputs;
  for(std::size_t port=0;port<values.size();++port) {
    const auto width=port<3?d:1;
    values[port].resize((n+1)*width);
    for(std::size_t i=0;i<values[port].size();++i)
      values[port][i]=port==3?-0.01f*float(i+1):port==4?0.15f+0.03f*float(i):float(int((i*3+port)%17)-8)*0.015625f;
    HYLE_ASSIGN_OR_RETURN(inputs[port],verify.load(port<3?qkv_desc:gate_desc,std::span(values[port]).first(n*width)));
  }
  HYLE_ASSIGN_OR_RETURN(auto s0,verify.load(state_desc,initial));
  HYLE_ASSIGN_OR_RETURN(auto samples,verify.load(window_desc,window));
  HYLE_ASSIGN_OR_RETURN(auto filter,verify.load(weights_desc,weights));
  std::vector<std::vector<float>> states{initial};std::vector<float> expected_y(n*d);
  // A new token at index n is deliberately outside the verified candidate.
  // Only assertions use this sequential formula; no oracle state is uploaded.
  const auto step=[&](std::span<const float> previous,std::size_t token,std::span<float> y) {
    std::vector<float> next(previous.begin(),previous.end());
    const double decay=std::exp(double(values[3][token]));
    for(std::size_t row=0;row<d;++row) {
      double projected=0;
      for(std::size_t col=0;col<d;++col)projected+=double(previous[row*d+col])*decay*values[1][token*d+col];
      const double delta=values[4][token]*(values[2][token*d+row]-projected);double score=0;
      for(std::size_t col=0;col<d;++col) {
        const double value=double(previous[row*d+col])*decay+delta*values[1][token*d+col];
        next[row*d+col]=float(value);score+=value*values[0][token*d+col];
      }
      y[row]=float(score/std::sqrt(double(d)));
    }
    return next;
  };
  for(std::size_t token=0;token<n;++token)
    states.push_back(step(states.back(),token,std::span(expected_y).subspan(token*d,d)));
  Work verification{c::Operation(c::GatedDeltaNet{.snapshot_count=n}),
      {qkv_desc,qkv_desc,qkv_desc,gate_desc,gate_desc,state_desc},{qkv_desc,snapshots_desc}};
  std::array<c::ReadLease,6> ports{inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],s0};
  HYLE_ASSIGN_OR_RETURN(auto verified,verify.math(verification,ports,false));
  HYLE_ASSIGN_OR_RETURN(auto observed_y,verify.read(verified[0],qkv_desc));HYLE_RETURN_IF_ERROR(close_values(observed_y,expected_y));
  HYLE_ASSIGN_OR_RETURN(auto selected,verified[1].subview((accepted-1)*state_bytes,state_bytes));
  HYLE_ASSIGN_OR_RETURN(auto selected_history,samples.subview(accepted*channels*sizeof(float),history_bytes));
  Work convolution{c::model_ops::causal_conv1d(),{verify_window_desc,weights_desc},{conv_desc}};
  HYLE_ASSIGN_OR_RETURN(auto verify_window,samples.subview(0,(k+n-1)*channels*sizeof(float)));
  HYLE_ASSIGN_OR_RETURN(auto convolved,verify.math(convolution,std::array{verify_window,filter},false));
  std::vector<float> expected_conv(n*channels);
  for(std::size_t token=0;token<n;++token)for(std::size_t channel=0;channel<channels;++channel)
    for(std::size_t tap=0;tap<k;++tap)expected_conv[token*channels+channel]+=window[(token+tap)*channels+channel]*weights[channel*k+tap];
  HYLE_ASSIGN_OR_RETURN(auto observed_conv,verify.read(convolved[0],conv_desc));HYLE_RETURN_IF_ERROR(close_values(observed_conv,expected_conv));
  const std::array specifications{c::StateEntrySpec{.description=state_desc},c::StateEntrySpec{.description=history_desc},
      c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(std::uint64_t)}};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::StateSchema::create(specifications));
  HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(schema));
  HYLE_ASSIGN_OR_RETURN(auto state_id,schema.entry(0));HYLE_ASSIGN_OR_RETURN(auto history_id,schema.entry(1));
  HYLE_ASSIGN_OR_RETURN(auto length_id,schema.entry(2));const std::uint64_t length=1+accepted;
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{state_id,selected},c::StateMember{history_id,selected_history},
      c::StateMember{length_id,{},std::as_bytes(std::span(&length,1))}}));
  HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));
  // CUDA's original copy owner handles both device<->pinned directions. The
  // source is the actual snapshot subrange, never S0 or an oracle upload.
  auto &copy_context=source_cuda?verify.context:continuation.context;
  auto &copy_queue=source_cuda?verify.queue:continuation.queue;
  HYLE_ASSIGN_OR_RETURN(auto source_view,selected.view());
  const std::array copy_specs{c::CopyResourceSpec{source_view.resource(),state_bytes},
      c::CopyResourceSpec{continuation.resource,state_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchProgram::prepare(copy_context,copy_specs));
  auto *endpoint=&copy_queue;
  HYLE_ASSIGN_OR_RETURN(auto clones,c::StateCloneRoute::prepare(store,
      {.resource=continuation.resource,.capacity_bytes=2*state_bytes,.slots=2},copy,
      [endpoint](const c::CopyBatchProgram &program)noexcept{return c::CopyBatchInvocation::create(program,*endpoint);}));
  HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(base));
  HYLE_ASSIGN_OR_RETURN(auto state_copy,draft.clone_for_write(state_id,clones));
  HYLE_ASSIGN_OR_RETURN(auto history_copy,draft.clone_for_write(history_id,clones));
  HYLE_ASSIGN_OR_RETURN(auto state_writer,state_copy.wait());HYLE_RETURN_IF_ERROR(state_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto history_writer,history_copy.wait());HYLE_RETURN_IF_ERROR(history_writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto candidate,draft.finalize());HYLE_RETURN_IF_ERROR(head.publish(base.identity(),candidate));
  HYLE_ASSIGN_OR_RETURN(auto report,clones.report());
  HYLE_RETURN_IF_ERROR(require(report.logical_bytes==state_bytes+history_bytes,"MTP hybrid did not copy the exact state/history ranges"));
  HYLE_ASSIGN_OR_RETURN(auto current,head.current());HYLE_ASSIGN_OR_RETURN(auto continued_state,current.member(state_id));
  HYLE_ASSIGN_OR_RETURN(auto continued_history,current.member(history_id));
  HYLE_ASSIGN_OR_RETURN(auto state_view,continued_state.view());
  HYLE_RETURN_IF_ERROR(require(state_view.resource()==continuation.resource,"MTP hybrid state stayed on the verify resource"));
  std::array<c::ReadLease,6> next_inputs;
  for(std::size_t port=0;port<values.size();++port) {
    const auto width=port<3?d:1;
    HYLE_ASSIGN_OR_RETURN(next_inputs[port],continuation.load(port<3?token_desc:scalar_desc,std::span(values[port]).subspan(n*width,width)));
  }
  next_inputs[5]=continued_state;
  HYLE_ASSIGN_OR_RETURN(auto next_filter,continuation.load(weights_desc,weights));
  HYLE_ASSIGN_OR_RETURN(auto new_sample,continuation.load(sample_desc,std::span(window).subspan((k+n-1)*channels,channels)));
  Work decoding{c::Operation(c::GatedDeltaNet{}),{token_desc,token_desc,token_desc,scalar_desc,scalar_desc,state_desc},{token_desc,state_desc}};
  Work append{c::Operation(c::Concat{.axis=1}),{history_desc,sample_desc},{next_window_desc}};
  Work next_convolution{c::model_ops::causal_conv1d(),{next_window_desc,weights_desc},{sample_desc}};
  std::vector<float> next_y(d);const auto next_state=step(states[accepted],n,next_y);
  std::vector<float> next_conv(channels);
  for(std::size_t channel=0;channel<channels;++channel)for(std::size_t tap=0;tap<k;++tap) {
    const auto row=tap<k-1?accepted+tap:k+n-1;
    next_conv[channel]+=window[row*channels+channel]*weights[channel*k+tap];
  }
  for(bool graph:{false,true}) {
    HYLE_ASSIGN_OR_RETURN(auto output,continuation.math(decoding,next_inputs,graph));
    HYLE_ASSIGN_OR_RETURN(auto y,continuation.read(output[0],token_desc));HYLE_RETURN_IF_ERROR(close_values(y,next_y));
    HYLE_ASSIGN_OR_RETURN(auto s,continuation.read(output[1],state_desc));HYLE_RETURN_IF_ERROR(close_values(s,next_state));
    HYLE_ASSIGN_OR_RETURN(auto appended,continuation.math(append,std::array{continued_history,new_sample},graph));
    HYLE_ASSIGN_OR_RETURN(auto convolved_next,continuation.math(next_convolution,std::array{appended[0],next_filter},graph));
    HYLE_ASSIGN_OR_RETURN(auto actual,continuation.read(convolved_next[0],sample_desc));HYLE_RETURN_IF_ERROR(close_values(actual,next_conv));
  }
  HYLE_ASSIGN_OR_RETURN(auto old_state,verify.read(selected,state_desc));HYLE_RETURN_IF_ERROR(close_values(old_state,states[accepted]));
  HYLE_RETURN_IF_ERROR(store.drain());
  std::printf("MTP %s: accepted snapshot subrange + history transfer, real destination GDN/conv Direct/Graph PASS\n",name);
  return Status::Ok();
}
