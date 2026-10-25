// The same13-resource ordinary sparse-attention component as target43.
// Upstream normalized NX/QR are declared BF16 inputs; original layer43
// projections/indexer/KV_b are authenticated cached weights. No FFN, vocabulary,
// generated target weights, MTP policy or alternative executor is introduced.
namespace flash_ordinary_startup {
struct State final {
  c::StateSchema schema;c::StateStore store;
  std::array<c::StateEntryId,kSparseEntries.size()> entries;
  static State create(const c::StorageOwner &owner) {
    const auto desc=descriptions();std::vector<c::StateEntrySpec> specs;
    for(const auto entry:kSparseEntries)specs.push_back({.description=desc[entry],.resource=owner.ref()});
    State s;s.schema=take(c::StateSchema::create(specs),"Flash ordinary13-resource schema");
    s.store=take(c::StateStore::create(s.schema,{.max_versions=8,.max_drafts=1,
        .max_retained_blocks=128,.max_backing_bytes=128u<<20}),"Flash ordinary bounded store");
    for(std::size_t i=0;i<s.entries.size();++i)s.entries[i]=take(s.schema.entry(i),"Flash ordinary state entry");return s;
  }
  std::size_t slot(Entry entry)const {
    const auto found=std::find(kSparseEntries.begin(),kSparseEntries.end(),entry);
    if(found==kSparseEntries.end())throw std::runtime_error("Flash ordinary non-sparse entry");
    return std::size_t(found-kSparseEntries.begin());
  }
  c::ReadLease member(const c::StateVersion &v,Entry entry)const {
    return take(v.member(entries[slot(entry)]),"Flash ordinary actual state member");
  }
  std::int32_t scalar(const c::StateVersion &v,Entry entry)const{return typed<std::int32_t>(member(v,entry)).at(0);}
  c::StateVersion empty(const c::StorageOwner &owner) {
    const auto desc=descriptions();std::vector<c::StateMember> members;
    for(std::size_t i=0;i<kSparseEntries.size();++i) {
      const auto entry=kSparseEntries[i];const auto &d=desc[entry];
      Bytes bytes(take(d.required_bytes(),"Flash ordinary positive state extent"));
      if(entry==kCache||entry==kPoolScales||entry==kQueryState) {
        bytes=sem::bytes(std::vector<float>(bytes.size()/sizeof(float),std::bit_cast<float>(std::uint32_t{0x7fc12345})));
      } else if(entry==kRawKey||entry==kRawGate) {
        bytes=sem::bytes(std::vector<std::uint16_t>(bytes.size()/sizeof(std::uint16_t),0x7fc1));
      } else if(entry==kPool)std::fill(bytes.begin(),bytes.end(),std::byte{0x7f});
      else if(entry==kSelection)bytes=sem::bytes(std::vector<std::int32_t>(std::size_t(kIds),-1));
      // Every cursor, ring offset, closed-pool count and tail-start remains0.
      members.push_back({entries[i],upload(owner,bytes)});
    }
    return take(store.import(members),"Flash ordinary genuinely empty checkpoint");
  }
  std::array<Bytes,kSparseEntries.size()> image(const c::StateVersion &v)const {
    std::array<Bytes,kSparseEntries.size()> result;
    for(std::size_t i=0;i<kSparseEntries.size();++i)result[i]=read(member(v,kSparseEntries[i]));return result;
  }
  c::StateVersion stage(const c::StateVersion &from,const Recipe &recipe,std::span<const c::ReadLease> outputs) {
    auto draft=take(store.begin(from),"Flash ordinary unpublished13-resource successor");
    for(std::size_t i=0;i<kSparseEntries.size();++i) {
      const auto port=recipe.state_ports[kSparseEntries[i]];
      if(port!=std::numeric_limits<std::size_t>::max())check(draft.replace(entries[i],outputs[port]),"Flash ordinary actual output lease");
    }
    return take(draft.finalize(),"Flash ordinary complete successor");
  }
};
inline Recipe build(std::int32_t position,std::shared_ptr<OriginalSparseLayer> source) {
  Recipe r;r.fresh=true;r.position=position;r.selected_tail=(position+1)%4;r.sparse=std::move(source);
  r.state_entries.assign(kSparseEntries.begin(),kSparseEntries.end());
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f=r.fragment;f.profile=r.sparse->profile;f.region=0x47464c4f52445354ULL;f.quantized=true;
  const auto desc=descriptions();std::array<Value,kSparseEntries.size()> state;
  for(std::size_t i=0;i<state.size();++i)state[i]=f.input(desc[kSparseEntries[i]]);
  const auto nx_stored=f.input(sem::dense({1,kHidden},c::DataType::kBFloat16));
  const auto qr=f.input(sem::dense({1,kQueryRank},c::DataType::kBFloat16));
  r.sparse_hidden=expose(r,nx_stored,true);r.sparse_query=expose(r,qr,true);
  const auto nx=codec::cast(f,nx_stored,c::DataType::kFloat32);r.nx=expose(r,nx);
  (void)append_sparse_attention(r,SparseState13(f,state),nx,0,nullptr,&qr);return r;
}
inline Bytes upstream(std::int64_t width,std::int32_t position,int salt) {
  auto values=d5_gguf_test::activation(width,salt+17*position);
  std::vector<std::uint16_t> stored(values.size());
  for(std::size_t i=0;i<values.size();++i)stored[i]=codec::bfloat(values[i]);return sem::bytes(stored);
}
inline void run(c::Context &context,const c::StorageOwner &owner,bool cuda,const std::filesystem::path &directory) {
  auto source=std::make_shared<OriginalSparseLayer>();
  source->indexer=std::make_shared<const original_index::Data>(original_index::Data::load(directory,original_target43::kIndexerSha256));
  source->indexer_roles=original_index::bind(*source->indexer,owner);
  source->attention=prepare_original_attention(std::make_shared<const original_attention::Data>(
      original_attention::Data::load(directory,original_target43::kAttentionSha256)),context,owner,cuda);
  std::array<std::unique_ptr<Program>,5> programs;
  for(std::int32_t position=0;position<5;++position)
    programs[std::size_t(position)]=Program::create(build(position,source),context,owner,cuda,1);
  auto queue=take(context.create_queue(),"Flash ordinary startup queue");auto state=State::create(owner);
  {
    auto initial=state.empty(owner);auto head=take(c::StateHead::create(state.store,initial),"Flash ordinary empty Head");
    std::vector<c::StateVersion> retained{initial};std::vector<std::array<Bytes,kSparseEntries.size()>> images{state.image(initial)};initial={};
    std::array<std::vector<std::uint16_t>,4> first_keys,first_gates;
    for(std::int32_t position=0;position<5;++position) {
      const auto before=take(head.current(),"Flash ordinary actual accepted Head");
      EXPECT_EQ(state.scalar(before,kMlaCursor),position);EXPECT_EQ(state.scalar(before,kIndexCursor),position);
      EXPECT_EQ(state.scalar(before,kPoolCount),position/4);EXPECT_EQ(state.scalar(before,kTailStart),position-position%4);
      EXPECT_EQ(state.scalar(before,kRingOffset),position%6);EXPECT_EQ(state.scalar(before,kTailValid),position%4);
      std::vector<c::ReadLease> arguments;for(const auto entry:kSparseEntries)arguments.push_back(state.member(before,entry));
      arguments.push_back(upload(owner,upstream(kHidden,position,73)));arguments.push_back(upload(owner,upstream(kQueryRank,position,97)));
      auto &program=*programs[std::size_t(position)];const auto outputs=program.execute(queue,arguments);
      const auto next=state.stage(before,program.recipe,outputs);
      const auto nx=typed<float>(outputs[program.recipe.nx]);
      observe_attention(state,before,next,program.recipe,outputs,nx,position);
      if(::testing::Test::HasFailure())throw std::runtime_error("Flash ordinary startup independent reference failed before publication");
      if(position<4) {
        first_keys[std::size_t(position)]=typed<std::uint16_t>(outputs[program.recipe.raw_key]);
        first_gates[std::size_t(position)]=typed<std::uint16_t>(outputs[program.recipe.raw_gate]);
      }
      if(position==3) {
        const auto raw=typed<std::uint16_t>(state.member(next,kRawKey)),gates=typed<std::uint16_t>(state.member(next,kRawGate));
        for(std::size_t t=0;t<4;++t)for(std::size_t d=0;d<std::size_t(kIndexDepth);++d) {
          EXPECT_EQ(raw[t*kIndexDepth+d],first_keys[t][d]);EXPECT_EQ(gates[t*kIndexDepth+d],first_gates[t][d]);
        }
        // Independent original APE/gates/actual key rows, not observed pool bytes
        // used as future inputs. observe_history also checks quantization itself.
        const std::array<std::int32_t,4> rows{0,1,2,3};
        const auto pooled=index_reference::pool_reference(raw,gates,source->indexer->ape,rows);
        sem::oracle::expect_close(typed<float>(outputs[program.recipe.compressed]),pooled,8.e-5F,4.e-5F);
        EXPECT_EQ(state.scalar(next,kPoolCount),1);EXPECT_EQ(state.scalar(next,kTailStart),4);
      }
      const auto ids=typed<std::int32_t>(state.member(next,kSelection));
      const auto live=std::size_t(position+1);ASSERT_EQ(ids.size(),std::size_t(kIds));
      std::vector<std::int32_t> valid(ids.begin(),ids.begin()+std::ptrdiff_t(live));std::sort(valid.begin(),valid.end());
      for(std::size_t i=0;i<live;++i)EXPECT_EQ(valid[i],std::int32_t(i));
      for(std::size_t i=live;i<ids.size();++i)EXPECT_EQ(ids[i],-1);
      const auto cache=typed<float>(state.member(next,kCache));
      for(std::size_t i=0;i<cache.size();++i)if(i<live*std::size_t(kLatent))EXPECT_TRUE(std::isfinite(cache[i]));else EXPECT_TRUE(std::isnan(cache[i]));
      for(const float x:typed<float>(outputs[program.recipe.attention]))EXPECT_TRUE(std::isfinite(x));
      for(const float x:typed<float>(outputs[program.recipe.output]))EXPECT_TRUE(std::isfinite(x));
      if(::testing::Test::HasFailure())throw std::runtime_error("Flash ordinary startup state/poison check failed before publication");
      check(head.publish(before.identity(),next),"Flash ordinary atomic13-resource publication");
      EXPECT_FALSE(head.publish(before.identity(),next).ok());
      for(std::size_t i=0;i<retained.size();++i)EXPECT_EQ(state.image(retained[i]),images[i]);
      retained.push_back(next);images.push_back(state.image(next));
    }
    const auto final=take(head.current(),"Flash ordinary final accepted state");
    EXPECT_EQ(state.scalar(final,kMlaCursor),5);EXPECT_EQ(state.scalar(final,kIndexCursor),5);
    EXPECT_EQ(state.scalar(final,kPoolCount),1);EXPECT_EQ(state.scalar(final,kTailValid),1);
  }
  check(state.store.collect(),"Flash ordinary retired state collection");
  EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().backing_bytes,0u);
  ::testing::Test::RecordProperty("actual_positions","0,1,2,3,4");
  ::testing::Test::RecordProperty("scope","original layer43 sparse attention/indexer13-resource component; declared upstream BF16 NX/QR; empty cache->first four real rows pooled->next tail; no full model/MTP/throughput claim");
}
} // namespace flash_ordinary_startup

#if defined(HYLE_TEST_GLM_STATE_CPU)
namespace flash_ordinary_startup {struct SystemResource final {m::MemoryResourceRef ref()noexcept{return m::system_resource();}};}
TEST(GlmFlashOrdinaryEarlyHistory,CpuOriginalSparseStartsEmptyAndFeedsBackThroughFirstPool) {
  const auto directory=original_directory("HYLE_GLM_MTP_TARGET43_DIR");
  if(directory.empty())GTEST_SKIP()<<"Set existing authenticated target43 directory; no implicit download";
  auto context=take(g::cpu::create_context(),"Flash ordinary CPU context");
  const auto owner=take(c::StorageOwner::adopt(context,flash_ordinary_startup::SystemResource{}),"Flash ordinary CPU owner");
  flash_ordinary_startup::run(context,owner,false,directory);
}
#endif
#if defined(HYLE_TEST_GLM_STATE_CUDA)
TEST(GlmFlashOrdinaryEarlyHistory,CudaOriginalSparseStartsEmptyAndFeedsBackThroughFirstPool) {
  const auto directory=original_directory("HYLE_GLM_MTP_TARGET43_DIR");
  if(directory.empty())GTEST_SKIP()<<"Set existing authenticated target43 directory; no implicit download";
  auto context=take(g::cuda::create_context(),"Flash ordinary CUDA context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"Flash ordinary CUDA memory")),"Flash ordinary CUDA owner");
  flash_ordinary_startup::run(context,owner,true,directory);
}
#endif
