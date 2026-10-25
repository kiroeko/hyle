// Included by the existing Flash component consumer TU. These are cold shape
// and original-codec binding checks, not publisher-payload/native witnesses.
namespace flash_early_case {
namespace extent=glm_flash_selection_extent;
inline constexpr std::array<std::int32_t,9> kPositions{0,1,2,3,4,2046,2047,2048,2055};

inline mlx_model::State input_state(sem::Fragment &f) {
  const auto input=[&](std::initializer_list<std::int64_t> dims){return f.input(sem::dense(dims));};
  const auto counter=[&]{return f.input(sem::dense({1},c::DataType::kInt32));};
  return {input({kCapacity,512}),input({6,128}),input({6,128}),input({kPoolCapacity,128}),
      input({1,4096}),input({1,1536}),counter(),counter(),counter(),counter(),counter(),counter(),counter(),
      f.input(sem::dense({extent::kStorage},c::DataType::kInt32)),input({2,4096}),counter()};
}
inline void check_graph(sem::Fragment &f,std::int32_t position,std::size_t expected_topk) {
  const auto e=extent::fresh(position);std::size_t topk=0,attention=0;
  for(const auto &description:f.descriptions)
    for(const auto dimension:description.dimensions())EXPECT_GT(dimension,0);
  for(const auto &node:f.nodes) {
    if(const auto *op=node.operation.get_if<c::TopK>()) {
      ++topk;EXPECT_EQ(op->count,std::size_t(e.selected_pools));
    }
    if(node.operation.get_if<c::ScaledDotProductAttention>()) {
      ++attention;EXPECT_EQ(f.description(node.inputs[1]).dimensions()[1],e.live_tokens());
      EXPECT_EQ(f.description(node.inputs[2]).dimensions()[1],e.live_tokens());
    }
  }
  EXPECT_EQ(topk,expected_topk);EXPECT_EQ(attention,1U);
  const auto graph=sem::make_graph(f);EXPECT_EQ(graph.outputs.size(),f.outputs.size());
}
template<class Parameters,class Linear,class PerHead>
inline Parameters parameters(sem::Fragment &f,Linear linear,PerHead per_head) {
  Parameters p;auto &b=p.nextn.block;
  b.query_b=linear(p,16384,1536);b.kv_a=linear(p,512,4096);b.output=linear(p,4096,16384);
  b.key=per_head(64,512,256);b.value=per_head(64,256,512);
  const auto norm=[&](std::int64_t n){return f.input(sem::dense({n}));};
  b.kv_norm=norm(512);b.index_norm=norm(128);b.index_bias=norm(128);
  b.index_key=linear(p,128,4096);b.index_query=linear(p,4096,1536);
  b.index_weights=f.input(sem::dense({32,4096}));
  p.index_gate=linear(p,128,4096);p.ape=f.input(sem::dense({4,128}));return p;
}
inline void mlx(std::int32_t position) {
  sem::Fragment f;f.profile=&mlx_runtime::kProfile;f.region=0x47464c4541524c59ULL;
  const auto linear=[&](mlx_model::Parameters &p,std::int64_t n,std::int64_t k) {
    if((n==128&&k==4096)||(n==4096&&k==1536))
      return f.input(sem::dense({n,k},c::DataType::kBFloat16));
    mlx_model::affine::Parameters role{
        f.input(sem::dense({n,k/8},c::DataType::kUInt32)),
        f.input(sem::dense({n,k/64},c::DataType::kFloat16)),
        f.input(sem::dense({n,k/64},c::DataType::kFloat16)),64};
    p.project.quantized.push_back(role);return role.codes;
  };
  const auto per_head=[&](std::int64_t h,std::int64_t n,std::int64_t k){
    return f.input(sem::dense({h,n,k},c::DataType::kBFloat16));};
  auto p=parameters<mlx_model::Parameters>(f,linear,per_head);
  p.nextn.block.index_weights=f.input(sem::dense({32,4096},c::DataType::kBFloat16));
  const auto s=input_state(f);const auto nx=f.input(sem::dense({1,4096})),qr=f.input(sem::dense({1,1536}));
  auto result=mlx_model::append_mla(f,s,nx,qr,p,true,position,0);
  EXPECT_EQ(f.description(result.state.selection),sem::dense({extent::kStorage},c::DataType::kInt32));
  EXPECT_EQ(result.index->scored,extent::fresh(position).visible_pools>0);
  result.index->visit_values([&](Value &value){EXPECT_LT(value.ordinal(),f.descriptions.size());});
  f.outputs={result.output.output,result.state.main,result.state.selection};
  check_graph(f,position,extent::fresh(position).visible_pools?2:0);
}
inline void gguf(std::int32_t position) {
  sem::Fragment f;f.profile=kModel.profile;f.region=0x47464c4541524c59ULL;
  const auto linear=[&](gguf_flash::PoolParameters &,std::int64_t n,std::int64_t k){
    const auto d=c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,k,n);
    if(!d)throw std::runtime_error("Flash original packed role description");return f.input(*d);};
  const auto per_head=[&](std::int64_t h,std::int64_t n,std::int64_t k){
    const auto d=c::gguf_grouped_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,h,n,k);
    if(!d)throw std::runtime_error("Flash original packed head description");return f.input(*d);};
  const auto p=parameters<gguf_flash::PoolParameters>(f,linear,per_head);
  const auto s=input_state(f);const auto nx=f.input(sem::dense({1,4096})),qr=f.input(sem::dense({1,1536}));
  auto result=mlx_model::append_mla(f,s,nx,qr,p,true,position,0,gguf_flash::kIndexPolicy);
  EXPECT_EQ(f.description(result.state.selection),sem::dense({extent::kStorage},c::DataType::kInt32));
  f.outputs={result.output.output,result.state.main,result.state.selection};
  check_graph(f,position,extent::fresh(position).visible_pools?1:0);
}
inline void official(std::int32_t position) {
  Recipe r;r.fresh=true;r.position=position;r.selected_tail=std::int32_t(extent::fresh(position).tail);
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());r.sparse=std::make_shared<OriginalSparseLayer>();
  r.sparse->attention=std::make_shared<OriginalAttention>();
  // Descriptor-only external roles follow the actual official parameter branch.
  // Empty leases are never bound or executed by this host graph check.
  for(std::size_t i=0;i<original_index::kParameters;++i)
    r.sparse->indexer_roles[i]={original_index::kNames[i],original_index::description(static_cast<original_index::Parameter>(i)),{}, {}};
  for(std::size_t i=0;i<original_attention::kParameters;++i)
    r.sparse->attention->roles[i]={original_attention::kNames[i],original_attention::description(static_cast<original_attention::Parameter>(i)),{}, {}};
  r.sparse->attention->prepared[0]={"actual prepared original key",sem::dense({64,512,256},c::DataType::kBFloat16),{}, {}};
  r.sparse->attention->prepared[1]={"actual prepared original value",sem::dense({64,256,512},c::DataType::kBFloat16),{}, {}};
  auto &f=r.fragment;f.profile=kModel.profile;f.region=0x47464c4541524c59ULL;
  const auto descriptions_=descriptions();std::array<Value,kSparseEntries.size()> values;
  for(std::size_t i=0;i<values.size();++i)values[i]=f.input(descriptions_[kSparseEntries[i]]);
  const SparseState13 state(f,values);const auto nx=f.input(sem::dense({1,4096})),qr=f.input(sem::dense({1,1536},c::DataType::kBFloat16));
  const auto output=append_sparse_attention(r,state,nx,0,nullptr,&qr);
  EXPECT_EQ(f.description(output),sem::dense({1,4096}));
  EXPECT_EQ(f.description(f.outputs[r.state_ports[kSelection]]),sem::dense({extent::kStorage},c::DataType::kInt32));
  check_graph(f,position,extent::fresh(position).visible_pools?1:0);
}
} // namespace flash_early_case

TEST(GlmFlashOrdinaryEarlyHistoryPlan,UsesActualPoolAndTailWidths) {
  namespace e=glm_flash_selection_extent;
  const std::array<std::int64_t,9> expected{1,2,3,4,5,2047,2048,2049,2048};
  for(std::size_t i=0;i<flash_early_case::kPositions.size();++i) {
    const auto position=flash_early_case::kPositions[i];const auto x=e::fresh(position);
    EXPECT_EQ(x.live_tokens(),expected[i]);EXPECT_EQ(x.live_tokens()+x.padding(),e::kStorage);
    EXPECT_EQ(x.visible_pools,(std::int64_t(position)+1)/4);
  }
  EXPECT_THROW((void)e::fresh(-1),std::runtime_error);
  EXPECT_THROW((void)e::fresh(std::numeric_limits<std::int32_t>::max()),std::runtime_error);
  for(std::int32_t tail=0;tail<4;++tail)EXPECT_EQ(e::carried(tail).live_tokens(),2048+tail);
}
TEST(GlmFlashOrdinaryEarlyHistoryPlan,OriginalMlxAndGgufParameterInputsUseActualHistory) {
  for(const auto position:flash_early_case::kPositions){flash_early_case::mlx(position);flash_early_case::gguf(position);}
}
TEST(GlmFlashOrdinaryEarlyHistoryPlan,OriginalFp8ParameterBranchUsesActualHistory) {
  for(const auto position:flash_early_case::kPositions)flash_early_case::official(position);
}
