#include "native_ragged_reference.h"
#include "reference.h"
#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"
#include <array>
#include <cstdio>
#include <cstdlib>

namespace hyle::bench::resident {
namespace c=compute;namespace v=c::vulkan;namespace d=v::detail;
namespace {
template<class T>T take(Result<T> value){if(!value){std::fprintf(stderr,"ragged original reference: %.*s\n",int(value.error().message().size()),value.error().message().data());std::exit(2);}return std::move(*value);}
void check(Status status){if(!status.ok()){std::fprintf(stderr,"ragged original reference: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
}
v::Program ragged_reference_program(const v::Context&context,
    const c::BatchLayoutVersion&layout,const c::RaggedAttention&attention){
  const auto cap=layout.schema().capacity();if(cap!=attention.capacity||cap.storage!=c::BatchStorage::kRagged||cap.query_type!=c::DataType::kFloat16||!layout.facts().query_rows)std::exit(2);
  auto state=std::make_shared<d::ProgramState>();state->runtime=d::Access::runtime(context);
  const auto q=take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{cap.max_query_rows,cap.query_heads,cap.qk_head_dim}));
  const auto k=take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{cap.max_kv_rows,cap.kv_heads,cap.qk_head_dim}));
  state->values={{q,{}},{k,{}},{k,{}}};state->inputs={0,1,2};state->input_descriptions={q,k,k};state->aliases={{0,0},{1,0},{2,0}};
  const auto host=layout.host();std::vector<std::uint32_t> parts;
  for(std::size_t request=0;request<layout.facts().requests;++request){
    const auto nq=host.query_indptr[request+1]-host.query_indptr[request],nk=host.kv_indptr[request+1]-host.kv_indptr[request];if(!nq)continue;
    const auto qd=take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{1,nq,cap.query_heads,cap.qk_head_dim}));
    const auto kd=take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,4>{1,nk,cap.kv_heads,cap.qk_head_dim}));
    // The already-qualified original shader recipe, including its tuning,
    // F32 query/output adapters, mask optimization and split-K reduction.
    auto child=reference_program(context,qd,kd,attention.causal);const auto &recipe=*d::Access::program(child);
    const auto base=static_cast<std::uint32_t>(state->values.size());
    state->values.insert(state->values.end(),recipe.values.begin(),recipe.values.end());
    for(std::uint32_t id=0;id<recipe.values.size();++id)state->aliases.push_back({base+recipe.aliases[id].root,recipe.aliases[id].byte_offset});
    state->aliases[base]={0,std::size_t(host.query_indptr[request])*cap.query_heads*cap.qk_head_dim*2};
    for(unsigned port=1;port<3;++port)state->aliases[base+port]={port,std::size_t(host.kv_indptr[request])*cap.kv_heads*cap.qk_head_dim*2};
    for(auto dispatch:recipe.dispatches){for(auto &id:dispatch.values)id+=base;state->dispatches.push_back(std::move(dispatch));}
    const auto packed=static_cast<std::uint32_t>(state->values.size());
    state->values.push_back({take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{nq,cap.query_heads,cap.qk_head_dim})),{}});
    state->aliases.push_back({base+recipe.outputs[0],0});parts.push_back(packed);
  }
  const auto output=static_cast<std::uint32_t>(state->values.size());const auto desc=take(c::TensorDesc::dense(cap.query_type,std::array<std::int64_t,3>{layout.facts().query_rows,cap.query_heads,cap.qk_head_dim}));
  state->values.push_back({desc,{}});state->aliases.push_back({output,0});state->outputs={output};state->output_descriptions={desc};
  // Joining request outputs is an existing storage operation, not attention
  // arithmetic. Both arms return one packed owning result through Program.
  check(d::lower_tensor(*state,{c::Operation(c::Concat{.axis=0}),std::move(parts),{output},{},{1,1,1},{}}));
  check(d::plan_storage(*state));return d::Access::program(std::move(state));
}
}
