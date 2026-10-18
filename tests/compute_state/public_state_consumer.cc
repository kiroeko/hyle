#include <hyle/compute/state_clone.h>
#include <hyle/compute/leased_operation.h>
#include <hyle/compute/model_ops.h>
#include <hyle/graph/leased_invocation.h>
#if defined(HYLE_COMPUTE_CPU)
#include <hyle/graph/cpu.h>
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include <hyle/graph/cuda.h>
#include <hyle/compute/backends/cuda.h>
#endif
#if defined(HYLE_COMPUTE_APPLE)
#include <hyle/graph/apple.h>
#include <hyle/compute/apple_memory.h>
#include <mlx/ops.h>
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include <hyle/graph/vulkan.h>
#include <hyle/compute/vulkan_memory.h>
#endif
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
using hyle::Status;
template<class T>using Result=hyle::Result<T>;
Status require(bool value,const char *message){return value?Status::Ok():Status::Internal(message);}
Result<c::TensorDesc> dense(std::initializer_list<std::int64_t> shape){return c::TensorDesc::dense(c::DataType::kFloat32,{shape.begin(),shape.size()});}
Status close_values(std::span<const float> actual,std::span<const float> expected) {
  HYLE_RETURN_IF_ERROR(require(actual.size()==expected.size(),"state consumer output extent"));
  for(std::size_t i=0;i<actual.size();++i)
    if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>2e-5f+2e-5f*std::abs(expected[i]))
      return Status::Internal("state consumer independent numerical oracle differs");
  return Status::Ok();
}
struct Work {
  c::Operation operation;
  std::vector<c::TensorDesc> inputs,outputs;
};
struct Logical {g::Graph graph;std::vector<g::ValueId> inputs,outputs;};
Result<Logical> logical(const Work &work) {
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());Logical result;
  for(const auto &d:work.inputs){HYLE_ASSIGN_OR_RETURN(auto id,builder.input(d));result.inputs.push_back(id);}
  HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(work.operation,result.inputs));
  for(std::size_t i=0;i<work.outputs.size();++i){HYLE_ASSIGN_OR_RETURN(auto id,builder.result(node,i));HYLE_RETURN_IF_ERROR(builder.publish(id));result.outputs.push_back(id);}
  HYLE_ASSIGN_OR_RETURN(result.graph,std::move(builder).finish());return result;
}

#if defined(HYLE_COMPUTE_CPU) || defined(HYLE_COMPUTE_CUDA)
template<class Compile>struct CommonRoute {
  c::Context &context;c::Queue &queue;m::MemoryResourceRef resource;c::LeasePool &pool;Compile compile;
  Result<c::ReadLease> load(const c::TensorDesc &description,std::span<const float> data) {
    HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
    if(bytes!=data.size_bytes())return std::unexpected(Status::InvalidArgument("state consumer initial bytes"));
    HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));
    HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(data)));return writer.publish();
  }
  Result<std::vector<float>> read(const c::ReadLease &lease,const c::TensorDesc &) {
    std::vector<float> values(lease.size()/sizeof(float));HYLE_RETURN_IF_ERROR(lease.view()->read(std::as_writable_bytes(std::span(values))));return values;
  }
  Result<c::StateCloneRoute> clones(c::StateStore &store,std::size_t bytes,m::MemoryResourceRef source={}) {
    const std::array specs{c::CopyResourceSpec{source.valid()?source:resource,bytes},c::CopyResourceSpec{resource,bytes}};
    HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
    auto *endpoint=&queue;
    return c::StateCloneRoute::prepare(store,{.resource=resource,.capacity_bytes=64u*1024u,.slots=2},program,
      [endpoint](const c::CopyBatchProgram &p) noexcept {return c::CopyBatchInvocation::create(p,*endpoint);});
  }
  Status update(c::StateWriter &writer,std::span<const float> data) {
    HYLE_RETURN_IF_ERROR(writer.lease().view()->write(std::as_bytes(data)));return writer.publish();
  }
  Result<std::vector<c::ReadLease>> math(const Work &work,std::span<const c::ReadLease> leases,bool graph) {
    std::printf("F4 common %s op=%.*s\n",graph?"Graph":"Direct",int(c::operation_name(work.operation).size()),c::operation_name(work.operation).data());std::fflush(stdout);
    std::vector<c::ReadLease> result;
    if(!graph) {
      std::vector<c::ConstTensorView> views;std::vector<c::TensorView> output_views;
      std::vector<c::WriteLease> outputs;std::vector<c::LeasedInput> inputs;std::vector<c::LeasedOutput> bound_outputs;
      outputs.reserve(work.outputs.size());
      for(std::size_t i=0;i<leases.size();++i){HYLE_ASSIGN_OR_RETURN(auto view,leases[i].tensor(work.inputs[i]));views.push_back(view);inputs.emplace_back(work.inputs[i],leases[i]);}
      for(const auto &desc:work.outputs){HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));outputs.push_back(std::move(writer));HYLE_ASSIGN_OR_RETURN(auto view,outputs.back().tensor(desc));output_views.push_back(view);}
      for(std::size_t i=0;i<outputs.size();++i)bound_outputs.emplace_back(work.outputs[i],outputs[i]);
      HYLE_ASSIGN_OR_RETURN(auto prepared,c::prepare(context,work.operation,views,output_views));
      c::WriteLease workspace;const auto required=prepared.workspace_requirement();
      if(required.size){HYLE_ASSIGN_OR_RETURN(workspace,pool.allocate(required.size,(std::max)(required.alignment,std::size_t{256})));}
      HYLE_ASSIGN_OR_RETURN(auto invocation,c::LeasedOperationInvocation::create(prepared,std::move(workspace)));
      HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue,inputs,bound_outputs));HYLE_RETURN_IF_ERROR(done.wait());
      for(auto &writer:outputs){HYLE_ASSIGN_OR_RETURN(auto ready,writer.publish());result.push_back(std::move(ready));}
    } else {
      HYLE_ASSIGN_OR_RETURN(auto graph_work,logical(work));HYLE_ASSIGN_OR_RETURN(auto compiled,compile(graph_work.graph));
      HYLE_ASSIGN_OR_RETURN(auto invocation,g::LeasedInvocation::create(compiled.executable));
      for(std::size_t i=0;i<leases.size();++i)HYLE_RETURN_IF_ERROR(invocation.bind(graph_work.inputs[i],work.inputs[i],leases[i]));
      for(std::size_t i=0;i<work.outputs.size();++i){HYLE_ASSIGN_OR_RETURN(auto bytes,work.outputs[i].required_bytes());HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));HYLE_RETURN_IF_ERROR(invocation.bind(graph_work.outputs[i],work.outputs[i],std::move(writer)));}
      HYLE_ASSIGN_OR_RETURN(auto done,invocation.submit(queue));HYLE_RETURN_IF_ERROR(done.wait());
      for(auto id:graph_work.outputs){HYLE_ASSIGN_OR_RETURN(auto output,invocation.output(id));HYLE_ASSIGN_OR_RETURN(auto lease,output.storage_lease());result.push_back(std::move(lease));}
    }
    return result;
  }
};
#endif

#if defined(HYLE_COMPUTE_APPLE)
struct AppleRoute {
  c::apple::Context &context;c::apple::MemoryResource &memory;c::LeasePool &pool;
  Result<c::ReadLease> load(const c::TensorDesc &desc,std::span<const float> data) {
    HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());
    if(bytes!=data.size_bytes())return std::unexpected(Status::InvalidArgument("Apple initial state extent"));
    HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(bytes));HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(data)));return writer.publish();
  }
  Result<std::vector<float>> read(const c::ReadLease &lease,const c::TensorDesc &desc) {
    HYLE_ASSIGN_OR_RETURN(auto array,memory.binding(lease,desc));std::vector<float> values(lease.size()/sizeof(float));
    const auto *data=array.data<float>();std::copy_n(data,values.size(),values.data());return values;
  }
  Result<c::StateCloneRoute> clones(c::StateStore &store,std::size_t bytes,m::MemoryResourceRef source={}) {
    const std::array specs{c::CopyResourceSpec{source.valid()?source:memory.ref(),bytes},c::CopyResourceSpec{memory.ref(),bytes}};
    HYLE_ASSIGN_OR_RETURN(auto program,c::apple::CopyBatchProgram::prepare(context,specs));
    return c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=64u*1024u,.slots=2},program,
      [](const c::apple::CopyBatchProgram &p) noexcept {return c::apple::CopyBatchInvocation::create(p);});
  }
  Status update(c::StateWriter &writer,std::span<const float> data) {
    HYLE_RETURN_IF_ERROR(writer.lease().view()->write(std::as_bytes(data)));return writer.publish();
  }
  Result<std::vector<c::ReadLease>> math(const Work &work,std::span<const c::ReadLease> leases,bool graph) {
    std::vector<mlx::core::array> inputs;
    for(std::size_t i=0;i<leases.size();++i){HYLE_ASSIGN_OR_RETURN(auto array,memory.binding(leases[i],work.inputs[i]));inputs.push_back(std::move(array));}
    c::apple::Program program;
    if(graph){HYLE_ASSIGN_OR_RETURN(auto graph_work,logical(work));HYLE_ASSIGN_OR_RETURN(program,g::apple::compile(context,graph_work.graph));}
    else {HYLE_ASSIGN_OR_RETURN(program,context.prepare(work.operation,inputs));}
    HYLE_ASSIGN_OR_RETURN(auto invocation,program.create_invocation());HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
    HYLE_ASSIGN_OR_RETURN(auto submitted,invocation.submit());HYLE_ASSIGN_OR_RETURN(auto output,submitted.wait());
    std::vector<c::ReadLease> result;
    for(std::size_t i=0;i<work.outputs.size();++i){HYLE_ASSIGN_OR_RETURN(auto lease,memory.import_readonly(output->arrays()[i],work.outputs[i]));result.push_back(std::move(lease));}
    return result;
  }
};
#endif

#if defined(HYLE_COMPUTE_VULKAN)
struct VulkanRoute {
  c::vulkan::Context &context;c::vulkan::MemoryResource &memory;c::vulkan::MemoryResource &staging;
  c::LeasePool &host_pool;
  Result<c::ReadLease> load(const c::TensorDesc &desc,std::span<const float> data) {
    HYLE_ASSIGN_OR_RETURN(auto tensor,context.upload(desc,std::as_bytes(data)));return memory.import_readonly(tensor);
  }
  Result<std::vector<float>> read(const c::ReadLease &lease,const c::TensorDesc &desc) {
    HYLE_ASSIGN_OR_RETURN(auto tensor,memory.binding(lease,desc));std::vector<float> values(lease.size()/sizeof(float));
    HYLE_RETURN_IF_ERROR(context.download(tensor,std::as_writable_bytes(std::span(values))));return values;
  }
  Result<c::StateCloneRoute> clones(c::StateStore &store,std::size_t bytes,m::MemoryResourceRef source={}) {
    const std::array specs{c::CopyResourceSpec{source.valid()?source:memory.ref(),bytes},c::CopyResourceSpec{memory.ref(),bytes}};
    HYLE_ASSIGN_OR_RETURN(auto program,c::vulkan::CopyBatchProgram::prepare(context,specs));
    return c::StateCloneRoute::prepare(store,{.resource=memory.ref(),.capacity_bytes=64u*1024u,.slots=2},program,
      [](const c::vulkan::CopyBatchProgram &p) noexcept {return c::vulkan::CopyBatchInvocation::create(p);});
  }
  Status update(c::StateWriter &writer,std::span<const float> data) {
    HYLE_ASSIGN_OR_RETURN(auto host_writer,host_pool.allocate(data.size_bytes()));HYLE_RETURN_IF_ERROR(host_writer.view()->write(std::as_bytes(data)));
    HYLE_ASSIGN_OR_RETURN(auto host,host_writer.publish());
    const std::array specs{c::CopyResourceSpec{staging.ref(),data.size_bytes()},c::CopyResourceSpec{memory.ref(),data.size_bytes()}};
    HYLE_ASSIGN_OR_RETURN(auto program,c::vulkan::CopyBatchProgram::prepare(context,specs));
    HYLE_ASSIGN_OR_RETURN(auto invocation,c::vulkan::CopyBatchInvocation::create(program));
    HYLE_RETURN_IF_ERROR(invocation.bind_source({0},host));HYLE_RETURN_IF_ERROR(invocation.bind_mutable({1},std::move(writer.lease())));
    const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,data.size_bytes()}}};
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto submitted,invocation.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
    HYLE_ASSIGN_OR_RETURN(auto ready,submitted.output({0},{1},0,data.size_bytes()));submitted={};HYLE_RETURN_IF_ERROR(invocation.reset());
    return writer.complete(ready);
  }
  Result<std::vector<c::ReadLease>> math(const Work &work,std::span<const c::ReadLease> leases,bool graph) {
    std::vector<c::vulkan::Tensor> inputs;std::vector<c::vulkan::Argument> arguments;
    for(std::size_t i=0;i<leases.size();++i){HYLE_ASSIGN_OR_RETURN(auto tensor,memory.binding(leases[i],work.inputs[i]));inputs.push_back(std::move(tensor));arguments.emplace_back(work.inputs[i]);}
    c::vulkan::Program program;
    if(graph){HYLE_ASSIGN_OR_RETURN(auto graph_work,logical(work));HYLE_ASSIGN_OR_RETURN(program,g::vulkan::compile(context,graph_work.graph));}
    else {HYLE_ASSIGN_OR_RETURN(program,c::vulkan::prepare(context,work.operation,std::span<const c::vulkan::Argument>(arguments)));}
    HYLE_ASSIGN_OR_RETURN(auto invocation,program.create_invocation());HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
    HYLE_ASSIGN_OR_RETURN(auto submitted,invocation.submit());HYLE_ASSIGN_OR_RETURN(auto output,submitted.wait());
    std::vector<c::ReadLease> result;
    for(const auto &tensor:output->tensors()){HYLE_ASSIGN_OR_RETURN(auto lease,memory.import_readonly(tensor));result.push_back(std::move(lease));}
    return result;
  }
};
#endif

#include "mtp_prefix_consumer.h"
#if defined(HYLE_COMPUTE_CPU) && defined(HYLE_COMPUTE_CUDA)
#include "mtp_hybrid_consumer.h"
#endif

template<class Route>Status exercise(Route &route,const char *name) {
  constexpr std::size_t depth=128,channels=128; // locked ggml CUDA SSM_CONV requires channels % 128 == 0
  HYLE_ASSIGN_OR_RETURN(auto state_desc,dense({1,1,depth,depth}));
  HYLE_ASSIGN_OR_RETURN(auto window_desc,dense({1,4,channels}));
  HYLE_ASSIGN_OR_RETURN(auto page_desc,dense({2,depth}));
  const std::array specs{c::StateEntrySpec{.description=state_desc},c::StateEntrySpec{.description=window_desc},
      c::StateEntrySpec{.description=page_desc},c::StateEntrySpec{.description=page_desc},
      c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=32}};
  HYLE_ASSIGN_OR_RETURN(auto schema,c::StateSchema::create(specs));HYLE_ASSIGN_OR_RETURN(auto store,c::StateStore::create(schema));
  std::array<c::StateEntryId,5> ids;for(std::size_t i=0;i<ids.size();++i){HYLE_ASSIGN_OR_RETURN(ids[i],schema.entry(i));}
  std::vector<float> state(depth*depth),window(4*channels),page(2*depth);
  for(std::size_t i=0;i<state.size();++i)state[i]=float(int(i%19)-9)*0.002f;
  for(std::size_t i=0;i<window.size();++i)window[i]=float(int(i%13)-6)*0.03125f;
  for(std::size_t i=0;i<page.size();++i)page[i]=float(i)*0.00390625f;
  const auto original_state=state,original_window=window,original_page=page;
  HYLE_ASSIGN_OR_RETURN(auto initial,route.load(state_desc,state));HYLE_ASSIGN_OR_RETURN(auto conv,route.load(window_desc,window));
  HYLE_ASSIGN_OR_RETURN(auto zero,route.load(page_desc,page));HYLE_ASSIGN_OR_RETURN(auto one,route.load(page_desc,page));
  std::array<std::uint64_t,4> layout{0,2,11,17}; // consumer-owned positions/page IDs; F5 math not claimed
  HYLE_ASSIGN_OR_RETURN(auto base,store.import(std::array{c::StateMember{ids[0],initial},c::StateMember{ids[1],conv},
      c::StateMember{ids[2],zero},c::StateMember{ids[3],one},c::StateMember{ids[4],{},std::as_bytes(std::span(layout))}}));
  HYLE_ASSIGN_OR_RETURN(auto head,c::StateHead::create(store,base));HYLE_ASSIGN_OR_RETURN(auto clones,route.clones(store,window.size()*sizeof(float)>page.size()*sizeof(float)?window.size()*sizeof(float):page.size()*sizeof(float)));
  Work recurrent{c::Operation(c::GatedDeltaNet{}),{}, {}};
  std::array<c::ReadLease,5> fixed;
  std::array<std::vector<float>,5> values;
  for(std::size_t i=0;i<fixed.size();++i) {
    HYLE_ASSIGN_OR_RETURN(auto desc,dense({1,1,1,static_cast<std::int64_t>(i<3?depth:1)}));recurrent.inputs.push_back(desc);
    values[i].resize(i<3?depth:1);
    for(std::size_t j=0;j<values[i].size();++j)values[i][j]=i==3?-0.02f:i==4?0.25f:float(int(j%(5+2*i))-2)*0.0078125f;
    HYLE_ASSIGN_OR_RETURN(fixed[i],route.load(desc,values[i]));
  }
  recurrent.inputs.push_back(state_desc);recurrent.outputs={recurrent.inputs[2],state_desc};
  HYLE_ASSIGN_OR_RETURN(auto weights_desc,dense({channels,4}));HYLE_ASSIGN_OR_RETURN(auto conv_output,dense({1,1,channels}));
  std::vector<float> weights(channels*4);for(std::size_t i=0;i<weights.size();++i)weights[i]=float(1+i%7)/64.f;
  HYLE_ASSIGN_OR_RETURN(auto filter,route.load(weights_desc,weights));
  Work convolution{c::model_ops::causal_conv1d(),{window_desc,weights_desc},{conv_output}};
  for(unsigned step=0;step<2;++step) {
    HYLE_ASSIGN_OR_RETURN(auto current,head.current());HYLE_ASSIGN_OR_RETURN(auto previous,current.member(ids[0]));
    std::array<c::ReadLease,6> inputs{fixed[0],fixed[1],fixed[2],fixed[3],fixed[4],previous};
    auto next_state=state;std::vector<float> scores(depth);
    for(std::size_t row=0;row<depth;++row) {
      double projection=0;for(std::size_t col=0;col<depth;++col)projection+=double(state[row*depth+col])*std::exp(double(values[3][0]))*values[1][col];
      const double delta=values[4][0]*(values[2][row]-projection);double score=0;
      for(std::size_t col=0;col<depth;++col){const double value=double(state[row*depth+col])*std::exp(double(values[3][0]))+delta*values[1][col];next_state[row*depth+col]=float(value);score+=value*values[0][col];}
      scores[row]=float(score/std::sqrt(double(depth)));
    }
    HYLE_ASSIGN_OR_RETURN(auto direct,route.math(recurrent,inputs,false));HYLE_ASSIGN_OR_RETURN(auto graph,route.math(recurrent,inputs,true));
    for(const auto &result:{direct,graph}){HYLE_ASSIGN_OR_RETURN(auto actual_score,route.read(result[0],recurrent.outputs[0]));HYLE_RETURN_IF_ERROR(close_values(actual_score,scores));HYLE_ASSIGN_OR_RETURN(auto actual_state,route.read(result[1],state_desc));HYLE_RETURN_IF_ERROR(close_values(actual_state,next_state));}
    HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(current));HYLE_ASSIGN_OR_RETURN(auto loser,store.begin(current));
    HYLE_RETURN_IF_ERROR(draft.replace(ids[0],direct[1]));HYLE_RETURN_IF_ERROR(loser.replace(ids[0],graph[1]));
    std::move(window.begin()+channels,window.end(),window.begin());for(std::size_t i=0;i<channels;++i)window[3*channels+i]=float(step+1)*0.0625f;
    HYLE_ASSIGN_OR_RETURN(auto window_copy,draft.clone_for_write(ids[1],clones));HYLE_ASSIGN_OR_RETURN(auto window_writer,window_copy.wait());HYLE_RETURN_IF_ERROR(route.update(window_writer,window));
    page[0]=float(step+2);HYLE_ASSIGN_OR_RETURN(auto page_copy,draft.clone_for_write(ids[3],clones));HYLE_ASSIGN_OR_RETURN(auto page_writer,page_copy.wait());HYLE_RETURN_IF_ERROR(route.update(page_writer,page));
    ++layout[0];HYLE_RETURN_IF_ERROR(draft.replace_metadata(ids[4],std::as_bytes(std::span(layout))));
    HYLE_ASSIGN_OR_RETURN(auto candidate,draft.finalize());HYLE_ASSIGN_OR_RETURN(auto alternate,loser.finalize());
    HYLE_RETURN_IF_ERROR(head.publish(current.identity(),candidate));HYLE_RETURN_IF_ERROR(require(head.publish(current.identity(),alternate).IsUnavailable(),"state stale base won publication"));
    HYLE_ASSIGN_OR_RETURN(auto conv_state,candidate.member(ids[1]));
    std::array<c::ReadLease,2> conv_inputs{conv_state,filter};std::vector<float> conv_expected(channels);
    for(std::size_t i=0;i<channels;++i)for(std::size_t tap=0;tap<4;++tap)conv_expected[i]+=window[tap*channels+i]*weights[i*4+tap];
    for(bool use_graph:{false,true}){HYLE_ASSIGN_OR_RETURN(auto output,route.math(convolution,conv_inputs,use_graph));HYLE_ASSIGN_OR_RETURN(auto actual,route.read(output[0],conv_output));HYLE_RETURN_IF_ERROR(close_values(actual,conv_expected));}
    HYLE_ASSIGN_OR_RETURN(auto untouched,candidate.member(ids[2]));HYLE_ASSIGN_OR_RETURN(auto unchanged,route.read(untouched,page_desc));HYLE_RETURN_IF_ERROR(close_values(unchanged,original_page));
    HYLE_ASSIGN_OR_RETURN(auto dirty,candidate.member(ids[3]));HYLE_ASSIGN_OR_RETURN(auto changed,route.read(dirty,page_desc));HYLE_RETURN_IF_ERROR(close_values(changed,page));
    state=std::move(next_state);
  }
  HYLE_ASSIGN_OR_RETURN(auto old,base.member(ids[0]));HYLE_ASSIGN_OR_RETURN(auto old_values,route.read(old,state_desc));HYLE_RETURN_IF_ERROR(close_values(old_values,original_state));
  HYLE_ASSIGN_OR_RETURN(auto old_window,base.member(ids[1]));HYLE_ASSIGN_OR_RETURN(auto old_window_values,route.read(old_window,window_desc));HYLE_RETURN_IF_ERROR(close_values(old_window_values,original_window));
  // Dense native pure output can also be cloned using its actual readonly
  // endpoint. Keep that source owner through this prepared copy route.
  HYLE_ASSIGN_OR_RETURN(auto native_current,head.current());HYLE_ASSIGN_OR_RETURN(auto native_source,native_current.member(ids[0]));
  HYLE_ASSIGN_OR_RETURN(auto native_schema,c::StateSchema::create(std::array{c::StateEntrySpec{.description=state_desc}}));
  HYLE_ASSIGN_OR_RETURN(auto native_id,native_schema.entry(0));HYLE_ASSIGN_OR_RETURN(auto native_store,c::StateStore::create(native_schema));
  HYLE_ASSIGN_OR_RETURN(auto native_base,native_store.import(std::array{c::StateMember{native_id,native_source}}));
  HYLE_ASSIGN_OR_RETURN(auto native_view,native_source.view());
  HYLE_ASSIGN_OR_RETURN(auto native_route,route.clones(native_store,native_source.size(),native_view.resource()));
  HYLE_ASSIGN_OR_RETURN(auto native_draft,native_store.begin(native_base));HYLE_ASSIGN_OR_RETURN(auto native_copy,native_draft.clone_for_write(native_id,native_route));
  HYLE_ASSIGN_OR_RETURN(auto native_writer,native_copy.wait());HYLE_RETURN_IF_ERROR(native_writer.publish());HYLE_ASSIGN_OR_RETURN(auto native_version,native_draft.finalize());
  HYLE_ASSIGN_OR_RETURN(auto copied_native,native_version.member(native_id));HYLE_ASSIGN_OR_RETURN(auto copied_values,route.read(copied_native,state_desc));HYLE_RETURN_IF_ERROR(close_values(copied_values,state));
  HYLE_ASSIGN_OR_RETURN(auto abandoned,store.begin(base));{HYLE_ASSIGN_OR_RETURN(auto ticket,abandoned.clone_for_write(ids[3],clones));
    const auto cancel=ticket.request_cancel();HYLE_RETURN_IF_ERROR(require(cancel.ok() || cancel.IsUnimplemented(),"state cancellation returned an unexpected native error"));
    auto cancelled=ticket.wait();HYLE_RETURN_IF_ERROR(require(!cancelled,"state cancelled clone returned a writer"));}
  HYLE_RETURN_IF_ERROR(require(abandoned.finalize().error().IsCancelled(),"state cancellation published"));
  store.collect().IgnoreError();HYLE_RETURN_IF_ERROR(clones.rearm());
  // Orthogonal asset matrix, same schema/ownership/clone protocol; no model fields in HYLE.
  HYLE_ASSIGN_OR_RETURN(auto asset_desc,dense({4,4}));std::vector<float> asset(16,3.f);
  HYLE_ASSIGN_OR_RETURN(auto asset_schema,c::StateSchema::create(std::array{c::StateEntrySpec{.description=asset_desc}}));HYLE_ASSIGN_OR_RETURN(auto asset_id,asset_schema.entry(0));
  HYLE_ASSIGN_OR_RETURN(auto assets,c::StateStore::create(asset_schema));HYLE_ASSIGN_OR_RETURN(auto asset_value,route.load(asset_desc,asset));
  HYLE_ASSIGN_OR_RETURN(auto original,assets.import(std::array{c::StateMember{asset_id,asset_value}}));HYLE_ASSIGN_OR_RETURN(auto update,assets.begin(original));
  HYLE_ASSIGN_OR_RETURN(auto asset_route,route.clones(assets,asset.size()*sizeof(float)));HYLE_ASSIGN_OR_RETURN(auto asset_copy,update.clone_for_write(asset_id,asset_route));HYLE_ASSIGN_OR_RETURN(auto asset_writer,asset_copy.wait());
  asset[0]=5.f;HYLE_RETURN_IF_ERROR(route.update(asset_writer,asset));HYLE_ASSIGN_OR_RETURN(auto updated,update.finalize());
  HYLE_ASSIGN_OR_RETURN(auto versioned_asset,updated.member(asset_id));Work negate{c::Operation(c::Unary{c::UnaryKind::kNeg}),{asset_desc},{asset_desc}};
  std::array<c::ReadLease,1> asset_inputs{versioned_asset};HYLE_ASSIGN_OR_RETURN(auto rendered,route.math(negate,asset_inputs,true));
  HYLE_ASSIGN_OR_RETURN(auto asset_output,route.read(rendered[0],asset_desc));for(auto &value:asset)value=-value;HYLE_RETURN_IF_ERROR(close_values(asset_output,asset));
  HYLE_RETURN_IF_ERROR(assets.close());HYLE_ASSIGN_OR_RETURN(auto still_old,original.member(asset_id));asset_inputs[0]=still_old;
  HYLE_ASSIGN_OR_RETURN(auto after_close,route.math(negate,asset_inputs,false));HYLE_ASSIGN_OR_RETURN(auto old_asset_output,route.read(after_close[0],asset_desc));HYLE_RETURN_IF_ERROR(close_values(old_asset_output,std::vector<float>(16,-3.f)));
  HYLE_RETURN_IF_ERROR(store.drain());HYLE_RETURN_IF_ERROR(assets.drain());
  std::printf("F4 %s: two-step recurrent/conv + layout/state + asset Direct/Graph/old-version/cancel PASS\n",name);return Status::Ok();
}
Status run() {
#if defined(HYLE_COMPUTE_CPU)
  {HYLE_ASSIGN_OR_RETURN(auto context,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(m::system_resource(),2u*1024u*1024u));
   CommonRoute route{context,queue,m::system_resource(),pool,[&](const g::Graph &graph){return g::compile(graph,context);}};HYLE_RETURN_IF_ERROR(exercise(route,"CPU"));HYLE_RETURN_IF_ERROR(exercise_mtp_prefixes(route,"CPU"));}
#endif
#if defined(HYLE_COMPUTE_CUDA)
  {HYLE_ASSIGN_OR_RETURN(auto context,g::cuda::create_context());HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());HYLE_ASSIGN_OR_RETURN(auto memory,c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),2u*1024u*1024u));
   CommonRoute route{context,queue,memory.ref(),pool,[&](const g::Graph &graph){return g::cuda::compile(graph,context,memory.ref());}};HYLE_RETURN_IF_ERROR(exercise(route,"CUDA"));HYLE_RETURN_IF_ERROR(exercise_mtp_prefixes(route,"CUDA"));}
#endif
#if defined(HYLE_COMPUTE_APPLE)
  {HYLE_ASSIGN_OR_RETURN(auto context,c::apple::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,c::apple::MemoryResource::create(context));HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),2u*1024u*1024u));AppleRoute route{context,memory,pool};HYLE_RETURN_IF_ERROR(exercise(route,"Apple"));HYLE_RETURN_IF_ERROR(exercise_mtp_prefixes(route,"Apple"));}
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  {HYLE_ASSIGN_OR_RETURN(auto context,c::vulkan::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,c::vulkan::MemoryResource::create(context,c::vulkan::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,c::vulkan::MemoryResource::create(context,c::vulkan::MemoryKind::kStaging));HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),64u*1024u));
   VulkanRoute route{context,memory,staging,pool};HYLE_RETURN_IF_ERROR(exercise(route,"Vulkan"));HYLE_RETURN_IF_ERROR(exercise_mtp_prefixes(route,"Vulkan"));}
#endif
#if defined(HYLE_COMPUTE_CPU) && defined(HYLE_COMPUTE_CUDA)
  {
    HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto gpu,g::cuda::create_context());HYLE_ASSIGN_OR_RETURN(auto gpu_queue,gpu.create_queue());
    HYLE_ASSIGN_OR_RETURN(auto pinned,c::cuda::MemoryResource::create(gpu,c::cuda::MemoryKind::kPinnedHost));
    HYLE_ASSIGN_OR_RETURN(auto device,c::cuda::MemoryResource::create(gpu,c::cuda::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(pinned.ref(),2u*1024u*1024u));
    HYLE_ASSIGN_OR_RETURN(auto device_pool,c::LeasePool::create(device.ref(),2u*1024u*1024u));
    const std::array storage{g::StorageDomain{.resource=pinned.ref()}};
    const std::array targets{g::ExecutionTarget{.context=&cpu,.default_storage=g::StorageDomainId{0}}};
    CommonRoute host{cpu,cpu_queue,pinned.ref(),host_pool,[&](const g::Graph &graph){return g::compile(graph,{.storage_domains=storage,.targets=targets});}};
    CommonRoute accelerator{gpu,gpu_queue,device.ref(),device_pool,[&](const g::Graph &graph){return g::cuda::compile(graph,gpu,device.ref());}};
    HYLE_RETURN_IF_ERROR(exercise_mtp_hybrid(host,accelerator,"CPU to CUDA"));
    HYLE_RETURN_IF_ERROR(exercise_mtp_hybrid(accelerator,host,"CUDA to CPU"));
  }
#endif
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(status.ok())return 0;char text[768]{};status.ToString(text,sizeof(text));std::fprintf(stderr,"F4 public consumer: %s\n",text);return 1;}
