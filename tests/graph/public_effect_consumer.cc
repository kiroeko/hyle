#include <hyle/compute/custom_kernel.h>
#include <hyle/compute/backends/cpu.h>
#include <hyle/graph/graph.h>
#include "native_effect_fixture.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
#if defined(HYLE_COMPUTE_VULKAN)
namespace n=c::vulkan;
#elif defined(HYLE_COMPUTE_APPLE)
namespace n=c::apple;
#endif
using hyle::Status;
template<class T>using Result=hyle::Result<T>;
Status require(bool value,const char *message){return value?Status::Ok():Status::Internal(message);}

// Consumer-owned resource operation, not a HYLE mathematical/copy kernel.
// The immutable schema owns its exact action; the explicit footprint declares
// the external memory it really touches. Caller keeps that backing alive.
struct Gate { std::atomic<int> phase{0}; };
struct Touch {
  static constexpr c::OperationId operation_id{0x464f554e44415449ULL,0x4f4e2d4546464543ULL};
  static constexpr std::string_view operation_name="consumer-resource-touch";
  std::int32_t *state=nullptr;
  std::int32_t *observed=nullptr;
  bool write=false;
  bool fail=false;
  Gate *gate=nullptr;
  c::OperationTraits operation_traits()const noexcept{
    return {.effect=c::EffectKind::kResourceAccess,.materializes=false,
            .resource_count=static_cast<std::uint8_t>(write?1:2)};
  }
  c::ResourceAccess resource_access(std::size_t port)const noexcept{
    return write||port==1?c::ResourceAccess::kWrite:c::ResourceAccess::kRead;
  }
  Status infer_output_descriptions(std::span<const c::TensorDesc> inputs,std::span<c::TensorDesc> outputs)const noexcept{
    return state&&observed&&inputs.empty()&&outputs.empty()?Status::Ok():Status::InvalidArgument("resource touch schema");
  }
};
constexpr c::ProviderRuntimeProperties runtime{
    .mode=c::ProviderRuntimeMode::kSharedQueue,.workspace_owner=c::ProviderWorkspaceOwner::kCaller,
    .replay_class=c::ProviderReplayClass::kPrepared,.blocks_host=true,.uses_private_threads=false,
    .supports_concurrent_invocations=true,.capture_compatible=false};
struct ResourceKernel {
  Touch action;
  struct Invocation {};
  Result<Invocation> create_invocation(m::BufferView,m::AllocatorRef)const noexcept{return Invocation{};}
  c::KernelCommandResult enqueue(Invocation&,c::detail::QueueSequence&,std::span<const c::ConstTensorView> inputs,
      std::span<const c::TensorView> outputs)const noexcept{
    if(!inputs.empty()||!outputs.empty())return c::KernelCommandResult::rejected(Status::InvalidArgument("touch ports"));
    if(action.write)*action.state=41;else *action.observed=*action.state;
    if(action.gate){
      int idle=0;
      if(action.gate->phase.compare_exchange_strong(idle,1))
        while(action.gate->phase.load()==1)action.gate->phase.wait(1);
    }
    return c::KernelCommandResult::accepted_command(action.fail?Status::DataLoss("accepted consumer write failed"):Status::Ok());
  }
};
struct Factory {
  Result<c::KernelProposal> propose(const c::KernelRequest &request,std::span<c::ResourceRequirement> inputs,
      std::span<c::ResourceRequirement> outputs)const noexcept{
    if(request.device.backend_id()!=c::cpu::kBackendId||!request.operation.custom_as<Touch>())
      return std::unexpected(Status::Unimplemented("consumer resource operation only"));
    if(!inputs.empty()||!outputs.empty())return std::unexpected(Status::InvalidArgument("touch request ports"));
    return c::KernelProposal{.route=1,.runtime=runtime,
      .cost={.workspace={},.persistent_bytes=sizeof(ResourceKernel),.copy_bytes=0,.repack_bytes=0,.synchronization_count=0,.dispatch_count=1}};
  }
  c::KernelPreparation prepare(const c::KernelPrepareRequest &request,m::AllocatorRef allocator)const noexcept{
    auto action=request.operation.custom_as<Touch>();
    if(request.route!=1||!action)return c::KernelPreparation::error(Status::InvalidArgument("touch preparation"));
    auto kernel=c::make_custom_kernel(ResourceKernel{*action},{.runtime=runtime,.persistent_bytes=sizeof(ResourceKernel),.invocation_private_bytes=0},allocator);
    return kernel?c::KernelPreparation::prepared(std::move(*kernel)):c::KernelPreparation::error(kernel.error());
  }
};

Status run(){
  HYLE_ASSIGN_OR_RETURN(auto provider,c::make_custom_provider(
      {{0x464f554e44415449ULL,0x4f4e2d50524f5649ULL},"consumer-resource","1"},c::cpu::kBackendId,Factory{}));
  HYLE_ASSIGN_OR_RETURN(auto cpu,c::Context::create(c::cpu::device(),provider));
  HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto gpu,n::Context::create());
#if defined(HYLE_COMPUTE_VULKAN)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(gpu,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(gpu,n::MemoryKind::kStaging));
  const auto staging_resource=staging.ref();
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(gpu));
  const auto staging_resource=device.ref();
#endif
  hyle::sched::Scheduler scheduler(1,m::system_allocator(),32);
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(),1);
  HYLE_ASSIGN_OR_RETURN(auto state,m::Buffer::create(m::system_resource(),4,64));
  HYLE_ASSIGN_OR_RETURN(auto mapping,state.view().map_host());
  HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());
  auto *word=::new(bytes.data())std::int32_t{7};std::int32_t observed=-1;
  HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto input_storage,m::Buffer::create(m::system_resource(),4,64));
  const std::array<float,1> value{3};HYLE_RETURN_IF_ERROR(input_storage.view().write(std::as_bytes(std::span(value))));
  HYLE_ASSIGN_OR_RETURN(auto input_tensor,c::ConstTensorView::create(desc,input_storage.view()));
  struct Drain{hyle::sched::Scheduler &s;~Drain(){s.drain();}}drain{scheduler};
  for(int scenario:{0,1,2}){
    const bool fail=scenario==1;Gate gate;
    *word=7;observed=-1;
    HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto resource,builder.effect_resource());
    HYLE_ASSIGN_OR_RETURN(auto observation,builder.effect_resource());
    HYLE_ASSIGN_OR_RETURN(auto write,builder.add_node(c::Operation::custom(Touch{word,&observed,true,fail,scenario==2?&gate:nullptr}),{}));
    HYLE_RETURN_IF_ERROR(builder.bind_resources(write,std::array{resource}));
    HYLE_ASSIGN_OR_RETURN(auto input,builder.input(desc));
    HYLE_ASSIGN_OR_RETURN(auto middle,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),std::array{input}));
    HYLE_ASSIGN_OR_RETURN(auto read,builder.add_node(c::Operation::custom(Touch{word,&observed,false,false}),{}));
    HYLE_RETURN_IF_ERROR(builder.bind_resources(read,std::array{resource,observation}));
    HYLE_RETURN_IF_ERROR(builder.depends_on(middle,write));HYLE_RETURN_IF_ERROR(builder.depends_on(read,middle));
    HYLE_ASSIGN_OR_RETURN(auto output,builder.result(middle));HYLE_RETURN_IF_ERROR(builder.publish(output));
    HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
    const std::array placements{g::NodePlacement{write,effect_test::kCpuTarget,g::PlacementStrength::kRequired},
        g::NodePlacement{middle,effect_test::kNativeTarget,g::PlacementStrength::kRequired},
        g::NodePlacement{read,effect_test::kCpuTarget,g::PlacementStrength::kRequired}};
    const std::array effects{g::EffectResourceBinding{resource,g::EffectResourceFootprint::memory(state.view())},
                            g::EffectResourceBinding{observation,g::EffectResourceFootprint::object(&observed)}};
    HYLE_ASSIGN_OR_RETURN(auto program,effect_test::compile(graph,cpu,gpu,device.ref(),staging_resource,
        {.effect_bindings=effects,.node_placements=placements}));
    HYLE_ASSIGN_OR_RETURN(auto slot,program.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto queues,effect_test::queues(program.report,queue));
    HYLE_ASSIGN_OR_RETURN(auto alias,c::ConstTensorView::create(desc,state.view()));
    HYLE_RETURN_IF_ERROR(require(slot.bind(input,alias).IsInvalidArgument(),"cross-fragment effect alias admitted"));
    HYLE_RETURN_IF_ERROR(slot.bind(input,input_tensor));HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,completion_waiters));
    auto terminal=[&]() -> Status {
      HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(queues));
      if(scenario!=2)return submitted.wait();
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
      while(gate.phase.load()==0&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
      if(gate.phase.load()!=1){
        gate.phase.store(2);gate.phase.notify_all();
        (void)submitted.request_cancel();(void)submitted.wait();
        return Status::DeadlineExceeded("consumer CPU prefix did not start");
      }
      auto cancel=submitted.request_cancel();
      gate.phase.store(2);gate.phase.notify_all();
      const auto retired=submitted.wait();
      return cancel.ok()?retired:cancel;
    }();
    if(scenario==2){HYLE_RETURN_IF_ERROR(require(terminal.IsCancelled()&&*word==41&&observed==-1,
        "cancelled CPU prefix did not retire before suppressing successors"));}
    else if(fail){HYLE_RETURN_IF_ERROR(require(terminal.IsDataLoss()&&*word==41&&observed==-1,"accepted effect prefix or failure propagation differs"));}
    else{
      HYLE_RETURN_IF_ERROR(terminal);
      HYLE_ASSIGN_OR_RETURN(auto result,slot.result(output));
      HYLE_RETURN_IF_ERROR(require(*word==41&&observed==41&&!slot.reset().ok(),"successful effect order or old result guard differs"));
      HYLE_RETURN_IF_ERROR(slot.close()); // Old result remains an owner.
      HYLE_ASSIGN_OR_RETURN(auto actual,effect_test::read(result,gpu,device,desc));
      HYLE_RETURN_IF_ERROR(require(actual==-3,"native successor math differs"));
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto resource,builder.effect_resource());
  HYLE_ASSIGN_OR_RETURN(auto action,builder.add_node(c::Operation::custom(Touch{word,&observed,true,false}),{}));
  HYLE_RETURN_IF_ERROR(builder.bind_resources(action,std::array{resource}));
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish()); // Effect-only, zero outputs.
  const std::array placement{g::NodePlacement{action,effect_test::kCpuTarget,g::PlacementStrength::kRequired}};
  const std::array effects{g::EffectResourceBinding{resource,g::EffectResourceFootprint::memory(state.view())}};
  HYLE_ASSIGN_OR_RETURN(auto program,effect_test::compile(graph,cpu,gpu,device.ref(),staging_resource,
      {.effect_bindings=effects,.node_placements=placement}));
  HYLE_ASSIGN_OR_RETURN(auto slot,program.executable.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto queues,effect_test::queues(program.report,queue));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,completion_waiters));*word=7;
  {HYLE_ASSIGN_OR_RETURN(auto completion,slot.submit(queues));HYLE_RETURN_IF_ERROR(completion.wait());
   HYLE_RETURN_IF_ERROR(require(*word==41,"zero-output CPU effect was pruned"));}
  HYLE_RETURN_IF_ERROR(slot.close());
  return Status::Ok();
}
} // namespace
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"public effect consumer: %.*s\n",int(status.message().size()),status.message().data());return 1;}std::puts("public CPU/native effect consumer PASS");}
