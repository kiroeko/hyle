#include "native_paged_joint_risk_fixture.h"
#include <chrono>
#include <thread>
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include <hyle/graph/apple.h>
#else
#include <hyle/graph/vulkan.h>
#endif
#endif
#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include "../compute_batch/native_apple_failure_fixture.h"
#include "hyle/compute/detail/apple_copy_stream_access.h"
#else
#include "../compute_batch/native_vulkan_failure_fixture.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#endif

namespace {
namespace jr=hyle_joint_page_risk;namespace r=jr::r;namespace c=r::c;namespace n=r::n;namespace g=jr::g;namespace route=jr::route;
using hyle::Status;
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
constexpr std::size_t source_capacity=4,cancel_source=1;
constexpr bool native_owner_sources=true;
#else
constexpr std::size_t source_capacity=2,cancel_source=source_capacity-1;
constexpr bool native_owner_sources=false;
#endif
enum class Mode { success,hole,cancel_before,cancel_after,abandon,allocation,capacity,unknown,native_rejection,closed,budget };
Status episode(Mode mode){
  hyle_f5_fault_fixture::ConcurrentAllocator allocator;
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close final {n::Context&context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_ASSIGN_OR_RETURN(auto data,r::inputs(context,device,staging));
  const std::array<std::int32_t,2> ptr{0,1};const std::array<std::int32_t,1> ids{1},tail{2};
  HYLE_ASSIGN_OR_RETURN(auto layout,data.schema.version({ptr,ptr,ids,tail}));data.layout=layout;
  HYLE_ASSIGN_OR_RETURN(auto cpu,g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto shape,jr::graph(data,true));
  c::ResourceBudget budget;
  if(mode==Mode::budget){
    const std::array limits{c::ResourceBudgetLimit{hyle::mem::system_resource(),64u<<20},
        c::ResourceBudgetLimit{device.ref(),SIZE_MAX},c::ResourceBudgetLimit{staging.ref(),SIZE_MAX}};
    HYLE_ASSIGN_OR_RETURN(budget,c::ResourceBudget::create({.limits=limits,.allocator=allocator.ref(),.allow_unattributed=true}));
  }
  HYLE_ASSIGN_OR_RETURN(auto env,jr::environment(cpu,context,device,staging));
  HYLE_ASSIGN_OR_RETURN(auto program,jr::compile(shape,env,allocator.ref(),budget));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),1);
  jr::Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto cpu_queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto query,jr::query(shape));
  HYLE_ASSIGN_OR_RETURN(auto lanes,jr::queues(program.report,cpu_queue));
  HYLE_ASSIGN_OR_RETURN(auto outputs,jr::output_pools(shape));
  HYLE_ASSIGN_OR_RETURN(auto receiver,jr::create(program,scheduler,completion_waiters,allocator.ref(),budget));
  std::vector<g::batch::NodeLayout> versions;
  for(const auto&node:shape.layouts)versions.push_back({node.node,layout});
  const auto prepare_sources=[&](std::size_t capacity){return g::batch::prepare(receiver,lanes,versions,
      {.allocator=allocator.ref(),.resource_budget=budget,.max_source_completions=capacity});};
  HYLE_RETURN_IF_ERROR(prepare_sources(0));
  const auto baseline=allocator.live.load();
  if(mode==Mode::budget){
    HYLE_ASSIGN_OR_RETURN(auto before,budget.usage(hyle::mem::system_resource()));
    HYLE_ASSIGN_OR_RETURN(auto full,budget.reserve(hyle::mem::system_resource(),before.limit_bytes-before.live_bytes));
    const auto refused=prepare_sources(source_capacity);
    HYLE_RETURN_IF_ERROR(r::require(refused.IsResourceExhausted() && allocator.live==baseline,
        "joint pending cold source bypassed shared metadata budget or leaked allocation"));
    full.reset();HYLE_ASSIGN_OR_RETURN(auto after,budget.usage(hyle::mem::system_resource()));
    HYLE_RETURN_IF_ERROR(r::require(after.live_bytes==before.live_bytes,"joint pending reserve rejection leaked budget charges"));
  }
  if(mode==Mode::allocation){
    bool complete=false;std::size_t count=0;
    for(std::size_t nth=0;nth<32;++nth){allocator.fail=allocator.attempts+nth;
      auto status=prepare_sources(source_capacity);allocator.fail=SIZE_MAX;
      if(status.ok()){complete=true;count=nth;break;}
      if(!status.IsResourceExhausted() || allocator.live!=baseline)
        std::fprintf(stderr,"pending reserve OOM nth=%zu code=%u live=%zu baseline=%zu attempts=%zu message=%.*s\n",
            nth,unsigned(status.code()),allocator.live.load(),baseline,allocator.attempts.load(),
            int(status.message().size()),status.message().data());
      HYLE_RETURN_IF_ERROR(r::require(status.IsResourceExhausted() && allocator.live==baseline,
          "joint pending reserve OOM leaked metadata or consumed original result slot"));}
    HYLE_RETURN_IF_ERROR(r::require(complete,"joint pending reserve has no finite allocation boundary"));
    std::fprintf(stderr,"%s joint pending all %zu reserve allocation refusals rollback PASS\n",r::route,count);
  }else HYLE_RETURN_IF_ERROR(prepare_sources(mode==Mode::capacity?1:source_capacity));
  HYLE_RETURN_IF_ERROR(receiver.bind(shape.query,shape.query_desc,query.read));
  HYLE_RETURN_IF_ERROR(jr::bind_outputs(receiver,shape,outputs));
  HYLE_ASSIGN_OR_RETURN(auto kw,data.keys.acquire(1));HYLE_ASSIGN_OR_RETURN(auto vw,data.values.acquire(1));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(staging.ref(),1024));
  HYLE_ASSIGN_OR_RETURN(auto host,pool.allocate(512));HYLE_RETURN_IF_ERROR(host.view()->fill(std::byte{0}));
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  std::array<std::uint16_t,256> source_words;source_words.fill(0xa800);
  HYLE_RETURN_IF_ERROR(host.view()->write(std::as_bytes(std::span(source_words))));
#endif
  HYLE_ASSIGN_OR_RETURN(auto ready,host.publish());
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  const std::array specs{c::CopyResourceSpec{device.ref(),512},c::CopyResourceSpec{device.ref(),data.keys.page_bytes()},
#else
  const std::array specs{c::CopyResourceSpec{staging.ref(),512},c::CopyResourceSpec{device.ref(),data.keys.page_bytes()},
#endif
      c::CopyResourceSpec{device.ref(),data.values.page_bytes()}};
  std::vector<c::CopyResourceSpec> copy_specs(specs.begin(),specs.end());
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  copy_specs.push_back({device.ref(),512});
  HYLE_ASSIGN_OR_RETURN(auto source_desc,c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,1>{256}));
  HYLE_ASSIGN_OR_RETURN(auto value_host,pool.allocate(512));source_words.fill(0xb400);
  HYLE_RETURN_IF_ERROR(value_host.view()->write(std::as_bytes(std::span(source_words))));
  HYLE_ASSIGN_OR_RETURN(auto value_ready,value_host.publish());
  HYLE_ASSIGN_OR_RETURN(auto producer_builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto producer_input,producer_builder.input(source_desc));
  HYLE_ASSIGN_OR_RETURN(auto positive,producer_builder.add(c::Unary{c::UnaryKind::kAbs},producer_input));
  HYLE_RETURN_IF_ERROR(producer_builder.publish(positive));
  HYLE_ASSIGN_OR_RETURN(auto producer_graph,std::move(producer_builder).finish());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto producer_program,g::apple::compile(context,producer_graph));
  HYLE_ASSIGN_OR_RETURN(auto key_input,device.allocate(source_desc));
  HYLE_ASSIGN_OR_RETURN(auto value_input,device.allocate(source_desc));
  HYLE_ASSIGN_OR_RETURN(auto key_tensor,ready.tensor(source_desc));
  HYLE_ASSIGN_OR_RETURN(auto value_tensor,value_ready.tensor(source_desc));
  HYLE_RETURN_IF_ERROR(device.copy_from_host(key_input,key_tensor));
  HYLE_RETURN_IF_ERROR(device.copy_from_host(value_input,value_tensor));
#else
  HYLE_ASSIGN_OR_RETURN(auto producer_program,g::vulkan::compile(context,producer_graph));
  source_words.fill(0xa800);
  HYLE_ASSIGN_OR_RETURN(auto key_input,context.upload(source_desc,std::as_bytes(std::span(source_words))));
  source_words.fill(0xb400);
  HYLE_ASSIGN_OR_RETURN(auto value_input,context.upload(source_desc,std::as_bytes(std::span(source_words))));
#endif
  HYLE_ASSIGN_OR_RETURN(auto key_producer,producer_program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto value_producer,producer_program.create_invocation());
  const std::array key_args{key_input},value_args{value_input};
  HYLE_RETURN_IF_ERROR(key_producer.bind(key_args));HYLE_RETURN_IF_ERROR(value_producer.bind(value_args));
#endif
  HYLE_ASSIGN_OR_RETURN(auto copy_program,n::CopyBatchProgram::prepare(context,copy_specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(copy_program));
#if !defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  HYLE_RETURN_IF_ERROR(copy.bind_source({0},ready));
#endif
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(kw.lease())));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({2},std::move(vw.lease())));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  const std::array<c::CopyRange,2> ranges{{{{0},{1},{1},0,0,mode==Mode::hole?128u:512u},{{
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
      3
#else
      0
#endif
      },{2},{1},0,0,512}}};
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  namespace mx=mlx::core;
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Release final {decltype(gate) gate;n::Context&context;~Release(){gate->release();context.close().IgnoreError();}}release{gate,context};
  const std::array blocker_specs{n::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto&inputs,mx::Stream stream){
    auto prefix=mx::square(inputs[0],stream);
    return std::vector{mx::array(prefix.shape(),prefix.dtype(),
      std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{prefix})};
  },blocker_specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());
  const std::array input{mx::array({1.F,2.F,3.F,4.F})};HYLE_RETURN_IF_ERROR(blocking.bind(input));
  {std::lock_guard lock(gate->mutex);gate->armed=true;}
  HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(),"joint pending risk has no actual native queued prefix"));
  const auto release_native=[&]{gate->release();};
#else
  hyle_f5_vk_failure_fixture::Boundary boundary(context);
  static std::atomic<bool> masked{true};masked=true;
  boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
    auto&hook=*hyle_f5_vk_failure_fixture::Boundary::active;
    if(hook.fail_poll.exchange(false))return VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto real=hook.original.GetFenceStatus(device,fence);
    return masked.load()?VK_NOT_READY:real;
  };
  const auto release_native=[&]{masked=false;};
  struct Release final {std::atomic<bool>& masked;~Release(){masked=false;}}release{masked};
#endif
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  HYLE_ASSIGN_OR_RETURN(auto key_result,key_producer.submit());
  HYLE_ASSIGN_OR_RETURN(auto value_result,value_producer.submit());
  const std::array<n::ResultCopyBinding,1> key_binding{{{{0},0,source_desc}}},value_binding{{{{3},0,source_desc}}};
  HYLE_RETURN_IF_ERROR(copy.bind_source(key_binding,std::move(key_result)));
  HYLE_RETURN_IF_ERROR(copy.bind_source(value_binding,std::move(value_result)));
  HYLE_RETURN_IF_ERROR(r::require(!key_result.valid() && !value_result.valid() && !key_producer.submit() && !value_producer.submit(),
      "independent native source owners were not both retained by original F2"));
#endif
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
  HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,n::CopyBatchUpdate::take(std::move(copy),std::move(submitted)));
  const std::array<c::PagedPageUpdate,1> key_update{{{&kw,{1}}}},value_update{{{&vw,{2}}}};
  HYLE_ASSIGN_OR_RETURN(auto k,data.keys.pending_read(layout,{},key_update));
  HYLE_ASSIGN_OR_RETURN(auto v,data.values.pending_read(layout,{},value_update));
  std::array<c::Completion*,source_capacity> original_sources{};
  HYLE_ASSIGN_OR_RETURN(auto count,n::detail::CopyStreamAccess::retirement_sources(update,original_sources));
  HYLE_RETURN_IF_ERROR(r::require(count==original_sources.size(),"joint pending risk lost original accepted producer/phase controls"));
  if(mode==Mode::cancel_before)HYLE_RETURN_IF_ERROR(update.request_cancel());

#if !defined(HYLE_G01_PAGE_RISK_APPLE)
  if(mode==Mode::native_rejection)boundary.fail_at=boundary.submits+1;
#endif
  if(mode==Mode::closed){release_native();HYLE_RETURN_IF_ERROR(update.wait());}
  if(mode==Mode::allocation){
    const auto before=allocator.live.load();allocator.fail=allocator.attempts.load();
    const auto refused=g::batch::bind(receiver,shape.key,k);allocator.fail=SIZE_MAX;
    HYLE_RETURN_IF_ERROR(r::require(refused.IsResourceExhausted() && allocator.live==before && update.valid() && k.valid(),
        "joint pending binding OOM changed source/permission or leaked metadata"));
  }
  if(mode==Mode::budget){
    HYLE_ASSIGN_OR_RETURN(auto before,budget.usage(hyle::mem::system_resource()));
    HYLE_ASSIGN_OR_RETURN(auto full,budget.reserve(hyle::mem::system_resource(),before.limit_bytes-before.live_bytes));
    const auto refused=g::batch::bind(receiver,shape.key,k);
    HYLE_RETURN_IF_ERROR(r::require(refused.IsResourceExhausted() && update.valid() && k.valid(),
        "joint pending binding bypassed shared metadata budget or consumed source"));
    full.reset();HYLE_ASSIGN_OR_RETURN(auto after,budget.usage(hyle::mem::system_resource()));
    HYLE_RETURN_IF_ERROR(r::require(after.live_bytes==before.live_bytes,"joint pending binding rejection leaked budget charges"));
  }
  HYLE_RETURN_IF_ERROR(r::require(!g::batch::bind(receiver,shape.query,k).ok(),"joint pending capacity accepted as ordinary query"));
  HYLE_RETURN_IF_ERROR(g::batch::bind(receiver,shape.key,k));HYLE_RETURN_IF_ERROR(g::batch::bind(receiver,shape.value,v));
  HYLE_RETURN_IF_ERROR(receiver.prepare_replay(lanes));
  // Ordinary replay preparation validates the live Source. Close only after
  // cold binding so this control still reaches actual typed submit admission.
  if(mode==Mode::closed)HYLE_RETURN_IF_ERROR(context.close());
  auto result=route::submit(receiver,lanes,std::move(update));
  if(mode==Mode::hole || mode==Mode::cancel_before || mode==Mode::capacity || mode==Mode::closed){
    HYLE_RETURN_IF_ERROR(r::require(!result && update.valid(),"receiver admission failure consumed caller Update"));
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::hole?result.error().IsOutOfRange():
        mode==Mode::cancel_before?result.error().IsCancelled():
        mode==Mode::closed?result.error().IsFailedPrecondition():result.error().IsResourceExhausted(),
        "receiver admission returned a different failure"));
    release_native();update.wait().IgnoreError();
    std::fprintf(stderr,"%s joint pending rejection mode=%u preserves original Update PASS\n",r::route,unsigned(mode));return Status::Ok();
  }
  HYLE_RETURN_IF_ERROR(r::require(result.has_value() && !update.valid() && !route::take_update(receiver),
      "accepted receiver lost its original source controls"));
  // Progress admission is asynchronous. Wait only for the real query guard,
  // while the original native source remains gated and cannot retire.
  const auto accepted_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
  while(!query.pool.stats().active_uses && std::chrono::steady_clock::now()<accepted_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  HYLE_RETURN_IF_ERROR(r::require(query.pool.stats().active_uses>0,
      "accepted receiver never admitted its actual query use"));
  // With independent producers, cancel only the second source; the first stays OK.
  if(mode==Mode::cancel_after)HYLE_RETURN_IF_ERROR(original_sources[cancel_source]->request_cancel());
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
  if(mode==Mode::unknown){
    boundary.fail_poll=true;
    auto source_diagnosis=original_sources.back()->poll();
    HYLE_RETURN_IF_ERROR(r::require(source_diagnosis && *source_diagnosis==c::CompletionPoll::kPending,
        "joint original source unknown did not retain actual copy prefix"));
    auto diagnosis=result->poll();
    std::fprintf(stderr,"joint unknown has_poll=%d poll=%d injected=%d query_uses=%llu source_uses=%llu\n",
        diagnosis.has_value(),diagnosis?int(*diagnosis):-1,!boundary.fail_poll.load(),
        static_cast<unsigned long long>(query.pool.stats().active_uses),static_cast<unsigned long long>(pool.stats().active_uses));
    // The existing dependent observer records the original error, but remains
    // pending until actual retirement. Its final wait must return that error.
    HYLE_RETURN_IF_ERROR(r::require(diagnosis && *diagnosis==c::CompletionPoll::kPending &&
        !boundary.fail_poll.load() && query.pool.stats().active_uses>0 && !route::take_update(receiver),
        "unknown receiver observation fabricated retirement or released source"));
  }
#endif
  if(mode==Mode::abandon){
    *result={};receiver={};
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
    HYLE_RETURN_IF_ERROR(r::require(!key_producer.submit() && !value_producer.submit(),
        "abandoned joint source released either producer slot before actual native retirement"));
#endif
    HYLE_RETURN_IF_ERROR(r::require((native_owner_sources || pool.stats().active_uses>0) && query.pool.stats().active_uses>0,
        "abandoned receiver dropped source/guards before actual suffix retirement"));
    release_native();HYLE_RETURN_IF_ERROR(context.close());scheduler.drain();
    HYLE_RETURN_IF_ERROR(r::require(pool.stats().active_uses==0 && query.pool.stats().active_uses==0,
        "abandoned original work did not retire all source/suffix guards"));
  }else{
    release_native();const auto outcome=result->wait();
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::cancel_after?outcome.IsCancelled():
        (mode==Mode::unknown || mode==Mode::native_rejection)?outcome.IsResourceExhausted():outcome.ok(),
        "receiver original completion did not merge the expected source/suffix terminal"));
    const auto observed=result->wait();
    HYLE_RETURN_IF_ERROR(r::require(observed.code()==outcome.code(),
        "repeated original completion observation changed its cached terminal"));
    std::vector<g::GraphOutput> held_outputs;
    if(outcome.ok()){
      for(const auto output:shape.outputs){
        HYLE_ASSIGN_OR_RETURN(auto owner,receiver.result(output));
        HYLE_ASSIGN_OR_RETURN(auto tensor,owner.tensor());
        HYLE_ASSIGN_OR_RETURN(auto bytes,tensor.description().required_bytes());
        std::vector<float> actual(bytes/sizeof(float));
        HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
        for(auto value:actual)HYLE_RETURN_IF_ERROR(r::require(native_owner_sources?value==-.25F:value==0.F,
            "actual pending sources and Graph CPU suffix differ from independent constant-value oracle"));
        held_outputs.push_back(std::move(owner));
      }
    }
    // A terminal Completion observes work; only successful GraphOutput owners
    // retain result reuse. Failed/cancelled work cannot publish such an owner.
    HYLE_RETURN_IF_ERROR(r::require(query.pool.stats().active_uses==0,
        "completed pending Graph retained query guards"));
    if(outcome.ok())HYLE_RETURN_IF_ERROR(r::require(!route::take_update(receiver),
        "retained GraphOutput allowed original source reuse"));
    else for(auto output:shape.outputs)HYLE_RETURN_IF_ERROR(r::require(!receiver.result(output),
        "failed/cancelled pending Graph published an output"));
    held_outputs.clear();*result={};
    const auto unrecovered=prepare_sources(source_capacity);
    if(native_owner_sources && mode==Mode::cancel_after)std::fprintf(stderr,"independent cancelled source reuse status=%u: %.*s\n",
        unsigned(unrecovered.code()),int(unrecovered.message().size()),unrecovered.message().data());
    HYLE_RETURN_IF_ERROR(r::require(mode==Mode::success || mode==Mode::allocation || mode==Mode::budget?
        unrecovered.IsUnavailable():native_owner_sources && mode==Mode::cancel_after?
        (unrecovered.IsUnavailable() || unrecovered.IsFailedPrecondition()):unrecovered.IsFailedPrecondition(),
        "unrecovered/failed original source permitted joint slot reuse"));
    HYLE_ASSIGN_OR_RETURN(update,route::take_update(receiver));
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
    if(mode==Mode::cancel_after)HYLE_RETURN_IF_ERROR(r::require(prepare_sources(source_capacity).IsFailedPrecondition() &&
        !update.take_mutable({1}) && !update.take_mutable({2}),
        "source-only failure was washed away after control recovery or published successful page writers"));
#endif
    if(mode==Mode::success || mode==Mode::allocation || mode==Mode::budget){
      HYLE_ASSIGN_OR_RETURN(kw.lease(),update.take_mutable({1}));HYLE_ASSIGN_OR_RETURN(vw.lease(),update.take_mutable({2}));
      HYLE_ASSIGN_OR_RETURN(auto key,kw.publish(2));HYLE_ASSIGN_OR_RETURN(auto value,vw.publish(2));
      HYLE_RETURN_IF_ERROR(r::require(data.keys.read(layout,std::span(&key,1)).has_value() &&
          data.values.read(layout,std::span(&value,1)).has_value(),"receiver could not recover exact initialized ready frames"));
    }
  }
  std::fprintf(stderr,"%s actual original joint pending Graph mode=%u final result/completion, source controls and guards PASS\n",r::route,unsigned(mode));
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  std::fprintf(stderr,"two independent accepted native producer owners -> same F2 Update -> two real joint page nodes, second-source error/abandon mode=%u PASS\n",unsigned(mode));
#endif
  return Status::Ok();
}
}
int main(){
#if defined(HYLE_PENDING_MULTI_NATIVE_PRODUCERS)
  for(auto mode:{Mode::success,Mode::cancel_after,Mode::abandon}){
#else
  for(auto mode:{Mode::success,Mode::hole,Mode::cancel_before,Mode::cancel_after,Mode::abandon,Mode::allocation,Mode::capacity,Mode::closed,Mode::budget
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
      ,Mode::unknown,Mode::native_rejection
#endif
  }){
#endif
    auto status=episode(mode);if(!status.ok()){
    std::fprintf(stderr,"joint pending risk mode=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());return 1;
  }}return 0;
}
