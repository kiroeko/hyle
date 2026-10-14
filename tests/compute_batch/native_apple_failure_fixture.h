#pragma once
#include <hyle/compute/apple_batch.h>
#include <hyle/compute/apple_memory.h>
#include <hyle/graph/apple.h>
#include <hyle/mem/quota_allocator.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <stdexcept>

namespace hyle_f5_apple_failure_fixture {
namespace c=hyle::compute;namespace a=c::apple;namespace mx=mlx::core;
using hyle::Status;
inline Status check(bool value,const char*message){return value?Status::Ok():Status::Internal(message);}
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool armed=false,entered=false,released=false,fail=false;
  void wait(){std::unique_lock lock(mutex);if(!armed)return;entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});if(fail)throw std::runtime_error("F5 original MLX accepted prefix failure");}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
class Checkpoint final:public mx::Primitive {
public:
  Checkpoint(mx::Stream stream,std::shared_ptr<Gate>gate):Primitive(stream),gate_(std::move(gate)){}
  void eval_cpu(const std::vector<mx::array>&,std::vector<mx::array>&)override{throw std::runtime_error("F5 checkpoint requires actual GPU");}
  void eval_gpu(const std::vector<mx::array>&inputs,std::vector<mx::array>&outputs)override{gate_->wait();outputs[0].copy_shared_buffer(inputs[0]);}
  const char*name()const override{return "HyleF5NativeCheckpoint";}
private:std::shared_ptr<Gate>gate_;
};
inline Status episode(unsigned mode,bool graph){
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  {
    HYLE_ASSIGN_OR_RETURN(auto context,a::Context::create());HYLE_ASSIGN_OR_RETURN(auto memory,a::MemoryResource::create(context));HYLE_ASSIGN_OR_RETURN(auto staging,a::MemoryResource::create(context));
    auto gate=std::make_shared<Gate>();struct Release{std::shared_ptr<Gate>gate;a::Context&context;~Release(){gate->release();context.close().IgnoreError();}}release{gate,context};
    c::BatchCapacity cap{.storage=c::BatchStorage::kRagged,.max_requests=1,.max_query_rows=2,.max_kv_rows=2,.max_pages=0,.max_page_refs=0,.query_heads=2,.kv_heads=2,.qk_head_dim=64,.value_head_dim=64,.page_size=1};
    HYLE_ASSIGN_OR_RETURN(auto schema,c::BatchSchema::create(cap));const std::array<std::int32_t,2> qptr{0,1},kvptr{0,2};HYLE_ASSIGN_OR_RETURN(auto layout,schema.version({qptr,kvptr,{},{}}));
    HYLE_ASSIGN_OR_RETURN(auto recipe,c::MaterializedBatchProgram::prepare(schema,{.max_kv_per_request=2,.max_host_metadata_bytes=1u<<20,.max_logical_intermediate_bytes=1u<<20,.scale=.125f}));
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(memory.ref(),4096));std::array<c::ReadLease,3> ready;
    for(auto&read:ready){HYLE_ASSIGN_OR_RETURN(auto writer,pool.allocate(512));HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{0}));HYLE_ASSIGN_OR_RETURN(read,writer.publish());}
    const c::MaterializedNativeBatchOptions options{.max_metadata_backing_bytes=1u<<20,.allocator=metadata.get()};
    auto made=graph?hyle::graph::apple::prepare_materialized_batch(context,recipe,memory,staging,options):a::MaterializedBatchInvocation::create(context,recipe,memory,staging,options);if(!made)return made.error();auto controller=std::move(*made);
    if(mode){HYLE_RETURN_IF_ERROR(controller.prepare_batch(layout));HYLE_RETURN_IF_ERROR(controller.bind(ready[0],ready[1],ready[2]));}
    const std::array specs{a::InputSpec{{4},mx::float32}};
    HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto&inputs,mx::Stream stream){auto prefix=mx::square(inputs[0],stream);return std::vector{mx::array(prefix.shape(),prefix.dtype(),std::make_shared<Checkpoint>(stream,gate),{prefix})};},specs));
    HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());const std::array values{mx::array({1.f,2.f,3.f,4.f})};HYLE_RETURN_IF_ERROR(blocking.bind(values));
    {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=mode==0 || mode==3;}
    HYLE_ASSIGN_OR_RETURN(auto prefix,blocking.submit());HYLE_RETURN_IF_ERROR(check(gate->await(),"Apple native checkpoint did not enter GPU boundary"));
    if(!mode){
      std::promise<void> started;auto entered=started.get_future();auto prepared=std::async(std::launch::async,[&]{started.set_value();return controller.prepare_batch(layout);});entered.get();
      const bool waited=prepared.wait_for(std::chrono::milliseconds(80))==std::future_status::timeout;gate->release();const auto status=prepared.get();
      HYLE_RETURN_IF_ERROR(check(waited && status.IsInternal() && !controller.submit() && !controller.prepare_batch(layout).ok(),"Apple metadata queued failure published/reused mirror"));controller.drain().IgnoreError();
      std::printf("Apple %s queued metadata prepare waited=%d failed-prefix/no-publication PASS\n",graph?"Graph":"Direct",waited);
    }else{
      HYLE_ASSIGN_OR_RETURN(auto queued,controller.submit());HYLE_RETURN_IF_ERROR(check(queued.poll() && pool.stats().active_uses==3 && !queued.result(),"Apple queued F5 native guards/result differ"));
      if(mode==1)HYLE_RETURN_IF_ERROR(queued.request_cancel());
      if(mode==2){queued={};controller={};HYLE_RETURN_IF_ERROR(check(pool.stats().active_uses==3,"Apple abandoned F5 work released guards before native prefix"));gate->release();HYLE_RETURN_IF_ERROR(context.close());}
      else{
        gate->release();auto result=queued.wait();HYLE_RETURN_IF_ERROR(check(!result && (mode==1?result.error().IsCancelled():result.error().IsInternal()),"Apple queued cancel/native failure lost first terminal"));queued={};
        HYLE_RETURN_IF_ERROR(check(!controller.prepare_batch(layout).ok() && !controller.submit(),"Apple open failed native slot reused after actual retirement"));controller.drain().IgnoreError();
      }
      HYLE_RETURN_IF_ERROR(check(pool.stats().active_uses==0,"Apple F5 work did not actually retire input guards"));
      std::printf("Apple %s queued math mode=%u guards-before=3 guards-after=0 cancel/abandon/failure/no-reuse PASS\n",graph?"Graph":"Direct",mode);
    }
    controller={};prefix={};context.close().IgnoreError();
  }return check(metadata.used()==0,"Apple F5 queued lifecycle metadata leaked");
}
inline Status run(){for(bool graph:{false,true})for(unsigned mode=0;mode<4;++mode)HYLE_RETURN_IF_ERROR(episode(mode,graph));return Status::Ok();}
} // namespace hyle_f5_apple_failure_fixture
