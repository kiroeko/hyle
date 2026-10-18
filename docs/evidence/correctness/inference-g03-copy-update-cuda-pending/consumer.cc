#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/detail/copy_stream_access.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {
namespace c=hyle::compute;namespace n=c::cuda;
using hyle::Status;
struct Gate final {
  std::mutex mutex;std::condition_variable condition;bool entered=false,released=false;
  void wait(){std::unique_lock lock(mutex);entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}
  bool await(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(3),[&]{return entered;});}
  void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto device_id,n::device(0));HYLE_ASSIGN_OR_RETURN(auto context,c::Context::create(device_id));
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue,context.create_queue());
  struct Drain{c::Queue& queue;~Drain(){queue.drain().IgnoreError();}}drain{queue};
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto source,pool.allocate(1024));HYLE_ASSIGN_OR_RETURN(auto destination,pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto pointer,n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto uses,c::LeaseUsePool::create());HYLE_ASSIGN_OR_RETURN(auto writing,uses.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source));
  const std::array specs{c::CopyResourceSpec{device.ref(),1024},c::CopyResourceSpec{device.ref(),1024}};
  HYLE_ASSIGN_OR_RETURN(auto program,c::CopyBatchProgram::prepare(context,specs));
  HYLE_ASSIGN_OR_RETURN(auto copy,c::CopyBatchInvocation::create(program,queue));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(destination)));
  std::fprintf(stderr,"producer probe: source/use/copy prepared\n");
  Gate gate;struct Release{Gate& gate;c::Queue& queue;~Release(){gate.release();queue.drain().IgnoreError();}}release{gate,queue};
  HYLE_ASSIGN_OR_RETURN(auto producer,n::submit(queue,[&](cudaStream_t stream)noexcept->Status{
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream,[](void* p){static_cast<Gate*>(p)->wait();},&gate)));
    return n::status(cudaMemsetAsync(pointer,49,1024,stream));
  }));
  std::fprintf(stderr,"producer probe: native prefix accepted\n");
  const auto attached=writing.attach(producer);
  if(!attached.ok()){gate.release();producer.wait().IgnoreError();return attached;}
  std::fprintf(stderr,"producer probe: original F1 guard attached uses=%zu\n",std::size_t(pool.stats().active_uses));
  if(!gate.await()){gate.release();producer.wait().IgnoreError();return Status::Internal("producer probe did not enter real CUDA checkpoint");}
  const auto bound=copy.bind_mutable({0},std::move(source));
  std::fprintf(stderr,"producer probe: pending binding ok=%d writer=%d uses=%zu status=%.*s\n",bound.ok(),source.valid(),std::size_t(pool.stats().active_uses),int(bound.message().size()),bound.message().data());
  if(!bound.ok()){gate.release();producer.wait().IgnoreError();return bound;}
  if(source.valid() || pool.stats().active_uses!=1){gate.release();producer.wait().IgnoreError();return Status::Internal("pending source binding lost its native guard/permission");}
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
  const std::array<c::CopyPhase,2> phases{{{}, {}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,1024}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
  HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update,c::CopyBatchUpdate::take(std::move(copy),std::move(submitted)));
  HYLE_ASSIGN_OR_RETURN(auto pending_input,c::detail::CopyStreamAccess::input(update,{1}));
  HYLE_ASSIGN_OR_RETURN(auto dependency,c::detail::CopyStreamAccess::dependency(update));
  if(!dependency.pending() || pending_input.view.size()!=1024 || pending_input.last_write.value!=0 ||
      c::detail::CopyStreamAccess::input(update,{0}) || c::detail::CopyStreamAccess::input(update,{2}) ||
      pending_input.writer->view() || !update.observe().IsUnavailable() || update.take_mutable({1}))
    return Status::Internal("pending update erased prefix, released writer or accepted unwritten source");
  gate.release();HYLE_RETURN_IF_ERROR(update.wait());
  HYLE_ASSIGN_OR_RETURN(auto output,update.take_mutable({1}));
  HYLE_ASSIGN_OR_RETURN(copy,update.release_invocation());
  HYLE_ASSIGN_OR_RETURN(auto ready,output.publish());
  HYLE_ASSIGN_OR_RETURN(auto result_pointer,n::pointer<std::byte>(ready.view().value()));
  std::array<std::byte,1024> actual{};
  HYLE_RETURN_IF_ERROR(n::status(cudaMemcpy(actual.data(),result_pointer,actual.size(),cudaMemcpyDeviceToHost)));
  for(auto value:actual)if(value!=std::byte{49})return Status::Internal("actual producer/copy bytes differ");
  // The copy control still owns its producer token and mutable source. Drop
  // those successful retired bindings before asserting all F1 keepers gone.
  HYLE_RETURN_IF_ERROR(copy.reset());
  if(pool.stats().active_uses!=0)return Status::Internal("producer/copy native guards did not retire");
  std::fprintf(stderr,"actual pending CUDA producer -> original F2 CopyBatchUpdate PASS: typed input, source-only/invalidID rejection, final inline phase keeps real prefix, no intermediate producer host wait; explicit final update wait, not attention qualification\n");
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"producer probe: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
