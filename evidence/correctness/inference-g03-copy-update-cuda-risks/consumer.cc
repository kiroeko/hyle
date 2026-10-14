#define HYLE_TESTING 1
#include "../../src/compute_backend_cuda/cuda_internal.h"
#include <hyle/compute/batch_transfer.h>
#include <hyle/compute/detail/copy_stream_access.h>
#include <hyle/mem/quota_allocator.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

namespace {
namespace c = hyle::compute;
namespace n = c::cuda;
namespace d = n::detail;
using hyle::Status;
Status require(bool value, const char *message) { return value ? Status::Ok() : Status::Internal(message); }
struct Gate final {
  std::mutex mutex; std::condition_variable condition; bool entered = false, released = false;
  void wait() { std::unique_lock lock(mutex); entered = true; condition.notify_all(); condition.wait(lock, [&] { return released; }); }
  bool await() { std::unique_lock lock(mutex); return condition.wait_for(lock, std::chrono::seconds(3), [&] { return entered; }); }
  void release() { std::lock_guard lock(mutex); released = true; condition.notify_all(); }
};
thread_local bool fail_query = false;
cudaError_t query(cudaEvent_t event) noexcept {
  const auto result = cudaEventQuery(event);
  if (std::exchange(fail_query, false)) return result == cudaErrorNotReady ? cudaErrorMemoryAllocation : result;
  return result;
}
Status episode(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto device_id, n::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(device_id));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  const std::array specs{c::CopyResourceSpec{device.ref(),1024}, c::CopyResourceSpec{device.ref(),1024}};
  HYLE_ASSIGN_OR_RETURN(auto program, c::CopyBatchProgram::prepare(context, specs, {.allocator=metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto copy, c::CopyBatchInvocation::create(program, queue));
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(),4096));
  HYLE_ASSIGN_OR_RETURN(auto source, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto pointer, n::pointer<std::byte>(source.view().value()));
  HYLE_ASSIGN_OR_RETURN(auto uses, c::LeaseUsePool::create());
  HYLE_ASSIGN_OR_RETURN(auto writing, uses.begin());
  HYLE_RETURN_IF_ERROR(writing.write(source));
  Gate gate;
  struct Release final { Gate &gate; c::Context &context; ~Release() { gate.release(); context.drain().IgnoreError(); } } release{gate,context};
  HYLE_ASSIGN_OR_RETURN(auto producer, n::submit(queue, [&](cudaStream_t stream) noexcept -> Status {
    HYLE_RETURN_IF_ERROR(n::status(cudaLaunchHostFunc(stream, [](void *p) { static_cast<Gate *>(p)->wait(); }, &gate)));
    return n::status(cudaMemsetAsync(pointer,49,1024,stream));
  }));
  HYLE_RETURN_IF_ERROR(writing.attach(producer));
  HYLE_RETURN_IF_ERROR(require(gate.await(), "actual CUDA producer did not reach checkpoint"));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({0},std::move(source)));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(destination)));
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0},std::move(producer)));
  const std::array<c::CopyPhase,2> phases{{{}, {}}};
  const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,1024}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
  HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto owner,c::CopyBatchUpdate::take(std::move(copy),std::move(submitted)));
  if (mode == 1) HYLE_RETURN_IF_ERROR(require(owner.request_cancel().code()==hyle::StatusCode::Unimplemented,
      "CUDA update changed original unsupported native cancellation"));
  if (mode == 2 || mode == 3) {
    const auto previous = d::set_completion_observation_hooks({&query,nullptr});
    fail_query = true;
    const auto observed = owner.observe();
    d::set_completion_observation_hooks(previous);
    HYLE_RETURN_IF_ERROR(require(observed.IsUnavailable() && !fail_query && owner.in_flight() &&
        pool.stats().active_uses>0 && !owner.take_mutable({1}),
        "unknown CUDA event observation fabricated native retirement/ready writer"));
  }
  if (mode == 3) {
    owner = {};
    HYLE_RETURN_IF_ERROR(require(metadata.used()>0 && pool.stats().active_uses>0,
        "abandoned update released accepted native metadata/guards"));
    gate.release(); HYLE_RETURN_IF_ERROR(context.drain()); program.reset();
    const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while (metadata.used()!=0 && std::chrono::steady_clock::now()<deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HYLE_RETURN_IF_ERROR(require(metadata.used()==0 && pool.stats().active_uses==0,
        "original CUDA reaper did not retire abandoned update controls"));
  } else {
    gate.release(); const auto terminal = owner.wait();
    HYLE_RETURN_IF_ERROR(require(mode==2 ? terminal.IsResourceExhausted() : terminal.ok(),
        "CUDA update lost original success/observation error"));
    if (mode==0) {
      HYLE_ASSIGN_OR_RETURN(auto output,owner.take_mutable({1}));
      HYLE_ASSIGN_OR_RETURN(auto ready,output.publish());
      HYLE_ASSIGN_OR_RETURN(auto output_pointer,n::pointer<std::byte>(ready.view().value()));
      std::array<std::byte,1024> bytes;
      HYLE_RETURN_IF_ERROR(n::status(cudaMemcpy(bytes.data(),output_pointer,bytes.size(),cudaMemcpyDeviceToHost)));
      for (auto byte:bytes) HYLE_RETURN_IF_ERROR(require(byte==std::byte{49},"actual CUDA update byte oracle differs"));
    } else {
      HYLE_RETURN_IF_ERROR(require(!owner.take_mutable({1}), "closed/unknown-failed update returned ready writer"));
      if (mode==2) HYLE_RETURN_IF_ERROR(require(!c::detail::CopyStreamAccess::input(owner,{1}) &&
          !c::detail::CopyStreamAccess::dependency(owner), "known failed update admitted pending input/dependency"));
    }
    HYLE_ASSIGN_OR_RETURN(copy,owner.release_invocation());
    copy.drain().IgnoreError();
  }
  std::fprintf(stderr,"actual CUDA CopyBatchUpdate mode=%u PASS:1024bytes real gate/memset/copy, "
      "success/unsupported-cancel/unknown/abandon original terminal and native retirement\n",mode);
  return Status::Ok();
}
Status run() { for(unsigned mode=0;mode<4;++mode) HYLE_RETURN_IF_ERROR(episode(mode)); return Status::Ok(); }
} // namespace
int main() { const auto status=run(); if(!status.ok()) std::fprintf(stderr,"CUDA update risks:%.*s\n",int(status.message().size()),status.message().data()); return status.ok()?0:1; }
