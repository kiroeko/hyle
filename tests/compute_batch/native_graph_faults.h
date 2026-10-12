#pragma once
#include "../../src/compute/context_state.h"
#include "../../src/compute/completion_internal.h"
#include "../../src/compute/provider_internal.h"
#include <hyle/compute/detail/batch_provider.h>
#include <hyle/compute/detail/completion_access.h>
#include <hyle/core/assert.h>
#include <atomic>
#include <future>
#include <utility>

namespace hyle_f5_graph_fault_fixture {
namespace cd=c::detail;
struct RunDecorator final {
  c::Provider original;const cd::ProviderVTable*base;cd::ProviderVTable table;
  const cd::BatchProviderVTable*batch_base=nullptr;cd::BatchProviderVTable batch{};
  unsigned count=0,fail_at=0;static inline RunDecorator*active=nullptr;
  explicit RunDecorator(c::Provider provider):original(std::move(provider)),base(cd::ProviderAccess::vtable(original)),table(*base){
    table.query_extension=[](void*context,c::ProviderExtensionId id)noexcept->const void*{
      auto&s=*active;const auto*result=s.base->query_extension(context,id);
      if(!result || id!=cd::kBatchPlanningExtensionId)return result;
      s.batch_base=static_cast<const cd::BatchProviderVTable*>(result);s.batch=*s.batch_base;
      s.batch.run=[](void*native,c::Queue&queue,std::span<const c::ConstTensorView>inputs,cd::BatchWorkspaceViews views)noexcept{
        auto&s=*active;auto result=s.batch_base->run(native,queue,inputs,views);++s.count;
        if(s.fail_at && s.count==s.fail_at){HYLE_FATAL_ASSERT(result.accepted);std::printf("F5 Graph actual region run=%u accepted; inject error before suffix\n",s.count);result.status.Update(Status::Unavailable("F5 Graph injected accepted region error"));}
        return result;
      };return &s.batch;
    };active=this;
  }
  c::Provider provider()const noexcept{return cd::ProviderAccess::with_vtable(original,&table);}
  ~RunDecorator(){active=nullptr;}
};
struct Gate final {
  cudaStream_t stream=nullptr;std::atomic<bool> released{false};
  static void CUDART_CB wait(void*p){auto&s=*static_cast<Gate*>(p);s.released.wait(false,std::memory_order_acquire);}
  void release(){released.store(true,std::memory_order_release);released.notify_all();}
  ~Gate(){release();if(stream)cudaStreamSynchronize(stream);}
};
struct Unknown final {c::Completion native;cd::CompletionRetention retention{};cd::CompletionLifetime forwarded{};};
inline const cd::CompletionVTable unknown{
  .destroy=[](void*p,cd::CompletionLifetime lifetime,Status hint)noexcept{
    auto*s=static_cast<Unknown*>(p);s->forwarded=lifetime;
    s->retention.own={s,[](void*p,const Status&terminal)noexcept{const auto&f=static_cast<Unknown*>(p)->forwarded;if(f.valid())f.terminal(f.state,terminal);},[](void*p)noexcept{auto*s=static_cast<Unknown*>(p);const auto f=s->forwarded;if(f.valid())f.release(f.state);delete s;}};
    auto native=std::move(s->native);cd::CompletionAccess::append_terminal(native,std::move(hint));HYLE_FATAL_ASSERT(cd::CompletionAccess::attach_retention(native,s->retention).ok());
  },
  .poll=[](void*)noexcept->Result<c::CompletionPoll>{return std::unexpected(Status::Unavailable("F5 Graph unknown accepted prefix"));},
  .wait=[](void*)noexcept{return Status::Unavailable("F5 Graph unknown accepted prefix");}
};
struct Decorator final {
  cd::ContextState*context;const cd::BackendVTable*original;cd::BackendVTable table;
  unsigned target,count=0;Gate*gate=nullptr;static inline Decorator*active=nullptr;
  Decorator(c::Context&owner,unsigned position,Gate*blocked=nullptr):context(cd::ContextAccess::state(owner)),original(context->backend_vtable),table(*original),target(position),gate(blocked){
    table.finish_sequence=[](void*q,void*sequence,bool accepted,Status terminal)noexcept{
      auto&s=*active;const bool selected=++s.count==s.target;
      if(selected){HYLE_FATAL_ASSERT(accepted);std::printf("F5 Graph fault accepted sequence=%u gate=%d\n",s.count,s.gate!=nullptr);
        if(s.gate)HYLE_FATAL_ASSERT(cudaLaunchHostFunc(s.gate->stream,&Gate::wait,s.gate)==cudaSuccess);
        else terminal.Update(Status::Unavailable("F5 Graph injected accepted prefix error"));
      }
      auto completion=s.original->finish_sequence(q,sequence,accepted,std::move(terminal));
      if(!selected || !s.gate)return completion;
      return cd::CompletionAccess::make(&unknown,new Unknown{std::move(completion)});
    };active=this;context->backend_vtable=&table;
  }
  ~Decorator(){context->backend_vtable=original;active=nullptr;}
};
} // namespace hyle_f5_graph_fault_fixture
