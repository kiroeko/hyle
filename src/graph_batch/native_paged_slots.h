#pragma once

#include "hyle/compute/detail/materialized_native_controller.h"
#include "hyle/graph/batch.h"
#include "hyle/mem/std_allocator.h"
#include <algorithm>
#include <memory>
#include <new>
#include <vector>

namespace hyle::graph::batch::detail {
// Owns only the opt-in array of original page controllers. Queue, metadata
// banks, math, page admission and native retirement stay in their old owners.
template<class Traits> struct NativePagedSlots final {
  using Controller=compute::detail::MaterializedNativeState<Traits,compute::detail::PagedMetadataPolicy>;
  struct Slot final {
    Slot(mem::AllocatorRef a,typename Traits::Context context,const NativePagedNodeBinding& b,
         std::size_t f,std::span<const std::size_t> ports)
        : preparation(std::move(context)),controller(a),binding(&b),fragment(f),inputs(ports) {}
    typename Traits::Context preparation;
    Controller controller;
    const NativePagedNodeBinding* binding;
    std::size_t fragment;
    std::span<const std::size_t> inputs;
    std::array<compute::PagedArenaRead,2> frames;
    bool output_bridges_ready=false;
  };
  struct Delete final {
    mem::AllocatorRef allocator;
    void operator()(Slot* slot)const noexcept{allocator.destroy(slot);}
  };
  using Owner=std::unique_ptr<Slot,Delete>;
  struct Pending final {
    explicit Pending(mem::AllocatorRef a):allocator(a),frames(mem::StdAllocator<std::array<compute::PagedArenaPendingRead,2>>(a)){}
    mem::AllocatorRef allocator;
    compute::detail::CopyOutcomeSource<typename Traits::CopyUpdate,typename Traits::CopyAccess> source;
    compute::detail::DependentCompletion dependent;
    compute::detail::OutcomeSources sources{};
    compute::detail::CompletionRetention retention{};
    std::vector<std::array<compute::PagedArenaPendingRead,2>,mem::StdAllocator<std::array<compute::PagedArenaPendingRead,2>>> frames;
    compute::detail::ResultStorage*outcome=nullptr;
    Status terminal;
    void*execution_owner=nullptr;
    compute::detail::CompletionLifetime::ReleaseFn release_owner=nullptr;
    bool submitted=false;
  };
  struct PendingDelete final {void operator()(Pending*p)const noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}};
  explicit NativePagedSlots(mem::AllocatorRef a):allocator(a),slots(mem::StdAllocator<Owner>(a)) {}
  mem::AllocatorRef allocator;
  std::vector<Owner,mem::StdAllocator<Owner>> slots;
  std::unique_ptr<Pending,PendingDelete> pending;
  bool prepared=false;
  Slot* fragment(std::size_t index) noexcept {
    const auto found=std::find_if(slots.begin(),slots.end(),[&](const auto& s){return s->fragment==index;});
    return found==slots.end()?nullptr:found->get();
  }
  bool page_input(std::size_t id)const noexcept {
    return std::any_of(slots.begin(),slots.end(),[&](const auto& s){return s->inputs[1]==id || s->inputs[2]==id;});
  }
  Status add(typename Traits::Context context,const NativePagedNodeBinding& binding,
      std::size_t fragment,std::span<const std::size_t> inputs,
      const typename Traits::Memory& device,const typename Traits::Memory& staging,
      const compute::WorkloadContext& workload) noexcept {
    auto selected=binding.options.allocator;
    if(binding.options.resource_budget){HYLE_ASSIGN_OR_RETURN(selected,binding.options.resource_budget.metadata_allocator());}
    auto* pointer=allocator.template construct<Slot>(selected,std::move(context),binding,fragment,inputs);
    if(!pointer)return Status::ResourceExhausted("native page fragment slot metadata");
    Owner owner(pointer,Delete{allocator});
    HYLE_RETURN_IF_ERROR(pointer->controller.initialize(pointer->preparation,{},binding.program,
        device,staging,binding.options,workload));
    try{slots.push_back(std::move(owner));return Status::Ok();}
    catch(const std::bad_alloc&){return Status::ResourceExhausted("native page fragment index metadata");}
  }
  Status validate_layouts(std::span<const NodeLayout> layouts)const noexcept {
    if(layouts.size()!=slots.size())return Status::InvalidArgument("native page preparation needs complete node set");
    for(std::size_t i=0;i<layouts.size();++i){
      const auto& layout=layouts[i];
      const auto found=std::find_if(slots.begin(),slots.end(),[&](const auto& s){return s->binding->node==layout.node;});
      if(found==slots.end() || !(*found)->binding->program.schema().owns(layout.layout) ||
         layout.layout.facts().query_rows!=(*found)->binding->layout.facts().query_rows)
        return Status::InvalidArgument("native page node/layout/active output shape differs");
      for(std::size_t j=0;j<i;++j)if(layouts[j].node==layout.node)
        return Status::InvalidArgument("duplicate native page preparation node");
      for(std::size_t j=0;j<i;++j){
        const auto other=std::find_if(slots.begin(),slots.end(),[&](const auto& s){return s->binding->node==layouts[j].node;});
        for(std::size_t p=1;p<3;++p)for(std::size_t q=1;q<3;++q)
          if((*found)->inputs[p]==(*other)->inputs[q] && layout.layout!=layouts[j].layout)
            return Status::InvalidArgument("shared page input needs identical owned layout");
      }
    }
    return Status::Ok();
  }
  Status prepare(std::span<const NodeLayout> layouts) noexcept {
    HYLE_RETURN_IF_ERROR(reusable());
    prepared=false;HYLE_RETURN_IF_ERROR(validate_layouts(layouts));
    for(auto& slot:slots){
      slot->frames={};
      const auto found=std::find_if(layouts.begin(),layouts.end(),[&](const auto& l){return l.node==slot->binding->node;});
      HYLE_RETURN_IF_ERROR(slot->controller.prepare(found->layout));
      if(!slot->controller.report.reused_native_plan)slot->output_bridges_ready=false;
    }
    prepared=true;return Status::Ok();
  }
  template<class Frame>Status validate_frame(std::size_t input,const Frame& frame)const noexcept {
    if(!prepared || !page_input(input))return Status::FailedPrecondition("page frame needs explicit prepared node");
    for(const auto& slot:slots){
      if(slot->inputs[1]!=input && slot->inputs[2]!=input)continue;
      const auto layout=slot->controller.version.layout();
      const bool empty=layout.facts().query_rows==0 && layout.facts().page_refs==0;
      if(empty){if(frame.valid())return Status::InvalidArgument("empty page episode needs no KV frame");}
      else if(!frame.valid() || frame.layout()!=layout || !slot->binding->program.schema().owns(frame.layout()))
        return Status::InvalidArgument("native page frame schema/generation differs");
    }
    return Status::Ok();
  }
  void commit_frame(std::size_t input,compute::PagedArenaRead frame) noexcept {
    for(auto& slot:slots)for(std::size_t p=1;p<3;++p)
      if(slot->inputs[p]==input)slot->frames[p-1]=frame;
  }
  Status reserve_pending(std::size_t capacity,compute::ResourceBudget budget)noexcept{
    if(!capacity || pending)return Status::InvalidArgument("native Graph pending source capacity must be reserved once");
    auto selected=allocator;if(budget.valid()){HYLE_ASSIGN_OR_RETURN(selected,budget.metadata_allocator());}
    auto*p=selected.template construct<Pending>(selected);
    if(!p)return Status::ResourceExhausted("native Graph pending metadata");
    std::unique_ptr<Pending,PendingDelete> owner(p);
    HYLE_ASSIGN_OR_RETURN(p->source,(compute::detail::CopyOutcomeSource<typename Traits::CopyUpdate,typename Traits::CopyAccess>::create(
        capacity,{.allocator=selected,.resource_budget=budget})));
    HYLE_ASSIGN_OR_RETURN(p->dependent,compute::detail::DependentCompletion::create(selected));
    try{p->frames.resize(slots.size());}catch(...){return Status::ResourceExhausted("native Graph pending frame capacity");}
    pending=std::move(owner);return Status::Ok();
  }
  Status reusable()noexcept{
    if(!pending)return Status::Ok();
    if(!pending->source.available() || !pending->dependent.available() || pending->source.update().valid())
      return Status::Unavailable("native Graph pending outcome/source remains owned");
    if(pending->submitted && !pending->terminal.ok())return Status::FailedPrecondition("native Graph pending epoch failed");
    return Status::Ok();
  }
  bool has_pending()const noexcept{
    if(!pending)return false;
    return std::any_of(pending->frames.begin(),pending->frames.end(),[](const auto&f){return f[0].valid() || f[1].valid();});
  }
  bool pending_input(std::size_t input)const noexcept{
    if(!pending)return false;
    for(std::size_t i=0;i<slots.size();++i)for(std::size_t p=1;p<3;++p)
      if(slots[i]->inputs[p]==input && pending->frames[i][p-1].valid())return true;
    return false;
  }
  Status commit_pending(std::size_t input,compute::PagedArenaPendingRead frame)noexcept{
    if(!pending)return Status::FailedPrecondition("reserve native Graph pending source capacity first");
    HYLE_RETURN_IF_ERROR(reusable());HYLE_RETURN_IF_ERROR(validate_frame(input,frame));
    for(std::size_t i=0;i<slots.size();++i)for(std::size_t p=1;p<3;++p)
      if(slots[i]->inputs[p]==input && (slots[i]->frames[p-1].valid() || pending->frames[i][p-1].valid()))
        return Status::FailedPrecondition("native Graph page input needs reset before rebinding");
    for(std::size_t i=0;i<slots.size();++i)for(std::size_t p=1;p<3;++p)
      if(slots[i]->inputs[p]==input)pending->frames[i][p-1]=frame;
    return Status::Ok();
  }
  Status take_source(typename Traits::CopyUpdate&&update)noexcept{
    HYLE_RETURN_IF_ERROR(reusable());
    if(!prepared || !has_pending())return Status::FailedPrecondition("native Graph needs prepared pending frames");
    HYLE_ASSIGN_OR_RETURN(auto prefix,Traits::CopyAccess::dependency(update));
    for(std::size_t i=0;i<slots.size();++i){const auto&f=pending->frames[i];
      if(!f[0].valid() && !f[1].valid())continue;
      if(!f[0].valid() || !f[1].valid())return Status::InvalidArgument("native Graph pending node needs both exact KV frames");
      HYLE_RETURN_IF_ERROR(Traits::qualify_source(slots[i]->preparation,prefix));
      for(const auto&frame:f)HYLE_RETURN_IF_ERROR((compute::detail::PagedArenaAccess::validate<typename Traits::CopyUpdate,typename Traits::CopyAccess>(frame,update)));
    }
    HYLE_ASSIGN_OR_RETURN(pending->sources,pending->source.take(std::move(update)));return Status::Ok();
  }
  void restore_source(typename Traits::CopyUpdate&update)noexcept{
    update=std::move(pending->source.update());
    const auto sources=std::exchange(pending->sources,{});sources.release(sources.state);
  }
  auto submit_fragment(Slot&slot,compute::ReadLease query)noexcept->Result<typename Traits::Submission>{
    const auto found=std::find_if(slots.begin(),slots.end(),[&](const auto&s){return s.get()==&slot;});
    const auto index=std::size_t(found-slots.begin());
    if(pending && (pending->frames[index][0].valid() || pending->frames[index][1].valid())){
      auto&f=pending->frames[index];
      HYLE_ASSIGN_OR_RETURN(auto prefix,Traits::CopyAccess::dependency(pending->source.update()));
      HYLE_RETURN_IF_ERROR(Traits::qualify_source(slot.preparation,prefix));
      HYLE_RETURN_IF_ERROR(slot.controller.bind_pages(std::move(query),f[0],f[1]));
      return slot.controller.template submit_pending<typename Traits::CopyUpdate,typename Traits::CopyAccess>(f[0],f[1],pending->source.update());
    }
    HYLE_RETURN_IF_ERROR(slot.controller.bind(std::move(query),slot.frames[0],slot.frames[1]));
    return slot.controller.submit();
  }
  template<class Result>void map_result(Result&result,void*owner,
      compute::detail::CompletionLifetime::ReleaseFn release)noexcept{
    auto&p=*pending;p.outcome=compute::detail::ResultAccess::storage(result);p.submitted=true;
    p.execution_owner=owner;p.release_owner=release;
    p.retention.own={&p,[](void*opaque,const Status&status)noexcept{static_cast<Pending*>(opaque)->terminal=status;},
      [](void*opaque)noexcept{auto*p=static_cast<Pending*>(opaque);
        const auto release=p->release_owner;auto*owner=p->execution_owner;release(owner);}};
    auto&completion=compute::detail::ResultAccess::completion(result);
    completion=p.dependent.publish(std::move(completion),std::exchange(p.sources,{}));
    HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(completion,p.retention).ok());
  }
  Result<typename Traits::CopyUpdate> take_update()noexcept{
    if(!pending || !pending->source.available() || !pending->dependent.available() ||
        !pending->source.update().valid() || pending->source.update().in_flight())
      return std::unexpected(Status::Unavailable("native Graph source/outcome remains retained"));
    if(pending->outcome){HYLE_RETURN_IF_ERROR(compute::detail::begin_result(pending->outcome));
      compute::detail::rollback_result(pending->outcome);pending->outcome=nullptr;}
    return std::move(pending->source.update());
  }
  void reset() noexcept {
    prepared=false;for(auto& slot:slots)slot->frames={};
    if(pending){for(auto&f:pending->frames)f={};}
  }
};
} // namespace hyle::graph::batch::detail
