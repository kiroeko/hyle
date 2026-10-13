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
  };
  struct Delete final {
    mem::AllocatorRef allocator;
    void operator()(Slot* slot)const noexcept{allocator.destroy(slot);}
  };
  using Owner=std::unique_ptr<Slot,Delete>;
  explicit NativePagedSlots(mem::AllocatorRef a):allocator(a),slots(mem::StdAllocator<Owner>(a)) {}
  mem::AllocatorRef allocator;
  std::vector<Owner,mem::StdAllocator<Owner>> slots;
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
    prepared=false;HYLE_RETURN_IF_ERROR(validate_layouts(layouts));
    for(auto& slot:slots){
      slot->frames={};
      const auto found=std::find_if(layouts.begin(),layouts.end(),[&](const auto& l){return l.node==slot->binding->node;});
      HYLE_RETURN_IF_ERROR(slot->controller.prepare(found->layout));
    }
    prepared=true;return Status::Ok();
  }
  Status validate_frame(std::size_t input,const compute::PagedArenaRead& frame)const noexcept {
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
  void reset() noexcept {prepared=false;for(auto& slot:slots)slot->frames={};}
};
} // namespace hyle::graph::batch::detail
