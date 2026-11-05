#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <vector>

#include "paged_layout_scratch.h"
#include "paged_sequence.h"
#include "hyle/graph/paged_append_source.h"

namespace hyle::inference {

// Engine policy for one logical sequence. Graph/Compute retain their general
// R-capacity contract; this helper owns no model acceptance alternatives.
struct PagedAppendCapacityBounds final {
  std::size_t minimum_before=0,maximum_before=0,maximum_after=0,physical_query_rows=0;
};
struct PagedAppendCapacityScratch final {
  PagedLayoutScratch minimum,previous,next;
  std::vector<compute::PagedArenaFootprint> sources;
  void prepare(std::size_t max_page_refs,std::size_t max_banks){
    minimum.prepare(1,max_page_refs);previous.prepare(1,max_page_refs);next.prepare(1,max_page_refs);
    sources.reserve(max_banks);sources.clear();
  }
};

// The caller retains the complete raw page-reference cohort until native
// retirement, separately from any selected canonical sequence prefix. Late
// failure leaves every acquired policy reference/writer in those containers.
inline Result<compute::BatchLayoutVersion> prepare_paged_append_capacity(
    const compute::BatchSchema&schema,const PagedSequenceState&full_base,
    PagedAppendCapacityBounds bounds,PagedPageInventory&inventory,PagedAppendCapacityScratch&scratch,
    std::span<std::vector<compute::PagedPageWriter>>dirty,
    std::span<graph::batch::PagedAppendPages>appends,std::vector<PagedPageRef>&raw_pages,
    std::span<graph::batch::PagedAppendSource>incoming={},graph::batch::PagedAppendOptions options={})noexcept{
  scratch.sources.clear();
  struct ReleaseSources final{std::vector<compute::PagedArenaFootprint>&values;~ReleaseSources(){values.clear();}}release{scratch.sources};
  const auto cap=schema.capacity();const auto banks=inventory.banks.size();
  constexpr auto imax=static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  if(!schema.valid() || cap.storage!=compute::BatchStorage::kPaged || cap.max_requests!=1 ||
      !cap.page_size || cap.page_size>imax || banks!=full_base.frames.size() || banks!=dirty.size() || banks!=appends.size() ||
      (!incoming.empty() && incoming.size()!=banks) || inventory.owners.size()>cap.max_pages || inventory.owners.size()>imax)
    return std::unexpected(Status::InvalidArgument("paged capacity append differs from its original R1 schema/banks"));
  if(bounds.minimum_before>bounds.maximum_before || bounds.minimum_before>bounds.maximum_after ||
      bounds.maximum_before>cap.max_kv_rows || bounds.maximum_before>imax || bounds.maximum_after>cap.max_kv_rows ||
      bounds.maximum_after>imax || !bounds.physical_query_rows || bounds.physical_query_rows>cap.max_query_rows ||
      bounds.physical_query_rows>imax)
    return std::unexpected(Status::OutOfRange("paged append capacity row bounds exceed the admitted sequence"));
  const auto page_count=[&](std::size_t rows){return rows/cap.page_size+(rows%cap.page_size!=0);};
  const auto old_count=page_count(bounds.maximum_before),minimum_count=page_count(bounds.minimum_before),
      next_count=page_count(bounds.maximum_after),changed_begin=bounds.minimum_before/cap.page_size;
  if(old_count>full_base.pages.size() || next_count>cap.max_page_refs || old_count>cap.max_page_refs ||
      minimum_count>scratch.minimum.max_page_refs() || old_count>scratch.previous.max_page_refs() ||
      next_count>scratch.next.max_page_refs() || next_count<changed_begin)
    return std::unexpected(Status::OutOfRange("paged append capacity exceeds its original pages or cold scratch"));
  if(next_count-changed_begin>inventory.free_ids.size())
    return std::unexpected(Status::ResourceExhausted("paged append capacity exceeds the available original page inventory"));
  if(scratch.sources.capacity()<banks || !raw_pages.empty())
    return std::unexpected(Status::FailedPrecondition("paged append capacity needs prepared scratch and a retired page cohort"));
  for(std::size_t bank=0;bank<banks;++bank)
    if(!inventory.banks[bank].valid() || appends[bank].valid() || !dirty[bank].empty() ||
        (!incoming.empty() && !incoming[bank].valid()))
      return std::unexpected(Status::FailedPrecondition("paged append capacity bank still belongs to another episode"));
  if(next_count>raw_pages.max_size())return std::unexpected(Status::OutOfRange("paged append capacity reference metadata overflows"));
  for(const auto&bank:dirty)if(next_count-changed_begin>bank.max_size())
    return std::unexpected(Status::OutOfRange("paged append capacity writer metadata overflows"));
  for(const auto&page:full_base.pages)
    if(page.page()>=cap.max_pages || page.page()>=inventory.owners.size() || !inventory.owners[page.page()] || page.page()>imax)
      return std::unexpected(Status::InvalidArgument("paged append capacity has a foreign original page reference"));
  try{
    scratch.minimum.reset();scratch.previous.reset();scratch.next.reset();
    const auto base_pages=std::span<const PagedPageRef>(full_base.pages);
    HYLE_RETURN_IF_ERROR(scratch.minimum.append(0,bounds.minimum_before,cap.page_size,base_pages.first(minimum_count)));
    HYLE_RETURN_IF_ERROR(scratch.previous.append(0,bounds.maximum_before,cap.page_size,base_pages.first(old_count)));
    HYLE_ASSIGN_OR_RETURN(auto minimum,schema.version(scratch.minimum.layout()));
    HYLE_ASSIGN_OR_RETURN(auto previous,schema.version(scratch.previous.layout()));
    if(incoming.empty())for(std::size_t bank=0;bank<banks;++bank){
      HYLE_ASSIGN_OR_RETURN(auto source,full_base.frames[bank].view());const auto layout=source.layout();
      if(!inventory.banks[bank].owns(source) || !schema.owns(layout) || layout.facts().requests!=1 ||
          layout.facts().kv_rows<bounds.maximum_before || layout.host().page_indices.size()!=full_base.pages.size())
        return std::unexpected(Status::InvalidArgument("paged append capacity differs from its ready base footprint"));
      for(std::size_t page=0;page<full_base.pages.size();++page)
        if(layout.host().page_indices[page]!=static_cast<std::int32_t>(full_base.pages[page].page()))
          return std::unexpected(Status::InvalidArgument("paged append capacity ready base has different physical IDs"));
      scratch.sources.push_back(std::move(source));
    }
    raw_pages.reserve(next_count);for(auto&bank:dirty)bank.reserve(next_count-changed_begin);
    for(std::size_t page=0;page<next_count;++page){
      if(page<changed_begin)raw_pages.push_back(full_base.pages[page]);
      else{HYLE_ASSIGN_OR_RETURN(auto acquired,inventory.acquire_into(dirty));raw_pages.push_back(std::move(acquired));}
    }
    HYLE_RETURN_IF_ERROR(scratch.next.append(bounds.physical_query_rows,bounds.maximum_after,cap.page_size,
        std::span<const PagedPageRef>(raw_pages)));
    HYLE_ASSIGN_OR_RETURN(auto next,schema.version(scratch.next.layout()));
    for(std::size_t bank=0;bank<banks;++bank){
      if(incoming.empty()){
        HYLE_ASSIGN_OR_RETURN(appends[bank],graph::batch::PagedAppendPages::create_lengths(inventory.banks[bank],next,
            minimum,previous,std::span<const compute::PagedArenaFootprint>(&scratch.sources[bank],1),dirty[bank],options));
      }else{
        HYLE_ASSIGN_OR_RETURN(appends[bank],graph::batch::PagedAppendPages::create_lengths(inventory.banks[bank],next,
            minimum,incoming[bank],dirty[bank],options));
      }
    }
    return next;
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("paged append capacity host metadata allocation failed"));}
}

} // namespace hyle::inference
