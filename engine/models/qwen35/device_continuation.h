#pragma once

// Included after the original Model definition. This is the Qwen-specific
// physical input preparation of one original successor episode; it neither
// chooses the producer's result on host nor publishes another model state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "../../inference/paged_append_capacity.h"

namespace qwen35 {

inline void Model::prepare_device_mtp_inputs(StepState &step, bool recording_only) {
  if (!step.phase || !step.phase->family || step.members.size() != 1 ||
      step.phase->family->role != GraphRole::kMtpContinuation ||
      step.phase->family->packing != RowPacking::kRectangular ||
      step.phase->family->completion != TextCompletion::kEmitNext || step.phase->query_rows != step.phase->prefix_rows ||
      step.state_slot >= step.phase->slots.size())
    throw std::logic_error("device MTP inputs require the original greedy R1 continuation episode");
  auto &phase = *step.phase;
  auto &io = phase.slots[step.state_slot].data;
  auto &member = step.members[0];
  const auto outcomes=phase.prefix_rows,proposals=outcomes-1,chain_steps=phase.draft_chain_steps;
  const auto append_banks=draft_bank+2+2*chain_steps;
  if (!member.base || !member.base->draft || !member.session || !member.owner_state || member.next ||
      outcomes<2 || member.input_tokens != outcomes || step.execution.dirty.size() != append_banks ||
      step.execution.tails.size() != append_banks || step.execution.appends.size() != append_banks ||
      io.ordinary_io.packet() || !phase.predecessor_prefix_port)
    throw std::logic_error("device MTP input preparation lost its original base, carrier or bank cohort");

  std::vector<std::size_t>target_before(outcomes),draft_before(outcomes);
  std::span<g::batch::PagedAppendSource> target_incoming, draft_incoming;
  if (recording_only) {
    if (step.execution.incoming || member.base->length || member.base->draft->frontier ||
        !member.base->paged.pages.empty() || !member.base->draft->paged.pages.empty())
      throw std::logic_error("record-only continuation inputs require their actual empty original base frames");
  } else {
    if (!step.execution.incoming || !step.execution.incoming->owner)
      throw std::logic_error("device MTP inputs require their actual producer Source");
    auto &incoming = *step.execution.incoming->owner;
    if (incoming.boundaries.size() != 1 || incoming.pages.size() != draft_bank + 2)
      throw std::logic_error("device MTP input Source has a different original member or bank cohort");
    const auto &boundary = incoming.boundaries[0];
    if (boundary.candidate != member.base || boundary.epoch != member.base_epoch ||
        boundary.session.lock() != member.session || boundary.length < outcomes ||
        boundary.length != member.base->length || boundary.draft_frontier != member.base->draft->frontier ||
        boundary.draft_frontier != boundary.length - 1)
      throw std::logic_error("device MTP input Source differs from its original full candidate boundary");
    for(std::size_t i=0;i<outcomes;++i){target_before[i]=boundary.length-proposals+i;
      draft_before[i]=target_before[i]-(i?1u:0u);}
    target_incoming = std::span(incoming.pages).first(draft_bank);
    draft_incoming = std::span(incoming.pages).subspan(draft_bank, 2);
  }
  constexpr auto integer_max = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  if (capacity < outcomes || outcomes>integer_max || target_before.back() > capacity - outcomes ||
      target_before.back() > integer_max - outcomes || draft_before.back()>target_before.back())
    throw std::out_of_range("device MTP input choices exceed their original logical or I32 capacity");
  std::size_t maximum_draft_after=0;
  for(std::size_t i=0;i<outcomes;++i)maximum_draft_after=(std::max)(maximum_draft_after,draft_before[i]+proposals+(i?1u:0u));

  member.choice_base_length = target_before[0];
  member.choice_draft_frontier = draft_before[0];
  member.selected_input_choice.reset();
  member.draft_input_rows = 0;
  step.active_draft_rows = 0;
  step.device_input_choice = true;
  step.draft_rows_deferred = !recording_only;
  // Keep the original begin_member ordering and candidate owner. These are
  // physical upper bounds, not a claim that the producer accepted two rows.
  for (auto &tails : step.execution.tails)
    if (!tails.try_reserve(1)) throw std::bad_alloc{};
  member.next = reserve_state(*member.owner_state);
  if (!member.next_tails.try_resize(draft_bank)) throw std::bad_alloc{};
  member.next->length = target_before.back() + outcomes;
  member.next->sampling = member.base->sampling;
  member.next->draft.emplace();
  if (!member.next->draft->paged.frames.try_resize(2)) throw std::bad_alloc{};
  member.next->draft->frontier = maximum_draft_after;

  // As in the ordinary append path, a ready sealed base needs an unsealed
  // metadata fork in this same candidate. A pending A must first publish its
  // original Frame; never fork that producer's old boundary before retirement.
  if (!step.execution.predecessor_episode) {
    for (std::size_t bank = 0; bank < draft_bank; ++bank) {
      auto source = take(inference::PagedSequenceEdit(member.base->paged, member.next->paged).source(bank));
      (void)source;
    }
    for (std::size_t bank = 0; bank < 2; ++bank) {
      auto source = take(inference::PagedSequenceEdit(member.base->draft->paged, member.next->draft->paged).source(bank));
      (void)source;
    }
  }
  member.target_capacity_layout = take(inference::prepare_paged_append_capacity(attention_schema,
      member.base->paged, {target_before.front(),target_before.back(),target_before.back()+outcomes,outcomes}, page_inventory, input_scratch.paged_append_capacity,
      std::span(step.execution.dirty.data(), draft_bank),
      std::span(step.execution.appends.data(), draft_bank), member.target_capacity_pages,
      target_incoming, {.resource_budget = cache_budget}));
  member.next->paged.pages = member.target_capacity_pages;
  member.layout = io.attention_layout = member.target_capacity_layout;
  member.draft_capacity_layout = take(inference::prepare_paged_append_capacity(attention_schema,
      member.base->draft->paged, {draft_before.front(),draft_before.back(),maximum_draft_after,outcomes}, draft_inventory, input_scratch.paged_append_capacity,
      std::span(step.execution.dirty.data() + draft_bank, 2),
      std::span(step.execution.appends.data() + draft_bank, 2), member.draft_capacity_pages,
      draft_incoming, {.resource_budget = cache_budget}));
  member.next->draft->paged.pages = member.draft_capacity_pages;
  member.draft_layout = step.draft_layout = member.draft_capacity_layout;
  const auto view=[&](const auto&pages,std::size_t q,std::size_t kv){
    auto&rows=input_scratch.paged_append_capacity.next;rows.reset();
    check(rows.append(q,kv,cache::Plan::page_rows,std::span<const inference::PagedPageRef>(pages).first(
        (kv+cache::Plan::page_rows-1)/cache::Plan::page_rows)));
    return take(attention_schema.version(rows.layout()));
  };
  member.target_input_choices.resize(target_before.size());member.draft_input_choices.resize(draft_before.size());
  for(std::size_t i=0;i<target_before.size();++i){
    member.target_input_choices[i]=i+1==outcomes?member.target_capacity_layout:
        view(member.target_capacity_pages,outcomes,target_before[i]+outcomes);
    const auto rows=proposals+(i?1u:0u);
    member.draft_input_choices[i]=i+1==outcomes?member.draft_capacity_layout:
        view(member.draft_capacity_pages,rows,draft_before[i]+rows);
  }
  member.draft_chain_pages.resize(chain_steps);member.draft_chain_layouts.resize(chain_steps);
  if(chain_steps && !recording_only && step.execution.incoming->owner->draft_stage_sources.size()!=2*chain_steps)
    throw std::logic_error("device draft stages lost their original independent Source reader bundles");
  for(std::size_t stage=0;stage<chain_steps;++stage){
    auto&pages=member.draft_chain_pages[stage];if(!pages.frames.try_resize(2))throw std::bad_alloc{};
    std::size_t maximum_after=0;
    for(std::size_t i=0;i<outcomes;++i)maximum_after=(std::max)(maximum_after,draft_before[i]+1+(i?1u:0u)+stage);
    const auto bank=draft_bank+2+2*stage;
    const auto incoming=recording_only?std::span<g::batch::PagedAppendSource>{}:
        std::span(step.execution.incoming->owner->draft_stage_sources).subspan(2*stage,2);
    const auto writer_capacity=take(inference::prepare_paged_append_capacity(attention_schema,
        member.base->draft->paged,{draft_before.front(),draft_before.back(),maximum_after,stage+2},draft_inventory,
        input_scratch.paged_append_capacity,std::span(step.execution.dirty.data()+bank,2),
        std::span(step.execution.appends.data()+bank,2),pages.pages,incoming,{.resource_budget=cache_budget}));
    member.draft_chain_layouts[stage]=view(pages.pages,1,writer_capacity.facts().kv_rows);
  }
  if(chain_steps){member.verification_inputs.assign(outcomes,0);member.verification_inputs[0]=member.first_input;}

  // One checked table row supplies RoPE positions and the original Q/KV
  // controls. KV is shared; each private head needs only its final query.
  std::vector<std::int32_t>target_controls,draft_controls;
  for(std::size_t i=0;i<outcomes;++i){
    for(std::size_t row=0;row<outcomes;++row)target_controls.push_back(static_cast<std::int32_t>(target_before[i]+row));
    target_controls.push_back(static_cast<std::int32_t>(outcomes));target_controls.push_back(static_cast<std::int32_t>(target_before[i]+outcomes));
    const auto append_controls=[&](std::size_t physical,std::size_t active){
      for(std::size_t row=0;row<physical;++row)draft_controls.push_back(static_cast<std::int32_t>(draft_before[i]+(std::min)(row,active-1)));
      draft_controls.push_back(static_cast<std::int32_t>(active));draft_controls.push_back(static_cast<std::int32_t>(draft_before[i]+active));
    };
    append_controls(outcomes,proposals+(i?1u:0u));
    for(std::size_t stage=0;stage<chain_steps;++stage)append_controls(stage+2,stage+1+(i?1u:0u));
  }
  io.positions.write<std::int32_t>(target_controls);
  io.draft_positions.write<std::int32_t>(draft_controls);
  if (recording_only) {
    // Safe storage initialization only. The recording owner may subsequently
    // lend A's physical OutputSource coordinates without observing their value.
    // No selector is marked known and no actual draft-row counter is inferred.
    constexpr std::array<std::int32_t, 1> zero{0};
    io.predecessor_prefix.write<std::int32_t>(zero);
    io.tokens.write<std::int32_t>(zero);
    io.catchup_token.write<std::int32_t>(zero);
  }
}

} // namespace qwen35
