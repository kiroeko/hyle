#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "append_transactions.h"
#include "paged_layout_scratch.h"
#include "paged_sequence.h"
#include "hyle/graph/paged_append_source.h"

namespace hyle::inference {

// Host integers and synchronous footprint borrows only. The caller cold
// prepares this scratch; native writers, policy refs and transactions stay in
// the original episode containers, including on every late failure.
struct PagedAppendChoiceScratch final {
  std::array<PagedLayoutScratch, 2> previous_rows, next_rows;
  std::vector<compute::PagedArenaFootprint> sources;

  void prepare(std::size_t max_page_refs, std::size_t max_banks) {
    for (auto &rows : previous_rows) rows.prepare(1, max_page_refs);
    for (auto &rows : next_rows) rows.prepare(1, max_page_refs);
    sources.reserve(max_banks);
    sources.clear();
  }
};

// Exactly one logical member and two cold-validated possibilities. Choice1
// is the maximum visible envelope. Each logical page changed by either
// choice has one new physical ID shared by both next page tables wherever
// that page exists; an otherwise unchanged prefix is copied to that same ID.
// No model acceptance, state version, submission or publication is owned here.
inline Result<std::array<compute::BatchLayoutVersion, 2>> prepare_paged_append_choices(
    const compute::BatchSchema &schema, const PagedSequenceState &full_base,
    std::array<std::size_t, 2> before_rows, std::array<std::size_t, 2> append_rows,
    PagedPageInventory &inventory, PagedAppendChoiceScratch &scratch,
    std::span<std::vector<compute::PagedPageWriter>> dirty,
    std::span<graph::batch::PagedAppendPages> appends,
    std::array<std::vector<PagedPageRef>, 2> &choice_pages,
    std::span<graph::batch::PagedAppendSource> incoming = {},
    graph::batch::PagedAppendOptions options = {},
    bool zero_fill_envelope = false) noexcept {
  scratch.sources.clear();
  struct ReleaseSources final {
    std::vector<compute::PagedArenaFootprint> &values;
    ~ReleaseSources() { values.clear(); }
  } release_sources{scratch.sources};
  const auto capacity = schema.capacity();
  const auto banks = inventory.banks.size();
  constexpr auto integer_max = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  if (!schema.valid() || capacity.storage != compute::BatchStorage::kPaged ||
      capacity.max_requests < 1 || !capacity.page_size || capacity.page_size > integer_max ||
      banks != full_base.frames.size() || banks != dirty.size() || banks != appends.size() ||
      (!incoming.empty() && incoming.size() != banks) ||
      inventory.owners.size() > capacity.max_pages || inventory.owners.size() > integer_max)
    return std::unexpected(Status::InvalidArgument("paged append choices differ from their R1 schema or physical banks"));
  if (scratch.sources.capacity() < banks)
    return std::unexpected(Status::FailedPrecondition("paged append choice source scratch was not cold prepared"));
  for (std::size_t bank = 0; bank < banks; ++bank) {
    if (!inventory.banks[bank].valid() || appends[bank].valid() || !dirty[bank].empty())
      return std::unexpected(Status::FailedPrecondition("paged append choice bank still belongs to an earlier episode"));
    if (!incoming.empty() && !incoming[bank].valid())
      return std::unexpected(Status::FailedPrecondition("paged append choices require their actual captured predecessor Source"));
  }
  for (const auto &pages : choice_pages)
    if (!pages.empty())
      return std::unexpected(Status::FailedPrecondition("paged append choice page references still belong to an earlier episode"));

  std::array<AppendExtent, 2> extents;
  std::array<std::size_t, 2> previous_pages;
  for (std::size_t choice = 0; choice < 2; ++choice) {
    HYLE_ASSIGN_OR_RETURN(extents[choice], append_extent(before_rows[choice], append_rows[choice],
        capacity.max_kv_rows, capacity.page_size));
    previous_pages[choice] = before_rows[choice] / capacity.page_size +
        (before_rows[choice] % capacity.page_size != 0);
    if (before_rows[choice] > integer_max || extents[choice].after_rows > integer_max ||
        append_rows[choice] > capacity.max_query_rows || append_rows[choice] > integer_max ||
        extents[choice].page_count > capacity.max_page_refs || extents[choice].page_count > integer_max ||
        previous_pages[choice] > full_base.pages.size() ||
        extents[choice].page_count > scratch.next_rows[choice].max_page_refs() ||
        previous_pages[choice] > scratch.previous_rows[choice].max_page_refs())
      return std::unexpected(Status::OutOfRange("paged append choice exceeds its original row, page or cold scratch capacity"));
  }
  if (extents[1].after_rows < extents[0].after_rows)
    return std::unexpected(Status::InvalidArgument("paged append choice1 must cover choice0's complete visible envelope"));
  for (const auto &page : full_base.pages)
    if (page.page() >= capacity.max_pages || page.page() >= inventory.owners.size() ||
        !inventory.owners[page.page()] || page.page() > integer_max)
      return std::unexpected(Status::InvalidArgument("paged append choices contain an out-of-range original page reference"));
  const auto changed = [&](std::size_t page) noexcept {
    return (page >= extents[0].prefix_pages && page < extents[0].page_count) ||
        (page >= extents[1].prefix_pages && page < extents[1].page_count);
  };
  std::size_t changed_pages = 0;
  for (std::size_t page = 0; page < extents[1].page_count; ++page) {
    if (changed(page)) ++changed_pages;
    else if (page >= full_base.pages.size())
      return std::unexpected(Status::InvalidArgument("paged append choice retained prefix is absent from its original page table"));
  }
  if (changed_pages > inventory.free_ids.size())
    return std::unexpected(Status::ResourceExhausted("paged append choices exceed the available original physical page inventory"));
  for (std::size_t choice = 0; choice < 2; ++choice)
    if (extents[choice].page_count > choice_pages[choice].max_size())
      return std::unexpected(Status::OutOfRange("paged append choice reference metadata extent overflows"));
  for (const auto &bank : dirty)
    if (changed_pages > bank.max_size())
      return std::unexpected(Status::OutOfRange("paged append choice writer metadata extent overflows"));

  // All known row/page/schema bounds precede any acquisition. These versions
  // describe real old page IDs; the low-level choice constructor separately
  // validates their exact ready/pending native coverage before moving writers.
  std::array<graph::batch::PagedAppendChoice, 2> choices;
  try {
    for (std::size_t choice = 0; choice < 2; ++choice) {
      scratch.previous_rows[choice].reset();
      scratch.next_rows[choice].reset();
      HYLE_RETURN_IF_ERROR(scratch.previous_rows[choice].append(0, before_rows[choice], capacity.page_size,
          std::span<const PagedPageRef>(full_base.pages).first(previous_pages[choice])));
      HYLE_ASSIGN_OR_RETURN(choices[choice].previous, schema.version(scratch.previous_rows[choice].layout()));
    }
    if (incoming.empty()) {
      for (std::size_t bank = 0; bank < banks; ++bank) {
        HYLE_ASSIGN_OR_RETURN(auto source, full_base.frames[bank].view());
        const auto layout = source.layout();
        const auto facts = layout.facts();
        const auto host = layout.host();
        if (!inventory.banks[bank].owns(source) || !schema.owns(layout) || facts.requests != 1 ||
            facts.kv_rows < (std::max)(before_rows[0], before_rows[1]) ||
            host.page_indices.size() != full_base.pages.size())
          return std::unexpected(Status::InvalidArgument("paged append choices differ from their actual ready base footprint"));
        for (std::size_t page = 0; page < full_base.pages.size(); ++page)
          if (host.page_indices[page] != static_cast<std::int32_t>(full_base.pages[page].page()))
            return std::unexpected(Status::InvalidArgument("paged append choice ready base has different original physical page IDs"));
        scratch.sources.push_back(std::move(source));
      }
    }
    // Reserve the complete caller-owned cohorts before the first real writer.
    // All acquired refs are immediately stored in choice1 (the max envelope),
    // so no late error can release a policy ID ahead of its native writers.
    for (std::size_t choice = 0; choice < 2; ++choice) choice_pages[choice].reserve(extents[choice].page_count);
    for (auto &bank : dirty) bank.reserve(changed_pages);
    for (std::size_t page = 0; page < extents[1].page_count; ++page) {
      if (changed(page)) {
        HYLE_ASSIGN_OR_RETURN(auto acquired, inventory.acquire_into(dirty));
        choice_pages[1].push_back(std::move(acquired));
      } else choice_pages[1].push_back(full_base.pages[page]);
      if (page < extents[0].page_count) choice_pages[0].push_back(choice_pages[1].back());
    }
    std::array<compute::BatchLayoutVersion, 2> next;
    for (std::size_t choice = 0; choice < 2; ++choice) {
      HYLE_RETURN_IF_ERROR(scratch.next_rows[choice].append(append_rows[choice], extents[choice].after_rows,
          capacity.page_size, std::span<const PagedPageRef>(choice_pages[choice])));
      HYLE_ASSIGN_OR_RETURN(choices[choice].next, schema.version(scratch.next_rows[choice].layout()));
      next[choice] = choices[choice].next;
    }
    for (std::size_t bank = 0; bank < banks; ++bank) {
      if (incoming.empty()) {
        HYLE_ASSIGN_OR_RETURN(appends[bank], graph::batch::PagedAppendPages::create_choices(
            inventory.banks[bank], choices, std::span<const compute::PagedArenaFootprint>(&scratch.sources[bank], 1),
            dirty[bank], zero_fill_envelope, options));
      } else {
        HYLE_ASSIGN_OR_RETURN(appends[bank], graph::batch::PagedAppendPages::create_choices(
            inventory.banks[bank], choices, incoming[bank], dirty[bank], zero_fill_envelope, options));
      }
    }
    return next;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("paged append choice host metadata allocation failed"));
  }
}

} // namespace hyle::inference
