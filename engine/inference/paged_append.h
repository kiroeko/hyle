#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "append_transactions.h"
#include "paged_layout_scratch.h"
#include "paged_sequence.h"
#include "hyle/graph/paged_append_source.h"

namespace hyle::inference {

// Cold host capacity shared by synchronous group operations. Source views are
// temporary borrows: each preparation clears them on success and every error.
// No span into this scratch is retained by a member or a later operation.
struct PagedAppendScratch final {
  PagedLayoutScratch old_rows, next_rows;
  std::vector<compute::PagedArenaFootprint> sources;
  std::vector<AppendTailOwner> tail_owners;

  void prepare(std::size_t max_members, std::size_t max_page_refs) {
    const auto count = (std::max)(members_, max_members);
    old_rows.prepare(count, max_page_refs); next_rows.prepare(count, max_page_refs);
    sources.reserve(count); tail_owners.reserve(count);
    sources.clear(); tail_owners.clear(); members_ = count;
  }
  std::size_t max_members() const noexcept { return members_; }

 private:
  std::size_t members_ = 0;
};

template<class TailBank>
struct PagedAppendBanks final {
  PagedPageInventory &inventory;
  std::span<std::vector<compute::PagedPageWriter>> dirty;
  std::span<TailBank> tails;
  std::span<graph::batch::PagedAppendPages> appends;
};
template<class TailBank, std::size_t D, std::size_t T, std::size_t A>
PagedAppendBanks(PagedPageInventory &, std::span<std::vector<compute::PagedPageWriter>, D>,
    std::span<TailBank, T>, std::span<graph::batch::PagedAppendPages, A>) -> PagedAppendBanks<TailBank>;

// The candidate may be absent during pure whole-group admission. begin(index)
// then creates the original model candidate before this view is borrowed again.
// Row facts describe this actual append, independently of model cache position.
struct PagedAppendMember final {
  const PagedSequenceState &base;
  PagedSequenceState *candidate;
  compute::BatchLayoutVersion &layout;
  std::size_t before_rows, append_rows;
  std::span<compute::PagedPageAppender> reusable_tails{};
  bool allow_tail_reuse = false;
};

struct PagedAppendPreparation final {
  std::size_t capacity_rows = 0;
  std::span<graph::batch::PagedAppendSource> incoming{};
  bool pending_predecessor = false;
  const compute::HostBatchLayout *previous_layout = nullptr;
  graph::batch::PagedAppendOptions append_options{};
};

// Accepted layouts and original physical tail owners are separate facts. A
// prefix selection may have removed the originally written final page, and a
// withdrawn member still belongs to whole-transaction capability retirement.
struct PagedAppendFinishMember final {
  const PagedSequenceState &base;
  PagedSequenceState &candidate;
  std::span<compute::PagedArenaCommit> commits;
  const compute::BatchLayoutVersion &accepted_layout;
  std::uint32_t physical_last_page, initialized_rows;
  std::span<compute::PagedPageAppender> returned_tails{};
  bool active = true, keep_tail = true;
};

namespace paged_append_detail {
template<class T>
struct ClearOnReturn final {
  std::vector<T> &values;
  ~ClearOnReturn() { values.clear(); }
};

template<class TailBank>
Status reserve_tails(std::span<TailBank> tails, std::size_t members) {
  for (auto &bank : tails) {
    if (members > (std::numeric_limits<std::size_t>::max)() - bank.size())
      return Status::OutOfRange("paged append tail metadata extent overflows");
    const auto required = bank.size() + members;
    if constexpr (requires { bank.try_reserve(required); }) {
      if (!bank.try_reserve(required))
        return Status::ResourceExhausted("paged append tail metadata allocation failed");
    } else bank.reserve(required);
  }
  return Status::Ok();
}

template<class FinishAt>
Status validate_finish(std::size_t banks, std::size_t count, FinishAt &member_at) noexcept {
  if (!count) return Status::InvalidArgument("paged append finish requires original members");
  for (std::size_t member = 0; member < count; ++member) {
    const auto view = member_at(member);
    if (view.active && (view.base.frames.size() != banks || view.candidate.frames.size() != banks ||
        view.commits.size() != banks || &view.base == &view.candidate))
      return Status::InvalidArgument("paged append finish differs from its original frame cohort");
  }
  return Status::Ok();
}
} // namespace paged_append_detail

// Complete page/layout/Append preparation over original episode containers.
// Accessors and invalidation are synchronous static borrows, never retained.
// begin preserves the model's original per-member candidate-allocation order.
// A late failure keeps all earlier refs/writers/tails/transactions in these
// containers for the enclosing original episode's retirement and recovery.
template<class TailBank, class MemberAt, class Invalidate = std::nullptr_t, class Begin = std::nullptr_t>
Result<compute::BatchLayoutVersion> prepare_paged_append(const compute::BatchSchema &schema,
    std::size_t count, PagedAppendBanks<TailBank> banks, PagedAppendScratch &scratch,
    MemberAt &&member_at, const PagedAppendPreparation &options,
    Invalidate &&invalidate = nullptr, Begin &&begin = nullptr) {
  static_assert(std::is_nothrow_invocable_v<MemberAt &, std::size_t> &&
      std::is_same_v<std::invoke_result_t<MemberAt &, std::size_t>, PagedAppendMember>,
      "paged member facts must be borrowed without throwing");
  static_assert(std::is_null_pointer_v<std::remove_cvref_t<Invalidate>> ||
      std::is_nothrow_invocable_v<Invalidate &, std::size_t>,
      "tail hint invalidation must not throw");
  static_assert(std::is_null_pointer_v<std::remove_cvref_t<Begin>> || std::is_invocable_v<Begin &, std::size_t>,
      "candidate preparation must be a synchronous member action");
  scratch.sources.clear();
  paged_append_detail::ClearOnReturn release_sources{scratch.sources};
  const auto bank_count = banks.inventory.banks.size();
  const auto capacity = schema.capacity();
  if (!schema.valid() || !count || count > capacity.max_requests || !capacity.page_size ||
      bank_count != banks.dirty.size() || bank_count != banks.tails.size() || bank_count != banks.appends.size() ||
      (!options.incoming.empty() && options.incoming.size() != bank_count) ||
      (options.pending_predecessor && options.incoming.empty() && bank_count))
    return std::unexpected(Status::InvalidArgument("paged append preparation differs from its schema or physical banks"));
  if (count > scratch.max_members() || count > scratch.sources.capacity())
    return std::unexpected(Status::FailedPrecondition("paged append member scratch was not cold prepared"));
  for (const auto &append : banks.appends)
    if (append.valid())
      return std::unexpected(Status::FailedPrecondition("paged append transaction still belongs to an earlier episode"));

  // All known row/reference bounds precede begin, page acquisition, tail
  // reservation or mutation of any original candidate. Candidate pointers are
  // deliberately not dereferenced during this pure admission pass.
  std::size_t query_rows = 0, kv_rows = 0, page_refs = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const auto member = member_at(index);
    HYLE_ASSIGN_OR_RETURN(const auto extent,
        append_extent(member.before_rows, member.append_rows, options.capacity_rows, capacity.page_size));
    const auto old_pages = member.before_rows / capacity.page_size + (member.before_rows % capacity.page_size != 0);
    if (member.base.frames.size() != bank_count || member.base.pages.size() != old_pages ||
        (!member.reusable_tails.empty() && member.reusable_tails.size() != bank_count))
      return std::unexpected(Status::InvalidArgument("paged append member differs from its original sequence banks or pages"));
    if (member.append_rows > capacity.max_query_rows - query_rows ||
        extent.after_rows > capacity.max_kv_rows - kv_rows ||
        extent.page_count > capacity.max_page_refs - page_refs ||
        extent.page_count > scratch.next_rows.max_page_refs() - page_refs)
      return std::unexpected(Status::OutOfRange("paged append group exceeds its prepared row or page-reference capacity"));
    query_rows += member.append_rows; kv_rows += extent.after_rows; page_refs += extent.page_count;
  }

  scratch.old_rows.reset(); scratch.next_rows.reset();
  try {
    HYLE_RETURN_IF_ERROR(paged_append_detail::reserve_tails(banks.tails, count));
    for (std::size_t index = 0; index < count; ++index) {
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Begin>>) begin(index);
      const auto member = member_at(index);
      if (!member.candidate || &member.base == member.candidate ||
          member.candidate->frames.size() != bank_count || !member.candidate->pages.empty())
        return std::unexpected(Status::InvalidArgument("paged append requires its fresh complete candidate sequence"));
      HYLE_ASSIGN_OR_RETURN(const auto extent,
          append_extent(member.before_rows, member.append_rows, options.capacity_rows, capacity.page_size));
      HYLE_ASSIGN_OR_RETURN(const bool reused, prepare_append_pages(extent,
          std::span<const PagedPageRef>(member.base.pages), member.candidate->pages,
          member.reusable_tails, member.allow_tail_reuse, banks.dirty, banks.tails,
          [&]() -> Result<PagedPageRef> {
            if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Invalidate>>) invalidate(index);
            return banks.inventory.acquire_into(banks.dirty);
          }));
      (void)reused;
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<Invalidate>>) invalidate(index);
      HYLE_RETURN_IF_ERROR(scratch.next_rows.append(member.append_rows, extent.after_rows, capacity.page_size,
          std::span<const PagedPageRef>(member.candidate->pages)));
      HYLE_ASSIGN_OR_RETURN(member.layout, scratch.next_rows.member_version(schema, index));
      if (count != 1 && !options.previous_layout)
        HYLE_RETURN_IF_ERROR(scratch.old_rows.append(0, member.before_rows, capacity.page_size,
            std::span<const PagedPageRef>(member.base.pages)));
    }
    compute::BatchLayoutVersion next;
    if (count == 1) next = member_at(0).layout;
    else { HYLE_ASSIGN_OR_RETURN(next, schema.version(scratch.next_rows.layout())); }
    compute::BatchLayoutVersion previous;
    if (count != 1 && options.incoming.empty() && bank_count) {
      HYLE_ASSIGN_OR_RETURN(previous, schema.version(options.previous_layout
          ? *options.previous_layout : scratch.old_rows.layout()));
    }
    for (std::size_t bank = 0; bank < bank_count; ++bank) {
      if (!options.incoming.empty()) {
        // A pending predecessor has no ready frame. When its original head is
        // already ready, fork sealed metadata once for the eventual commit.
        if (!options.pending_predecessor) for (std::size_t index = 0; index < count; ++index) {
          const auto member = member_at(index);
          HYLE_ASSIGN_OR_RETURN(auto source, PagedSequenceEdit(member.base, *member.candidate).source(bank));
          (void)source;
        }
        HYLE_ASSIGN_OR_RETURN(banks.appends[bank], graph::batch::make_append_from_source(
            banks.inventory.banks[bank], next, options.incoming[bank], banks.dirty[bank], options.append_options));
      } else {
        scratch.sources.resize(count); // The cold capacity was proved above.
        for (std::size_t index = 0; index < count; ++index) {
          const auto member = member_at(index);
          HYLE_ASSIGN_OR_RETURN(scratch.sources[index],
              PagedSequenceEdit(member.base, *member.candidate).source(bank));
        }
        const auto before = count == 1 ? scratch.sources[0].layout() : previous;
        HYLE_ASSIGN_OR_RETURN(banks.appends[bank], graph::batch::PagedAppendPages::create(
            banks.inventory.banks[bank], next, before, scratch.sources,
            banks.dirty[bank], banks.tails[bank], options.append_options));
        scratch.sources.clear();
      }
    }
    return next;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("paged append host metadata allocation failed"));
  }
}

// Caller has observed whole-Graph success and taken every original append.
// Each bank retains its original ready_delta authority, including an exported
// producer Source. Earlier prepared tickets survive any later refusal.
template<class FinishAt>
Status prepare_paged_append_commits(std::span<graph::batch::PagedAppendPages> appends,
    std::size_t count, FinishAt &&member_at,
    std::span<const graph::batch::PagedAppendSource> outgoing = {}) noexcept {
  static_assert(std::is_nothrow_invocable_v<FinishAt &, std::size_t> &&
      std::is_same_v<std::invoke_result_t<FinishAt &, std::size_t>, PagedAppendFinishMember>,
      "paged finish facts must be borrowed without throwing");
  if (!outgoing.empty() && outgoing.size() != appends.size())
    return Status::InvalidArgument("paged append ready sources differ from their original banks");
  HYLE_RETURN_IF_ERROR(paged_append_detail::validate_finish(appends.size(), count, member_at));
  for (std::size_t bank = 0; bank < appends.size(); ++bank)
    for (std::size_t index = 0; index < count; ++index) {
      const auto member = member_at(index);
      if (!member.active) continue;
      PagedSequenceEdit edit(member.base, member.candidate, member.commits);
      HYLE_ASSIGN_OR_RETURN(auto expected, edit.selected(bank).view());
      const auto ready_delta = [&]() noexcept -> Result<compute::PagedArenaDelta> {
        if (outgoing.empty()) return appends[bank].ready_delta(expected, member.accepted_layout);
        return outgoing[bank].ready_delta(appends[bank], expected, member.accepted_layout);
      };
      HYLE_ASSIGN_OR_RETURN(auto delta, ready_delta());
      HYLE_RETURN_IF_ERROR(edit.prepare_commit(bank, std::move(delta)));
    }
  return Status::Ok();
}

// A separate success-stage operation: callers retain their original ordering
// around model completion, accepted-prefix checks and joint head publication.
// Rebuild the original physical owner table at this call, including withdrawn
// members. Nested private draft work may have reused the same integer scratch.
template<class FinishAt>
Status return_paged_append_tails(const AppendTransactions &transactions, std::size_t bank_count,
    std::size_t count, PagedAppendScratch &scratch, FinishAt &&member_at) noexcept {
  static_assert(std::is_nothrow_invocable_v<FinishAt &, std::size_t> &&
      std::is_same_v<std::invoke_result_t<FinishAt &, std::size_t>, PagedAppendFinishMember>,
      "paged tail destinations must be borrowed without throwing");
  scratch.tail_owners.clear();
  paged_append_detail::ClearOnReturn release_owners{scratch.tail_owners};
  if (transactions.size() != bank_count || !count)
    return Status::InvalidArgument("paged append tail return differs from its original transaction cohort");
  if (!bank_count)
    return transactions.return_tails({}, [](std::size_t, std::size_t) noexcept -> AppendTailDestination { return {}; });
  if (count > scratch.max_members() || count > scratch.tail_owners.capacity())
    return Status::FailedPrecondition("paged append tail-owner scratch was not cold prepared");
  for (std::size_t index = 0; index < count; ++index) {
    const auto member = member_at(index);
    if (member.active && member.keep_tail && member.returned_tails.size() != bank_count)
      return Status::InvalidArgument("paged append tail destinations differ from their original bank cohort");
    scratch.tail_owners.emplace_back(member.physical_last_page, index);
  }
  std::sort(scratch.tail_owners.begin(), scratch.tail_owners.end());
  return transactions.return_tails(scratch.tail_owners,
      [&](std::size_t bank, std::size_t index) noexcept -> AppendTailDestination {
        const auto member = member_at(index);
        if (!member.active || !member.keep_tail) return {};
        return {member.initialized_rows, &member.returned_tails[bank]};
      });
}

} // namespace hyle::inference
