#pragma once

#include "hyle/compute/paged_frame.h"
#include <algorithm>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace qwen35::probe {
// Diagnostics compare each physical page once, in ascending ID order. A
// repeated logical reference uses its largest initialized prefix; page_prefix
// keeps the original guard alive without exporting unused capacity tails.
inline hyle::Result<std::vector<hyle::compute::ReadLease>> unique_page_prefixes(
    const hyle::compute::PagedArenaFootprint &footprint) noexcept {
  if (!footprint.valid())
    return std::unexpected(hyle::Status::FailedPrecondition("diagnostic page footprint is empty or stale"));
  try {
    const auto host = footprint.layout().host();
    std::vector<std::int32_t> pages(host.page_indices.begin(), host.page_indices.end());
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    std::vector<hyle::compute::ReadLease> guards;
    guards.reserve(pages.size());
    for (const auto page : pages) {
      HYLE_ASSIGN_OR_RETURN(auto guard, footprint.page_prefix(static_cast<std::uint32_t>(page)));
      guards.push_back(std::move(guard));
    }
    return guards;
  } catch (const std::bad_alloc &) {
    return std::unexpected(hyle::Status::ResourceExhausted("diagnostic page prefix metadata allocation failed"));
  }
}
} // namespace qwen35::probe
