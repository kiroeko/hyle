#pragma once

#include <cstddef>
#include <span>

#include "hyle/compute/resource_leases.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {
struct LeaseSubmissionState;
struct LeaseSubmissionOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  compute::ResourceBudget resource_budget{};
  std::size_t max_page_reads = 0;
};

// Cold permission metadata for native composition owners. The original owner
// executes its own program and supplies only lifetime/terminal callbacks here.
class LeaseSubmission final {
public:
  using ReferenceFn = void (*)(void *) noexcept;
  using TerminalFn = void (*)(void *, Status, bool) noexcept;
  LeaseSubmission() noexcept = default;
  LeaseSubmission(const LeaseSubmission &) = delete;
  LeaseSubmission &operator=(const LeaseSubmission &) = delete;
  HYLE_GRAPH_API LeaseSubmission(LeaseSubmission &&) noexcept;
  HYLE_GRAPH_API LeaseSubmission &operator=(LeaseSubmission &&) noexcept;
  HYLE_GRAPH_API ~LeaseSubmission();
  HYLE_GRAPH_API static Result<LeaseSubmission> create(std::size_t inputs, std::size_t prefixes,
      LeaseSubmissionOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API void bind(std::size_t input, compute::ReadLease source) noexcept;
  // Trusted composition owners must qualify actual reads against their layout.
  // Copies exact guards cold; never creates permission for an addressable arena.
  // A rejected bind preserves the old input. Control is externally serialized;
  // accepted prefixes must actually retire before rebind/reset/destruction.
  HYLE_GRAPH_API Status bind_pages(std::size_t input, std::span<const compute::ReadLease>) noexcept;
  HYLE_GRAPH_API void clear(std::size_t input) noexcept;
  HYLE_GRAPH_API void reset() noexcept;
  HYLE_GRAPH_API Status begin(void *owner, ReferenceFn retain, ReferenceFn release, TerminalFn terminal) noexcept;
  HYLE_GRAPH_API Status protect(const compute::Completion &completion) noexcept;
  HYLE_GRAPH_API void finish(Status terminal, bool accepted) noexcept;
  HYLE_GRAPH_API Status observed_error() const noexcept;
private:
  explicit LeaseSubmission(LeaseSubmissionState *state) noexcept : state_(state) {}
  LeaseSubmissionState *state_ = nullptr;
};
} // namespace hyle::graph::detail
