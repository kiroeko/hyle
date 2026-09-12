#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"
#include "hyle/graph/executable.h"

namespace hyle::graph {

// Versioned, provider-local Graph region contract. Compute only sees this as a
// named extension table and therefore never depends on Graph types.
inline constexpr compute::ProviderExtensionId kRegionProviderExtensionId{
    0x48594c4547524150ULL, 0x48524547494f4e31ULL, 2};

// Half-open contiguous range in Graph's canonical topological node order.
// Restricting the first contract to intervals makes deterministic weighted
// selection O(Q log Q), while still covering the fused regions exposed by the
// initial inference providers. A future non-contiguous contract requires a
// separate version rather than silently changing these semantics.
struct RegionNodeRange final {
  std::uint32_t begin = 0;
  std::uint32_t count = 0;

  constexpr std::uint64_t end() const noexcept {
    return static_cast<std::uint64_t>(begin) + count;
  }
};

// Comparable cold-path facts for the exact static graph specialization. Net
// savings is relative to the provider's natural single-op fallback sequence
// and drives conservative selection. The remaining fields make hidden costs
// visible to policy/reporting. An absent optional means that the provider
// cannot report the estimate; an engaged zero means exact none. A provider
// that cannot defend positive savings should not emit the proposal under the
// default conservative policy.
struct RegionCostFacts final {
  std::uint64_t estimated_net_savings_ns = 0;
  std::optional<std::uint64_t> estimated_compile_ns{};
  std::optional<std::size_t> persistent_bytes{};
  std::optional<std::size_t> workspace_bytes{};
  std::size_t boundary_copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Number of backend execution episodes: one synchronous CPU executor
  // fan-out or one device command launch counts as one. Workers, tasks,
  // vector chunks and leaf calls do not count separately.
  std::uint32_t dispatch_count = 1;
  bool blocks_host = false;
  bool uses_private_threads = false;
};

struct RegionProposal final {
  RegionNodeRange nodes{};
  RegionCostFacts cost{};
  // Provider-defined cold-path discriminator. It may select a fusion/compile
  // route but is never interpreted or searched on the steady path.
  std::uint64_t route = 0;
};

// Fixed-capacity proposal sink owned by Graph compile. Providers cannot grow
// it or hide allocation. Exhaustion is an explicit compile failure so a
// proposal budget never changes the chosen plan silently.
class RegionProposalWriter final {
public:
  explicit RegionProposalWriter(std::span<RegionProposal> storage) noexcept
      : storage_(storage) {}

  Status add(RegionProposal proposal) noexcept {
    if (proposal.nodes.count == 0 ||
        proposal.nodes.end() > (std::numeric_limits<std::uint32_t>::max)()) {
      return Status::InvalidArgument("graph region proposal range is invalid");
    }
    if (size_ == storage_.size()) {
      return Status::ResourceExhausted(
          "graph region proposal budget is exhausted");
    }
    storage_[size_++] = proposal;
    return Status::Ok();
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return storage_.size(); }
  std::span<const RegionProposal> proposals() const noexcept {
    return storage_.first(size_);
  }

private:
  std::span<RegionProposal> storage_{};
  std::size_t size_ = 0;
};

struct RegionCompileRequest final {
  const Graph *graph = nullptr;
  RegionNodeRange nodes{};
  std::span<const ValueId> inputs{};
  std::span<const ValueId> outputs{};
  std::span<const ConstantBinding> constants{};
  std::uint64_t route = 0;
};

struct RegionArtifactProperties final {
  compute::WorkspaceRequirement workspace{};
  // nullopt means that opaque provider-owned code/metadata bytes are not
  // reportable. An engaged zero means that the artifact has none.
  std::optional<std::size_t> persistent_bytes{};
  // Packing is an additional provider-owned representation. Graph v3 keeps
  // every source ConstantBinding borrowed because aggregate counts do not
  // identify which source could be released.
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  // Exact backend execution episodes per submit; see RegionCostFacts.
  std::uint32_t dispatch_count = 1;
  bool blocks_host = false;
  bool uses_private_threads = false;
  bool supports_concurrent_invocations = true;
};

struct ProviderRegionPreparedVTable final {
  void (*destroy_prepared)(void *state,
                           mem::AllocatorRef allocator) noexcept = nullptr;
  compute::AliasSupport (*alias_support)(
      void *prepared, std::size_t output_index,
      std::size_t input_index) noexcept = nullptr;
  Status (*create_invocation)(void *prepared, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept = nullptr;
  void (*destroy_invocation)(void *invocation,
                             mem::AllocatorRef allocator) noexcept = nullptr;
  // Success returns a Completion for the complete ordered Queue prefix through
  // this region. Dropping an earlier Completion must not cancel that prefix or
  // prevent a later prefix Completion from reporting an earlier backend
  // failure. A synchronous Result error means this region accepted no work.
  Result<compute::Completion> (*submit)(
      void *prepared, void *invocation, compute::Queue &queue,
      std::span<const compute::ConstTensorView> inputs,
      std::span<const compute::TensorView> outputs) noexcept = nullptr;
};

struct ProviderRegionPrepared final {
  const ProviderRegionPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  RegionArtifactProperties properties{};
};

// A provider proposes candidate intervals, then compiles only intervals chosen
// by Graph. Both callbacks are cold-path and run under the owning Context's
// admission gate. Returned artifacts must use the supplied allocator and must
// not retain request spans or Graph references unless they create an explicit
// owner of their own.
struct RegionProviderVTable final {
  Status (*propose)(void *provider_context, const Graph &graph,
                    RegionProposalWriter &writer) noexcept = nullptr;
  Status (*compile)(void *provider_context, const RegionCompileRequest &request,
                    mem::AllocatorRef allocator,
                    ProviderRegionPrepared *prepared) noexcept = nullptr;
};

} // namespace hyle::graph
