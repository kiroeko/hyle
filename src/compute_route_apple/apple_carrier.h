#pragma once

#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/generated_result.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace hyle::compute::apple::detail {

// Descriptor-only cold selection shared by Direct and Graph. NativeProgram's
// own validator owns its shape/encoding/output domain; actual pipeline/resource
// preparation remains in ContextAccess::compile. There is no model-op roster.
HYLE_COMPUTE_ROUTE_APPLE_API Result<RegionKind>
analyze_native_carrier(const Operation &, std::span<const TensorDesc>) noexcept;

// Retains the actual producer Job, never its Completion or result storage.
// Cold binding checks the source DAG before taking these references into a
// consumer, so native slots cannot retain one another in a cycle.
class InputSource final {
public:
  InputSource() noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API InputSource(const InputSource &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API InputSource &operator=(const InputSource &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API InputSource(InputSource &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API InputSource &operator=(InputSource &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~InputSource();
  bool valid() const noexcept { return state_ != nullptr; }
  bool same_output(const InputSource &other) const noexcept {
    return state_ && state_ == other.state_ && port_ == other.port_;
  }
  HYLE_COMPUTE_ROUTE_APPLE_API void reset() noexcept;

private:
  friend struct InvocationAccess;
  friend struct InvocationState;
  InputSource(InvocationState *, std::size_t) noexcept;
  InvocationState *state_ = nullptr;
  std::size_t port_ = 0;
};

struct InputBinding final {
  // Exactly one member is populated. Ready values remain immutable native
  // owners; sources resolve only inside the original FIFO Job after success.
  std::optional<mlx::core::array> value;
  InputSource source;
};

struct InvocationAccess final {
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<InputSource>
  source(const Invocation &, std::size_t output_port) noexcept;
  // Original external FIFO consumers use the same per-producer epoch and
  // reader monitor as native input edges. Only accepted episodes consume an
  // epoch; every pin must be released after actual retirement or failed admit.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<std::uint64_t>
  source_epoch(const InputSource &, const std::shared_ptr<Runtime> &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<std::uint64_t>
  pin_source(const InputSource &, const std::shared_ptr<Runtime> &,
             std::uint64_t consumed_epoch) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array>
  source_output(const InputSource &, const std::shared_ptr<Runtime> &,
                std::uint64_t expected_epoch) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static void release_source(const InputSource &) noexcept;
  // Sources must submit after this cold baseline and before the consumer.
  // The consumer snapshots each source's own epoch, consumes it once, and
  // pins that native source slot through its checked drain. Failed admission
  // drops every pin; the producer cannot rebind/clear/replay while pinned.
  HYLE_COMPUTE_ROUTE_APPLE_API static Status
  prepare_inputs(Invocation &, std::span<const InputBinding>) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status release_inputs(Invocation &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status can_detach_inputs(
      const Invocation &, const compute::detail::ResultStorageOwner &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static void detach_inputs(Invocation &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status reuse_gate(Invocation &) noexcept;
  // The original FIFO Job owns the supplied uses and borrows paired preflight
  // metadata through actual retirement; no new queue or submission path.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<OwnedResult<Results>>
  submit_ordered(Invocation &, LeaseUseBatch,
      Status (*preflight)(const void *) noexcept, const void *source) noexcept;
  // Readonly, successfully completed output only; close does not revoke it.
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<mlx::core::array>
  output(const Invocation &, std::size_t output_port) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static std::shared_ptr<Runtime>
  runtime(const Invocation &) noexcept;
};

} // namespace hyle::compute::apple::detail
