#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/lifetime_plan.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

inline constexpr std::uint32_t kNoWorkspaceRequest = UINT32_MAX;
inline constexpr std::uint32_t kNoInPlaceSource = UINT32_MAX;
inline constexpr std::uint8_t kExternalInput = 1u << 0;
inline constexpr std::uint8_t kExternalOutput = 1u << 1;

struct PlanNode final {
  PlanNode(compute::PreparedOperation prepared_value,
           std::span<const std::uint32_t> input_values,
           std::uint32_t output_value) noexcept
      : prepared(std::move(prepared_value)), output(output_value),
        input_count(static_cast<std::uint8_t>(input_values.size())) {
    for (std::size_t index = 0; index < input_values.size(); ++index) {
      inputs[index] = input_values[index];
    }
  }

  PlanNode(const PlanNode &) = delete;
  PlanNode &operator=(const PlanNode &) = delete;
  PlanNode(PlanNode &&) noexcept = default;
  PlanNode &operator=(PlanNode &&) noexcept = default;

  compute::PreparedOperation prepared{};
  std::array<std::uint32_t, 2> inputs{};
  std::uint32_t output = 0;
  std::uint8_t input_count = 0;
};

using PlanNodes = container::SmallVector<PlanNode, 8, mem::AllocatorRef>;
using ValueIds = container::SmallVector<ValueId, 8, mem::AllocatorRef>;
using TensorDescriptions =
    container::SmallVector<compute::TensorDesc, 16, mem::AllocatorRef>;
using ValueOrdinals =
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
using BindingRoles =
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;

struct PlanState final : util::RefCounted<PlanState> {
  PlanState(mem::AllocatorRef allocator_value,
            std::uint64_t generation_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        nodes(allocator_value), input_ids(allocator_value),
        output_ids(allocator_value), descriptions(allocator_value),
        workspace_requests(allocator_value), in_place_sources(allocator_value),
        binding_roles(allocator_value) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~PlanState();
    owner.deallocate(this, sizeof(PlanState), alignof(PlanState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  PlanNodes nodes;
  ValueIds input_ids;
  ValueIds output_ids;
  TensorDescriptions descriptions;
  ValueOrdinals workspace_requests;
  // For an externally bound output, identifies the sole external input that
  // may use the exact same byte range. All other output overlap is rejected.
  ValueOrdinals in_place_sources;
  BindingRoles binding_roles;
  mem::LifetimePlan workspace_plan{};
  std::size_t workspace_bytes = 0;
  std::size_t workspace_backing_count = 0;
};

inline void retain(PlanState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(PlanState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

using ReadableBindings =
    container::SmallVector<compute::ConstTensorView, 16, mem::AllocatorRef>;
using WritableBindings =
    container::SmallVector<compute::TensorView, 16, mem::AllocatorRef>;

struct InvocationState final {
  InvocationState(PlanState *plan_value,
                  mem::PlannedBuffers workspace_value) noexcept
      : allocator(plan_value->allocator), plan(plan_value), readable(allocator),
        writable(allocator), bound(allocator),
        workspace(std::move(workspace_value)) {
    retain(plan);
  }

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    PlanState *retained_plan = plan;
    this->~InvocationState();
    release(retained_plan);
    owner.deallocate(this, sizeof(InvocationState), alignof(InvocationState));
  }

  mem::AllocatorRef allocator{};
  PlanState *plan = nullptr;
  ReadableBindings readable;
  WritableBindings writable;
  BindingRoles bound;
  mem::PlannedBuffers workspace{};
};

struct ExecutableAccess final {
  static Executable adopt(PlanState *state) noexcept {
    return Executable(state);
  }
};

struct InvocationAccess final {
  static Invocation adopt(InvocationState *state) noexcept {
    return Invocation(state);
  }
};

inline void destroy(InvocationState *state) noexcept {
  if (state != nullptr)
    state->destroy();
}

} // namespace hyle::graph::detail
