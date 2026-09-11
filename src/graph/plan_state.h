#pragma once

#include <array>
#include <utility>

#include "hyle/graph/executable.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

struct PlanState final : util::RefCounted<PlanState> {
  PlanState(mem::AllocatorRef allocator_value,
            compute::PreparedOperation prepared_value,
            std::span<const ValueId> input_ids_value, ValueId output_id_value,
            std::span<const compute::TensorDesc> input_descs_value,
            compute::TensorDesc output_desc_value) noexcept
      : allocator(allocator_value), prepared(std::move(prepared_value)),
        output_id(output_id_value), output_desc(output_desc_value),
        input_count(static_cast<std::uint8_t>(input_ids_value.size())) {
    for (std::size_t index = 0; index < input_ids_value.size(); ++index) {
      input_ids[index] = input_ids_value[index];
      input_descs[index] = input_descs_value[index];
    }
  }

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~PlanState();
    owner.deallocate(this, sizeof(PlanState), alignof(PlanState));
  }

  mem::AllocatorRef allocator{};
  compute::PreparedOperation prepared{};
  std::array<ValueId, 2> input_ids{};
  ValueId output_id{};
  std::array<compute::TensorDesc, 2> input_descs{};
  compute::TensorDesc output_desc{};
  std::uint8_t input_count = 0;
};

struct ExecutableAccess final {
  static Executable adopt(PlanState* state) noexcept { return Executable(state); }
};

inline void retain(PlanState* state) noexcept {
  if (state != nullptr) state->incref();
}

inline void release(PlanState* state) noexcept {
  if (state != nullptr) state->decref();
}

}  // namespace hyle::graph::detail
