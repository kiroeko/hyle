#pragma once

#include "hyle/graph/cpu_vulkan.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/mem/std_allocator.h"

#include <optional>

namespace hyle::graph::cpu_vulkan::detail {
template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;

struct Value final {
  ValueId id;
  compute::TensorDesc description, carrier;
  compute::ResourceBudget::Reservation constant_charge;
  mem::Buffer constant_storage;
  compute::ConstTensorView constant;
  ConstantPolicy constant_policy = ConstantPolicy::kBorrow;
  std::optional<compute::vulkan::Tensor> native_constant;
  graph::Executable repack;
  ValueId repack_input, repack_output;
  bool cpu_use = false, vulkan_use = false, published = false, fragment_output = false;
  ExecutionTargetId producer;
};

struct Fragment final {
  explicit Fragment(mem::AllocatorRef allocator)
      : inputs(mem::StdAllocator<std::size_t>(allocator)),
        outputs(mem::StdAllocator<std::size_t>(allocator)),
        cpu_inputs(mem::StdAllocator<ValueId>(allocator)),
        cpu_outputs(mem::StdAllocator<ValueId>(allocator)) {}
  ExecutionTargetId target;
  Vector<std::size_t> inputs, outputs;
  Vector<ValueId> cpu_inputs, cpu_outputs;
  graph::Executable cpu;
  compute::vulkan::Program vulkan;
  compute::ResourceBudget::Reservation private_storage;
};

struct Plan final {
  explicit Plan(mem::AllocatorRef owner)
      : allocator(owner), values(mem::StdAllocator<Value>(owner)),
        inputs(mem::StdAllocator<std::size_t>(owner)),
        fragments(mem::StdAllocator<Fragment>(owner)) {}
  mem::AllocatorRef allocator;
  compute::ResourceBudget budget;
  compute::WorkloadContext workload;
  Environment environment;
  compute::detail::ContextRef cpu_context;
  Vector<Value> values;
  Vector<std::size_t> inputs;
  Vector<Fragment> fragments;
  std::size_t host_workspace_bytes = 0;
};

} // namespace hyle::graph::cpu_vulkan::detail
