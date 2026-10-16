#pragma once

#include <memory>

#include "sampling_fixture.h"
#include <hyle/compute/prepared_operation.h>
#include <hyle/mem/buffer.h>

#if defined(HYLE_G08_CUDA)
#include <hyle/compute/backends/cuda.h>
#if defined(HYLE_G08_GRAPH)
#include <hyle/graph/cuda.h>
#else
#include <hyle/compute/cuda.h>
#endif
#else
#if defined(HYLE_G08_GRAPH)
#include <hyle/graph/cpu.h>
#else
#include <hyle/compute/cpu.h>
#endif
#endif

namespace sampling_test {
namespace m = hyle::mem;
#if defined(HYLE_G08_GRAPH)
namespace g = hyle::graph;
#endif

class CommonSession final {
public:
  CommonSession(c::Operation operation, std::vector<c::TensorDesc> inputs)
      : descriptions_(std::move(inputs)), outputs_(outputs(operation, descriptions_)) {
#if defined(HYLE_G08_CUDA)
#if defined(HYLE_G08_GRAPH)
    context_ = take(g::cuda::create_context());
#else
    context_ = take(c::cuda::create_context());
#endif
    device_memory_ = take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice));
    memory_ = device_memory_.ref();
#else
#if defined(HYLE_G08_GRAPH)
    context_ = take(g::cpu::create_context());
#else
    context_ = take(c::cpu::create_context());
#endif
    memory_ = m::system_resource();
#endif
    for (auto &queue : queues_) queue = take(context_.create_queue());
    for (const auto &description : descriptions_) {
      inputs_.push_back(take(m::Buffer::create(memory_, take(description.required_bytes()), 256)));
      views_.push_back(take(c::ConstTensorView::create(description, inputs_.back().view())));
    }
#if defined(HYLE_G08_GRAPH)
    auto builder = take(g::GraphBuilder::create());
    std::vector<g::ValueId> produced;
    for (const auto &description : descriptions_) {
      const auto input = take(builder.input(description));
      input_ids_.push_back(input);
      produced.push_back(take(builder.add(c::Copy{}, input)));
    }
    const auto node = take(builder.add_node(operation, produced));
    for (std::size_t port = 0; port < outputs_.size(); ++port) {
      const auto value = take(builder.result(node, port));
      const auto copy = take(builder.add(c::Copy{}, value));
      output_ids_.push_back(copy);
      require(builder.publish(copy));
    }
    auto logical = take(std::move(builder).finish());
#if defined(HYLE_G08_CUDA)
    auto compiled = take(g::cuda::compile(logical, context_, memory_));
#else
    auto compiled = take(g::compile(logical, context_));
#endif
    program_ = std::move(compiled.executable);
#else
    program_ = take(c::prepare(context_, operation, descriptions_, outputs_, memory_));
#endif
#if defined(HYLE_G08_GRAPH)
    for (auto &slot : slots_) slot = take(program_.create_invocation());
#else
    const auto requirement = program_.workspace_requirement();
    workspace_size_ = requirement.size;
    workspace_guard_ = std::max(std::size_t{256}, requirement.alignment);
    for (std::size_t bank = 0; bank < slots_.size(); ++bank) {
      if (requirement.size != 0) {
        workspaces_[bank] = take(m::Buffer::create(memory_, requirement.size + workspace_guard_*2, workspace_guard_));
        require(workspaces_[bank].view().fill(std::byte{0xa5}));
        const auto range = take(workspaces_[bank].view().subview(workspace_guard_, requirement.size));
        slots_[bank] = take(program_.create_invocation(range));
      } else slots_[bank] = take(program_.create_invocation());
    }
#endif
  }

  hyle::Result<Output> run(const Payloads &payloads, int bank) {
    check(payloads.size() == inputs_.size(), "consumer input arity differs");
    for (std::size_t i = 0; i < inputs_.size(); ++i)
      HYLE_RETURN_IF_ERROR(inputs_[i].view().write(payloads[i]));
    auto buffers = std::make_shared<std::vector<m::Buffer>>();
    std::vector<c::TensorView> destination;
    for (const auto &description : outputs_) {
      HYLE_ASSIGN_OR_RETURN(auto buffer, m::Buffer::create(memory_, take(description.required_bytes()), 256));
      buffers->push_back(std::move(buffer));
      HYLE_ASSIGN_OR_RETURN(auto view, c::TensorView::create(description, buffers->back().view()));
      destination.push_back(view);
    }
    auto &slot = slots_[std::size_t(bank)];
    auto &queue = queues_[std::size_t(bank)];
#if defined(HYLE_G08_GRAPH)
    for (std::size_t i = 0; i < views_.size(); ++i)
      HYLE_RETURN_IF_ERROR(slot.bind(input_ids_[i], views_[i]));
    for (std::size_t i = 0; i < destination.size(); ++i)
      HYLE_RETURN_IF_ERROR(slot.bind(output_ids_[i], destination[i]));
    const std::array binding{g::QueueBinding{.lane=g::ExecutionLaneId{0}, .queue=&queue}};
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(binding,
        {.mode=g::ReplayMode::kCommands, .host_wait=g::HostWaitPolicy::kReject}));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit(queue));
#else
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit(queue, views_, destination));
#endif
    HYLE_RETURN_IF_ERROR(work.wait());
#if !defined(HYLE_G08_GRAPH)
    if (workspace_size_ != 0) {
      for (const auto offset : {std::size_t{0},workspace_guard_+workspace_size_}) {
        Bytes guard(workspace_guard_);
        HYLE_RETURN_IF_ERROR(take(workspaces_[std::size_t(bank)].view().subview(offset,workspace_guard_)).read(guard));
        check(std::ranges::all_of(guard,[](std::byte value){return value==std::byte{0xa5};}),
              "numeric operation wrote outside its declared caller workspace");
      }
    }
#endif
    auto reader = [buffers] {
      Payloads result;
      for (auto &buffer : *buffers) {
        Bytes bytes(buffer.size());
        require(buffer.view().read(bytes));
        result.push_back(std::move(bytes));
      }
      return result;
    };
    return Output{reader(), std::move(reader)};
  }
  void close() {
    for (auto &slot : slots_) slot = {};
    program_ = {};
    for (auto &queue : queues_) require(queue.close());
    require(context_.close());
  }
private:
  c::Context context_;
#if defined(HYLE_G08_CUDA)
  c::cuda::MemoryResource device_memory_;
#endif
  m::MemoryResourceRef memory_;
  // Two replay slots plus a held-result slot. Vulkan explicitly prevents
  // overwriting retained tensors; the shared fixture observes that contract.
  std::array<c::Queue, 3> queues_;
  std::vector<c::TensorDesc> descriptions_, outputs_;
  std::vector<m::Buffer> inputs_;
  std::vector<c::ConstTensorView> views_;
#if defined(HYLE_G08_GRAPH)
  g::Executable program_;
  std::array<g::Invocation, 3> slots_;
  std::vector<g::ValueId> input_ids_, output_ids_;
#else
  std::array<m::Buffer, 3> workspaces_;
  std::size_t workspace_size_ = 0, workspace_guard_ = 256;
  c::PreparedOperation program_;
  std::array<c::OperationInvocation, 3> slots_;
#endif
};
} // namespace sampling_test
