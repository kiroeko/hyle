#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "hyle/core/assert.h"
#include "readback.h"

namespace hyle::inference {

template<class Scope> class ProgramCarrier;

// Cold IO for one concrete program. Fixed views borrow their original cold
// Tensor owners; dynamic bindings live only in Graph. Readbacks are the same
// caller-owned carriers as PreparedReadback, not copies of their leases.
// This inventory owns no Invocation, Completion, transaction or sampling rule.
class ProgramIo final {
  template<class Tag> class Handle final {
   public:
    Handle() = default;
    explicit operator bool() const noexcept { return owner_ != nullptr; }
   private:
    friend class ProgramIo;
    Handle(const ProgramIo *owner, std::size_t index) noexcept : owner_(owner), index_(index) {}
    const ProgramIo *owner_ = nullptr;
    std::size_t index_ = 0;
  };
  struct InputTag;
  struct ReadbackTag;
  struct OutputTag;
 public:
  using Input = Handle<InputTag>;
  using Readback = Handle<ReadbackTag>;
  using Output = Handle<OutputTag>;

  ProgramIo() = default;
  ProgramIo(const ProgramIo &) = delete;
  ProgramIo &operator=(const ProgramIo &) = delete;
  ProgramIo(ProgramIo &&) = delete;
  ProgramIo &operator=(ProgramIo &&) = delete;

  // Capacities are cold bounds. No registration or vector growth is allowed
  // after seal(); all runtime lookups use handles from this stable inventory.
  Status prepare(const graph::CompiledGraph &compiled, std::size_t fixed_bindings,
      std::size_t inputs, std::size_t readbacks, std::size_t outputs) noexcept {
    if (compiled_) return Status::FailedPrecondition("program IO is already prepared");
    HYLE_RETURN_IF_ERROR(detail::check_output_route(compiled));
    if (fixed_bindings > (std::numeric_limits<std::size_t>::max)() - readbacks ||
        inputs > (std::numeric_limits<std::size_t>::max)() - outputs)
      return Status::OutOfRange("program IO capacity overflows");
    try {
      fixed_.reserve(fixed_bindings); inputs_.reserve(inputs); readbacks_.reserve(readbacks);
      outputs_.reserve(outputs); installation_.reserve(fixed_bindings + readbacks);
      withdrawals_.reserve(inputs + outputs);
    } catch (...) { return Status::ResourceExhausted("program IO cold metadata"); }
    compiled_ = &compiled;
    return Status::Ok();
  }

  Status add_fixed(graph::ValueId value, compute::ConstTensorView tensor) noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    if (!value || !tensor.valid()) return Status::InvalidArgument("program fixed input is invalid");
    if (fixed_.size() == fixed_.capacity() || installation_.size() == installation_.capacity())
      return Status::ResourceExhausted("program fixed input exceeds its cold capacity");
    installation_.push_back({false, fixed_.size()});
    fixed_.push_back({value, tensor});
    return Status::Ok();
  }

  // Borrow the original caller-owned mutable Tensor view. It is an output
  // binding, not a read-only input or another readback/pool allocation.
  Status add_fixed_output(graph::ValueId value, compute::TensorView tensor) noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    if (!value || !tensor.valid()) return Status::InvalidArgument("program fixed output is invalid");
    if (fixed_.size() == fixed_.capacity() || installation_.size() == installation_.capacity())
      return Status::ResourceExhausted("program fixed output exceeds its cold capacity");
    installation_.push_back({false, fixed_.size()});
    fixed_.push_back({value, tensor});
    return Status::Ok();
  }

  Result<Input> add_input(graph::ValueId value) noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    if (!value) return std::unexpected(Status::InvalidArgument("program dynamic input is invalid"));
    if (inputs_.size() == inputs_.capacity() || withdrawals_.size() == withdrawals_.capacity())
      return std::unexpected(Status::ResourceExhausted("program input exceeds its cold capacity"));
    const Input handle(this, inputs_.size());
    inputs_.push_back(value); withdrawals_.push_back(value);
    return handle;
  }

  Result<Readback> add_readback(graph::ValueId value, const Backend &backend,
      const compute::ResourceBudget &budget = {}) noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    if (readbacks_.size() == readbacks_.capacity() || installation_.size() == installation_.capacity())
      return std::unexpected(Status::ResourceExhausted("program readback exceeds its cold capacity"));
    HYLE_ASSIGN_OR_RETURN(auto prepared, PreparedReadback::prepare(*compiled_, value, backend, budget));
    const Readback handle(this, readbacks_.size());
    installation_.push_back({true, readbacks_.size()});
    readbacks_.push_back(std::move(prepared));
    return handle;
  }

  Result<Output> add_output(graph::ValueId value) noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    if (outputs_.size() == outputs_.capacity() || withdrawals_.size() == withdrawals_.capacity())
      return std::unexpected(Status::ResourceExhausted("program retained output exceeds its cold capacity"));
    HYLE_ASSIGN_OR_RETURN(auto facts, prepare_output(*compiled_, value));
    HYLE_ASSIGN_OR_RETURN(auto bytes, facts.description.required_bytes());
    const Output handle(this, outputs_.size());
    if (facts.caller_owned()) withdrawals_.push_back(value);
    outputs_.push_back({std::move(facts), bytes});
    return handle;
  }

  Status seal() noexcept {
    HYLE_RETURN_IF_ERROR(registering());
    sealed_ = true;
    return Status::Ok();
  }

  // Installation borrows this exact immutable program inventory. A sealed IO
  // for a different child must not consume fixed writers on a new candidate.
  Status check_program(const graph::CompiledGraph &compiled) const noexcept {
    if (!sealed_) return Status::FailedPrecondition("program IO is not sealed for initialization");
    if (compiled_ != &compiled)
      return Status::FailedPrecondition("program IO belongs to a different prepared program");
    return Status::Ok();
  }

  // Installation order is the original cold registration order. First check
  // this inventory's exact program, then install on its unpublished original
  // Invocation (a carrier candidate or the caller's synchronous candidate).
  // If a later bind fails, destroying that candidate returns earlier writers
  // to their original pools; PreparedReadback::bind can allocate them on retry.
  Status bind_fixed(graph::Invocation &run) noexcept {
    if (!sealed_) return Status::FailedPrecondition("program IO is not sealed");
    for (const auto action : installation_) {
      if (action.readback) HYLE_RETURN_IF_ERROR(readbacks_[action.index].bind(run));
      else {
        const auto &binding = fixed_[action.index];
        if (const auto *input = std::get_if<compute::ConstTensorView>(&binding.tensor))
          HYLE_RETURN_IF_ERROR(run.bind(binding.value, *input));
        else HYLE_RETURN_IF_ERROR(run.bind(binding.value, std::get<compute::TensorView>(binding.tensor)));
      }
    }
    return Status::Ok();
  }

  Status bind(graph::Invocation &run, Input input, const compute::TensorDesc &description,
      const compute::ReadLease &lease) const noexcept {
    HYLE_RETURN_IF_ERROR(check(input, inputs_.size()));
    return run.bind(inputs_[input.index_], description, lease);
  }
  Status bind(graph::Invocation &run, Input input, compute::ConstTensorView tensor) const noexcept {
    HYLE_RETURN_IF_ERROR(check(input, inputs_.size()));
    return run.bind(inputs_[input.index_], tensor);
  }

  // Caller ownership is a prepared report fact, never inferred from backend.
  // The supplied pool remains with the original program and its old budget.
  Status bind_output(graph::Invocation &run, Output output, compute::LeasePool &pool) const noexcept {
    HYLE_RETURN_IF_ERROR(check(output, outputs_.size()));
    const auto &item = outputs_[output.index_];
    if (!item.facts.caller_owned()) return Status::Ok();
    HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(item.bytes));
    return run.bind(item.facts.value, item.facts.description, std::move(writer));
  }

  Status read(const Backend &backend, graph::Invocation &run, Readback output,
      std::span<std::byte> bytes) const noexcept {
    HYLE_RETURN_IF_ERROR(check(output, readbacks_.size()));
    return readbacks_[output.index_].read(backend, run, bytes);
  }
  Result<std::uint8_t> predicate(const Backend &backend, graph::Invocation &run,
      Readback output) const noexcept {
    HYLE_RETURN_IF_ERROR(check(output, readbacks_.size()));
    return readbacks_[output.index_].predicate(backend, run);
  }

  // Borrow the sole original owner for its facts, Source pool proof or explicit
  // retired replacement. The handle grants no release permission beyond Graph.
  Result<PreparedReadback *> readback(Readback output) noexcept {
    HYLE_RETURN_IF_ERROR(check(output, readbacks_.size()));
    return &readbacks_[output.index_];
  }
  Result<const PreparedReadback *> readback(Readback output) const noexcept {
    HYLE_RETURN_IF_ERROR(check(output, readbacks_.size()));
    return &readbacks_[output.index_];
  }
  Result<compute::ReadLease> retain(graph::Invocation &run, Output output,
      const compute::ResourceBudget &budget = {}) const noexcept {
    HYLE_RETURN_IF_ERROR(check(output, outputs_.size()));
    return retain_output(run, outputs_[output.index_].facts, budget);
  }

  // Like a span subscript: the handle must come from this same cold inventory.
  const PreparedOutputFacts &output_facts(Output output) const noexcept {
    HYLE_FATAL_ASSERT(output.owner_ == this && output.index_ < outputs_.size());
    return outputs_[output.index_].facts;
  }

  // Call only after output observation and any original transaction extraction
  // which must precede withdrawal. The carrier and Graph remain authoritative;
  // this list is immutable schema, not a mirror of currently bound permission.
  template<class Scope> Status release_inputs(ProgramCarrier<Scope> &carrier) const noexcept {
    if (!sealed_) return Status::FailedPrecondition("program IO is not sealed");
    return carrier.release_inputs(withdrawals_);
  }

 private:
  Status registering() const noexcept {
    return compiled_ && !sealed_ ? Status::Ok()
        : Status::FailedPrecondition("program IO is not open for cold registration");
  }
  template<class Tag> Status check(Handle<Tag> handle, std::size_t size) const noexcept {
    if (!sealed_ || handle.owner_ != this || handle.index_ >= size)
      return Status::InvalidArgument("program IO handle does not belong to this sealed inventory");
    return Status::Ok();
  }
  struct Fixed final {
    graph::ValueId value;
    std::variant<compute::ConstTensorView, compute::TensorView> tensor;
  };
  struct Installation final { bool readback; std::size_t index; };
  struct RetainedOutput final { PreparedOutputFacts facts; std::size_t bytes; };
  const graph::CompiledGraph *compiled_ = nullptr;
  std::vector<Fixed> fixed_;
  std::vector<graph::ValueId> inputs_, withdrawals_;
  std::vector<PreparedReadback> readbacks_;
  std::vector<RetainedOutput> outputs_;
  std::vector<Installation> installation_;
  bool sealed_ = false;
};

} // namespace hyle::inference
