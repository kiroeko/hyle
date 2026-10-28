#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "prepared_output.h"

namespace hyle::inference {

// One immutable, concrete route. Executable and report are always selected
// together from the same finite family; output facts retain their published
// order. These are cold Graph handles and metadata, not another native owner.
class PreparedProgram final {
 public:
  graph::CompiledGraph compiled;

  const PreparedOutputFacts &output(graph::ValueId value) const {
    if (!value || value.ordinal() >= output_indices_.size())
      throw std::runtime_error("value is outside the prepared child output index");
    const auto index = output_indices_[value.ordinal()];
    if (index >= outputs_.size() || outputs_[index].value != value)
      throw std::runtime_error("value is not a prepared output of this exact child");
    return outputs_[index];
  }

  std::span<const PreparedOutputFacts> outputs() const noexcept { return outputs_; }

 private:
  friend class PreparedProgramFamily;
  std::vector<PreparedOutputFacts> outputs_;
  std::vector<std::uint32_t> output_indices_;
};

// Cold finite-exact inventory shared by execution carriers. The inventory and
// each selected child have stable addresses and must outlive all borrowed
// ProgramCarrier/ProgramIo references. Invocation, IO, transactions, admission,
// retirement and model-specific shape metadata remain with their callers.
class PreparedProgramFamily final {
 public:
  PreparedProgramFamily() = default;
  PreparedProgramFamily(const PreparedProgramFamily &) = delete;
  PreparedProgramFamily &operator=(const PreparedProgramFamily &) = delete;
  PreparedProgramFamily(PreparedProgramFamily &&) = delete;
  PreparedProgramFamily &operator=(PreparedProgramFamily &&) = delete;

  // A failed preparation publishes nothing. Every child is complete before
  // any carrier can borrow it; retry never encounters a partial inventory.
  Status prepare(graph::CompiledGraph compiled, std::size_t value_count,
      std::size_t expected_variants) noexcept {
    if (parent_.executable || parent_.report)
      return Status::FailedPrecondition("program family is already prepared");
    if (!compiled.executable || !compiled.report || !expected_variants ||
        compiled.executable.exact_variant_count() != expected_variants ||
        compiled.report.exact_variant_count() != expected_variants)
      return Status::InvalidArgument("program family exact variant counts differ");
    try {
      std::vector<std::unique_ptr<PreparedProgram>> children;
      children.reserve(expected_variants);
      for (std::size_t index = 0; index < expected_variants; ++index) {
        auto child = std::make_unique<PreparedProgram>();
        HYLE_ASSIGN_OR_RETURN(child->compiled.executable, compiled.executable.exact_variant(index));
        HYLE_ASSIGN_OR_RETURN(child->compiled.report, compiled.report.exact_variant(index));
        child->outputs_.resize(child->compiled.executable.output_count());
        child->output_indices_.resize(value_count);
        HYLE_RETURN_IF_ERROR(prepare_outputs(child->compiled, child->outputs_, child->output_indices_));
        children.push_back(std::move(child));
      }
      parent_ = std::move(compiled);
      children_ = std::move(children);
    } catch (...) { return Status::ResourceExhausted("program family cold metadata"); }
    return Status::Ok();
  }

  std::size_t size() const noexcept { return children_.size(); }
  const graph::CompiledGraph &parent() const noexcept { return parent_; }
  const PreparedProgram &child(std::size_t index) const { return *children_.at(index); }

 private:
  graph::CompiledGraph parent_;
  std::vector<std::unique_ptr<PreparedProgram>> children_;
};

} // namespace hyle::inference
