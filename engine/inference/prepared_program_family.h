#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/graph/slots.h"
#include "prepared_output.h"

namespace hyle::inference {

struct ResourceBindings final {
  std::span<const graph::slots::SourceBinding> stores{};
  std::span<const graph::NodeId> extra_stores{};
};

// One immutable, concrete route. Executable and report are always selected
// together from the same finite family; output facts retain their published
// order. These are cold Graph handles and metadata, not another native owner.
class PreparedProgram final {
  const graph::Graph *logical_ = nullptr;
  // Descriptions only; destroy the compiled route before this owned Graph.
  graph::Graph specialized_resources_;

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
  const graph::Graph &resource_graph() const noexcept {
    return specialized_resources_ ? specialized_resources_ : *logical_;
  }

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
  // logical is the caller's stable original Graph and must outlive every
  // child. Variant and resource spans are borrowed only during this call.
  Status prepare(graph::CompiledGraph compiled, const graph::Graph &logical,
      std::span<const graph::ExactShapeVariant> variants,
      ResourceBindings resources = {},
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept {
    if (parent_.executable || parent_.report)
      return Status::FailedPrecondition("program family is already prepared");
    if (!logical || !compiled.executable || !compiled.report || variants.empty() ||
        compiled.executable.exact_variant_count() != variants.size() ||
        compiled.report.exact_variant_count() != variants.size())
      return Status::InvalidArgument("program family exact variant counts differ");
    HYLE_ASSIGN_OR_RETURN(const bool specialize_resources, needs_specialization(logical, resources));
    try {
      std::vector<std::unique_ptr<PreparedProgram>> children;
      children.reserve(variants.size());
      for (std::size_t index = 0; index < variants.size(); ++index) {
        auto child = std::make_unique<PreparedProgram>();
        child->logical_ = &logical;
        HYLE_ASSIGN_OR_RETURN(child->compiled.executable, compiled.executable.exact_variant(index));
        HYLE_ASSIGN_OR_RETURN(child->compiled.report, compiled.report.exact_variant(index));
        HYLE_RETURN_IF_ERROR(check_child(child->compiled, logical, variants[index].shape));
        child->outputs_.resize(child->compiled.executable.output_count());
        child->output_indices_.resize(logical.value_count());
        HYLE_RETURN_IF_ERROR(prepare_outputs(child->compiled, child->outputs_, child->output_indices_));
        if (specialize_resources) {
          HYLE_ASSIGN_OR_RETURN(auto specialized, graph::specialize_exact(logical, variants[index].shape, allocator));
          if (specialized.signature != child->compiled.executable.specialization())
            return Status::FailedPrecondition("resource graph specialization differs from its selected child");
          child->specialized_resources_ = std::move(specialized.graph);
        }
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
  static Status check_resource_value(const graph::Graph &logical, graph::ValueId value,
      bool &specialize) noexcept {
    auto description = logical.value_description(value);
    if (description) return Status::Ok();
    if (!description.error().IsFailedPrecondition()) return description.error();
    specialize = true;
    return Status::Ok();
  }

  static Result<bool> needs_specialization(const graph::Graph &logical,
      ResourceBindings resources) noexcept {
    bool specialize = false;
    for (const auto &association : resources.stores) {
      HYLE_ASSIGN_OR_RETURN(const auto bank, logical.node_input(association.gather, 0));
      HYLE_ASSIGN_OR_RETURN(const auto ids, logical.node_input(association.gather, 1));
      HYLE_ASSIGN_OR_RETURN(const auto gathered, logical.result(association.gather, 0));
      HYLE_ASSIGN_OR_RETURN(const auto updates, logical.node_input(association.store, 0));
      HYLE_ASSIGN_OR_RETURN(const auto destinations, logical.node_input(association.store, 1));
      for (const auto value : {bank, ids, gathered, updates, destinations})
        HYLE_RETURN_IF_ERROR(check_resource_value(logical, value, specialize));
    }
    for (const auto store : resources.extra_stores) {
      HYLE_ASSIGN_OR_RETURN(const auto updates, logical.node_input(store, 0));
      HYLE_ASSIGN_OR_RETURN(const auto destinations, logical.node_input(store, 1));
      HYLE_RETURN_IF_ERROR(check_resource_value(logical, updates, specialize));
      HYLE_RETURN_IF_ERROR(check_resource_value(logical, destinations, specialize));
    }
    return specialize;
  }

  static Status check_child(const graph::CompiledGraph &compiled, const graph::Graph &logical,
      std::span<const graph::ShapeBinding> required) noexcept {
    HYLE_RETURN_IF_ERROR(detail::check_output_route(compiled));
    const auto &executable = compiled.executable;
    if (executable.input_count() != logical.input_count() ||
        executable.output_count() != logical.output_count() ||
        executable.node_count() != logical.node_count() || compiled.report.node_count() != logical.node_count())
      return Status::InvalidArgument("program child logical graph counts differ");
    for (std::size_t index = 0; index < logical.input_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(const auto expected, logical.input_id(index));
      HYLE_ASSIGN_OR_RETURN(const auto actual, executable.input_id(index));
      if (actual != expected) return Status::InvalidArgument("program child logical input differs");
    }
    for (std::size_t index = 0; index < logical.output_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(const auto expected, logical.output_id(index));
      HYLE_ASSIGN_OR_RETURN(const auto actual, executable.output_id(index));
      if (actual != expected) return Status::InvalidArgument("program child logical output differs");
    }
    // The SDK signature contains canonical derived bindings as well. Match
    // required inputs as a subset, without specializing a concrete graph.
    const auto signature = executable.specialization();
    for (const auto binding : required) {
      bool found = false;
      for (std::size_t index = 0; index < signature.binding_count(); ++index) {
        HYLE_ASSIGN_OR_RETURN(const auto actual, signature.binding(index));
        if (actual.dimension != binding.dimension) continue;
        if (actual.value != binding.value)
          return Status::FailedPrecondition("program child signature differs from required bindings");
        found = true;
        break;
      }
      if (!found) return Status::FailedPrecondition("program child signature is missing a required binding");
    }
    return Status::Ok();
  }

  graph::CompiledGraph parent_;
  std::vector<std::unique_ptr<PreparedProgram>> children_;
};

} // namespace hyle::inference
