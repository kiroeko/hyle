#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

namespace hyle::inference {

// Cold facts for one concrete published output. Resources and their owners
// remain with the original compile environment; generated output addresses
// are still obtained from GraphOutput after successful completion.
struct PreparedOutputFacts final {
  graph::ValueId value{};
  compute::TensorDesc description{};
  graph::StorageDomainId storage{};
  graph::OutputOwnership ownership = graph::OutputOwnership::kCaller;

  bool caller_owned() const noexcept {
    return ownership == graph::OutputOwnership::kCaller;
  }
};

namespace detail {
inline Status check_output_route(const graph::CompiledGraph &compiled) noexcept {
  const auto &executable = compiled.executable;
  const auto &report = compiled.report;
  if (!executable || !report)
    return Status::FailedPrecondition(
        "prepared output requires a valid executable and compile report");
  if (!executable.has_execution_route() || !report.has_execution_route() ||
      executable.is_shape_domain() || report.is_shape_domain())
    return Status::FailedPrecondition(
        "prepared output requires a concrete execution route");
  const auto signature = executable.specialization();
  const auto reported_signature = report.specialization();
  if (!signature || !reported_signature || signature != reported_signature)
    return Status::InvalidArgument(
        "prepared output executable and report specializations differ");
  return Status::Ok();
}
} // namespace detail

// Pass the Executable and report from the same compile result, or the same
// selected exact child. Static default-route compilation is already exact.
// A shape-domain parent has no single concrete output description. Neither
// the backend name nor workspace/output flags determine output ownership.
inline Result<PreparedOutputFacts>
prepare_output(const graph::CompiledGraph &compiled,
               graph::ValueId value) noexcept {
  const auto &executable = compiled.executable;
  const auto &report = compiled.report;
  HYLE_RETURN_IF_ERROR(detail::check_output_route(compiled));
  if (!value)
    return std::unexpected(
        Status::InvalidArgument("prepared output value is invalid"));
  PreparedOutputFacts result;
  result.value = value;
  bool published = false;
  for (std::size_t index = 0; index < executable.output_count(); ++index) {
    auto output = executable.output_id(index);
    if (!output) return std::unexpected(output.error());
    if (*output != value) continue;
    if (published)
      return std::unexpected(Status::InvalidArgument(
          "prepared output value has ambiguous published outputs"));
    auto ownership = executable.output_ownership(index);
    if (!ownership) return std::unexpected(ownership.error());
    result.ownership = *ownership;
    published = true;
  }
  if (!published)
    return std::unexpected(
        Status::NotFound("value is not a prepared executable output"));

  bool found = false;
  for (std::size_t index = 0; index < report.physical_value_count(); ++index) {
    auto physical = report.physical_value(index);
    if (!physical) return std::unexpected(physical.error());
    if (physical->value != value || !physical->external_output) continue;
    if (found)
      return std::unexpected(Status::InvalidArgument(
          "prepared output value has ambiguous physical outputs"));
    result.description = physical->description;
    result.storage = physical->storage;
    found = true;
  }
  if (!found)
    return std::unexpected(
        Status::NotFound("prepared output has no physical output report"));
  return result;
}

// Caller-owned cold storage: one fact per published output and a compact
// ordinal-to-output index. Scan each report once rather than once per port.
// The caller publishes these arrays only after success. Full ValueId equality
// remains authoritative.
inline Status prepare_outputs(const graph::CompiledGraph &compiled,
    std::span<PreparedOutputFacts> outputs,
    std::span<std::uint32_t> output_indices) noexcept {
  HYLE_RETURN_IF_ERROR(detail::check_output_route(compiled));
  const auto &executable = compiled.executable;
  if (outputs.size() != executable.output_count() ||
      outputs.size() > (std::numeric_limits<std::uint32_t>::max)())
    return Status::InvalidArgument("prepared output storage count differs");
  constexpr auto missing = (std::numeric_limits<std::uint32_t>::max)();
  for (auto &index : output_indices) index = missing;
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto value, executable.output_id(index));
    if (!value || value.ordinal() >= output_indices.size())
      return Status::InvalidArgument("prepared output ordinal exceeds index storage");
    if (output_indices[value.ordinal()] != missing)
      return Status::InvalidArgument("prepared output value has ambiguous published outputs");
    HYLE_ASSIGN_OR_RETURN(auto ownership, executable.output_ownership(index));
    outputs[index] = {.value = value, .ownership = ownership};
    output_indices[value.ordinal()] = static_cast<std::uint32_t>(index);
  }
  const auto &report = compiled.report;
  for (std::size_t index = 0; index < report.physical_value_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto physical, report.physical_value(index));
    if (!physical.external_output || !physical.value ||
        physical.value.ordinal() >= output_indices.size()) continue;
    const auto output_index = output_indices[physical.value.ordinal()];
    if (output_index == missing) continue;
    auto &output = outputs[output_index];
    if (output.value != physical.value) continue;
    if (output.description.valid())
      return Status::InvalidArgument("prepared output value has ambiguous physical outputs");
    output.description = physical.description;
    output.storage = physical.storage;
  }
  for (const auto &output : outputs)
    if (!output.description.valid() || !output.storage.valid())
      return Status::NotFound("prepared output has no physical output report");
  return Status::Ok();
}

// Capture the original completed output owner before append extraction/reset
// revokes result observation. The model selects its actual prefix first and
// controls publication; this helper neither waits, copies nor publishes state.
// Facts and Invocation must belong to the same prepared exact child.
inline Result<compute::ReadLease> retain_output(graph::Invocation &run,
    const PreparedOutputFacts &facts,
    const compute::ResourceBudget &budget = {}) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto output, run.result(facts.value));
  return output.storage_lease(budget);
}

// Both constraints apply to the one original completed output owner and its
// actual prepared storage domain. This does not reserve provider allocations.
inline Result<compute::ReadLease> retain_output(graph::Invocation &run,
    const PreparedOutputFacts &facts, const compute::ResourceBudget &budget,
    const compute::ResourceBudget &additional) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto output, run.result(facts.value));
  return output.storage_lease(budget, additional);
}

} // namespace hyle::inference
