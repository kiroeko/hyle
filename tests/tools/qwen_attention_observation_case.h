#pragma once

#include <cstring>
#include <limits>

#include "../../engine/models/qwen35/generation_publication.h"
#include "hyle/graph/detail/region_access.h"

// Included after the original product Model. This observer borrows its retired
// region outputs; it neither publishes state nor adds graph outputs/commands.
namespace qwen_attention_observation_case {
namespace q = qwen35;
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gd = hyle::graph::detail;

struct TensorImage final {
  c::TensorDesc description;
  std::vector<std::byte> bytes;
};

inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// The report identifies the actual physical input, including placement copies.
// A metadata-only View can project an original ready lease, but cannot invent
// permission for a raw native buffer or an unrelated allocation tail.
inline c::ReadLease output_lease(q::StepState &step, std::uint32_t physical,
    std::size_t depth = 0) {
  const auto &report = step.phase->child->compiled.report;
  const auto &graph = step.phase->state_graph();
  require(depth < report.physical_value_count(), "attention observation has a cyclic physical alias");
  const auto facts = q::take(report.physical_value(physical));
  for (std::size_t index = 0; index < report.step_count(); ++index) {
    const auto selected = q::take(report.step(index));
    if (selected.kind != g::CompileStepKind::kProviderRegion) continue;
    for (std::size_t port = 0; port < selected.output_count; ++port) {
      if (q::take(report.step_output(index, port)) != physical) continue;
      require(selected.node_count != 0, "attention observation producer has no original node");
      struct Visit final {
        std::uint32_t step;
        std::size_t port;
        c::ReadLease lease;
      } visit{static_cast<std::uint32_t>(index), port, {}};
      q::check(gd::RegionAccess::visit_retired(*step.execution.run,
          q::take(report.step_node(index, 0)),
          +[](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
            auto &request = *static_cast<Visit *>(opaque);
            if (view.step != request.step || !view.prepared_vtable || !view.prepared_vtable->output_lease)
              return hyle::Status::Unimplemented("attention observation requires the original region output lease");
            HYLE_ASSIGN_OR_RETURN(request.lease,
                view.prepared_vtable->output_lease(view.prepared, view.invocation, request.port));
            return hyle::Status::Ok();
          }, &visit));
      require(visit.lease.valid(), "attention observation producer returned no ready lease");
      return q::take(visit.lease.subview(0, q::take(facts.description.required_bytes())));
    }
  }
  require(facts.alias_source.has_value() && facts.value.kind() == g::ValueKind::kNodeOutput,
      "attention observation value has no original region output or metadata alias");
  const auto producer = q::take(graph.producer(facts.value));
  const auto operation = q::take(graph.operation(producer));
  const auto *view = operation.get_if<c::View>();
  require(view && q::take(graph.node_input_count(producer)) == 1,
      "attention observation cannot reinterpret an unowned non-View value");
  const auto parent = q::take(report.physical_value(*facts.alias_source));
  require(parent.value == q::take(graph.node_input(producer, 0)),
      "attention observation alias is not its original View input");
  std::size_t offset = 0;
  if (view->kind() == c::ViewKind::kSlice) {
    auto axis = view->axis();
    if (axis < 0) axis += static_cast<std::int32_t>(parent.description.rank());
    require(axis >= 0 && static_cast<std::size_t>(axis) < parent.description.rank() &&
        view->start() >= 0 && view->step() == 1 && parent.description.is_plain(),
        "attention observation only projects an original contiguous positive Slice");
    const auto stride = parent.description.byte_strides()[axis];
    require(stride >= 0 && (stride == 0 || static_cast<std::uint64_t>(view->start()) <=
        std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(stride)),
        "attention observation Slice offset overflows");
    offset = static_cast<std::size_t>(view->start()) * static_cast<std::size_t>(stride);
  } else require(view->kind() == c::ViewKind::kReshape || view->kind() == c::ViewKind::kPermute ||
      view->kind() == c::ViewKind::kBroadcast, "attention observation has an unsupported metadata View");
  auto original = output_lease(step, *facts.alias_source, depth + 1);
  return q::take(original.subview(offset, q::take(facts.description.required_bytes())));
}

inline TensorImage read(q::Model &model, q::StepState &step, std::uint32_t physical) {
  TensorImage image{q::take(step.phase->child->compiled.report.physical_value(physical)).description, {}};
  auto lease = output_lease(step, physical);
  image.bytes.resize(q::take(image.description.required_bytes()));
  q::check(model.execution.read(q::take(lease.view()), image.bytes));
  return image;
}

struct AttentionObservation final {
  bool captured = false, device_lengths = false;
  q::GraphRole role = q::GraphRole::kText;
  std::size_t before_length = 0, input_rows = 0, accepted_rows = 0;
  std::uint32_t attention_step = UINT32_MAX;
  TensorImage query, attention, query_length_tensor, kv_length_tensor;
  std::vector<std::int32_t> query_lengths, kv_lengths;

  void capture(q::Model &model, q::StepState &step) {
    require(!captured && step.execution.run && step.phase && step.phase->child &&
        step.members.size() == 1 && !step.phase->attention_nodes.empty() &&
        gd::RegionAccess::successful_outcome(*step.execution.run),
        "attention observation requires its original successful single-request verifier");
    const auto &graph = step.phase->state_graph();
    const auto &report = step.phase->child->compiled.report;
    const auto node = step.phase->attention_nodes.front();
    for (std::size_t index = 0; index < report.step_count(); ++index) {
      const auto selected = q::take(report.step(index));
      for (std::size_t n = 0; n < selected.node_count; ++n) if (q::take(report.step_node(index, n)) == node) {
        require(attention_step == UINT32_MAX && selected.kind == g::CompileStepKind::kProviderRegion &&
            selected.node_count == 1, "attention observation requires its unchanged singleton attention region");
        attention_step = static_cast<std::uint32_t>(index);
      }
    }
    require(attention_step != UINT32_MAX, "attention observation cannot find its actual attention region");
    const auto selected = q::take(report.step(attention_step));
    const auto input = [&](std::size_t port) {
      const auto logical = q::take(graph.node_input(node, port));
      std::uint32_t found = UINT32_MAX;
      for (std::size_t i = 0; i < selected.input_count; ++i) {
        const auto physical = q::take(report.step_input(attention_step, i));
        if (q::take(report.physical_value(physical)).value != logical) continue;
        require(found == UINT32_MAX, "attention observation input has ambiguous physical instances");
        found = physical;
      }
      require(found != UINT32_MAX, "attention observation logical input is not an actual region boundary");
      return found;
    };
    std::uint32_t output = UINT32_MAX;
    const auto result = q::take(graph.result(node));
    for (std::size_t i = 0; i < selected.output_count; ++i) {
      const auto physical = q::take(report.step_output(attention_step, i));
      if (q::take(report.physical_value(physical)).value != result) continue;
      require(output == UINT32_MAX, "attention observation output has ambiguous physical instances");
      output = physical;
    }
    require(output != UINT32_MAX, "attention observation cannot find its original output");
    query = read(model, step, input(0));
    attention = read(model, step, output);
    device_lengths = q::take(graph.node_input_count(node)) == 5;
    const auto &member = step.members.front();
    before_length = member.base->length; input_rows = member.input_tokens; accepted_rows = member.accepted_tokens;
    role = step.phase->family->role;
    if (device_lengths) {
      query_length_tensor = read(model, step, input(3));
      kv_length_tensor = read(model, step, input(4));
      const auto integer = [](const TensorImage &tensor) {
        require(tensor.description.data_type() == c::DataType::kInt32 && tensor.bytes.size() == sizeof(std::int32_t),
            "attention observation device length is not an original single-request I32 value");
        std::int32_t value;
        std::memcpy(&value, tensor.bytes.data(), sizeof(value));
        return value;
      };
      query_lengths.push_back(integer(query_length_tensor));
      kv_lengths.push_back(integer(kv_length_tensor));
    } else {
      const auto host = member.layout.host();
      require(host.query_indptr.size() == 2 && host.kv_indptr.size() == 2 &&
          host.last_page_lengths.size() == 1 && host.kv_indptr[1] > host.kv_indptr[0],
          "attention observation requires the original nonempty single-request host layout");
      query_lengths.push_back(host.query_indptr[1] - host.query_indptr[0]);
      kv_lengths.push_back(static_cast<std::int32_t>((host.kv_indptr[1] - host.kv_indptr[0] - 1) *
          q::cache::Plan::page_rows + host.last_page_lengths[0]));
    }
    require(query_lengths.front() == static_cast<std::int32_t>(input_rows) &&
        kv_lengths.front() == static_cast<std::int32_t>(before_length + input_rows),
        "attention observation actual Q/KV lengths differ from this original verifier episode");
    captured = true;
  }
};

// Composition preserves the original ticket, record and failure/retirement
// handling. selected() only observes after the original selection was recorded.
class ObservedPublication final {
  q::GenerationPublicationBatch &original_;
  q::Model &model_;
  AttentionObservation &observation_;
public:
  ObservedPublication(q::GenerationPublicationBatch &original, q::Model &model, AttentionObservation &observation)
      : original_(original), model_(model), observation_(observation) {}
  q::GenerationCommitRecord *record(std::size_t index) const noexcept { return original_.record(index); }
  hyle::inference::PreparedRecordCommit *ticket(std::size_t index) noexcept { return original_.ticket(index); }
  hyle::inference::PreparedRecordCommit *prepare(std::size_t index, const q::EvaluationAttemptFacts &facts) {
    return original_.prepare(index, facts);
  }
  template<class Result>
  void selected(std::size_t index, const Result &result, std::size_t quota) {
    original_.selected(index, result, quota);
    require(index == 0 && model_.active_step, "attention observation lost its original selected episode");
    observation_.capture(model_, *model_.active_step);
  }
  void complete(std::size_t index, const q::EvaluationResultFacts &facts) { original_.complete(index, facts); }
  void finished(std::chrono::steady_clock::time_point endpoint, double seconds) noexcept { original_.finished(endpoint, seconds); }
  void retired(const hyle::inference::RetirementObservation &observation) noexcept { original_.retired(observation); }
};
} // namespace qwen_attention_observation_case
