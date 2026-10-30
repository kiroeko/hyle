#pragma once

#include "../../engine/inference/program_slots.h"
#include "../../engine/inference/prepared_program_family.h"

// Two public Graph invocations from one concrete program. No Model, synthetic
// Completion, replacement resource owner or test-only execution route.
namespace inference_program_carrier_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace g = hyle::graph;
namespace c = hyle::compute;
using qwen_host_image_case::require;
struct Scope final { c::ReadLease first, second; };
struct IoHandles final {
  i::ProgramIo::Input first, second;
  i::ProgramIo::Readback readback;
};
using ProgramSlots = i::ProgramSlots<Scope, IoHandles, 2>;

inline std::shared_ptr<Scope> inputs(i::Backend &backend,
    const std::array<float, 4> &first, const std::array<float, 4> &second) {
  const auto ready = [&](const auto &values) {
    auto buffer = q::take(q::mem::Buffer::create(backend.device(), sizeof(values), 256));
    q::check(backend.write(buffer.view(), std::as_bytes(std::span(values))));
    return q::take(backend.adopt_buffer(std::move(buffer)));
  };
  return std::make_shared<Scope>(Scope{ready(first), ready(second)});
}

inline int run(q::BackendKind kind) {
  auto backend = q::take(i::Backend::create({.primary = kind}));
  const auto description = q::dense(q::Type::kFloat32, {4});
  const auto bad_description = q::dense(q::Type::kFloat32, {2});
  const std::array<std::int64_t, 1> four{4}, padded_stride{8}, broadcast_stride{0};
  const auto padded_description = q::take(c::TensorDesc::strided_bytes(q::Type::kFloat32, four, padded_stride));
  const auto broadcast_description = q::take(c::TensorDesc::strided_bytes(q::Type::kFloat32, four, broadcast_stride));
  auto builder = q::take(g::GraphBuilder::create());
  const auto first = q::take(builder.input(description));
  const auto second = q::take(builder.input(description));
  const auto output = q::take(builder.add(c::Binary{.kind = c::BinaryKind::kAdd}, first, second));
  q::check(builder.publish(output));
  const auto graph = q::take(std::move(builder).finish());
  const auto compiled = q::take(backend.compile(graph, {}));
  // Cold metadata only: neither a model fixture nor another native submit.
  {
    auto exact_builder = q::take(g::GraphBuilder::create());
    const auto rows = q::take(exact_builder.dimension({1, 4, 1}));
    const auto spec = q::take(g::TensorSpec::dense(q::Type::kFloat32, std::array{g::Dimension(rows)}));
    const auto input = q::take(exact_builder.input(spec));
    const auto unused = q::take(exact_builder.input(padded_description));
    const auto sum = q::take(exact_builder.add(c::Binary{.kind = c::BinaryKind::kAdd}, spec, input, input));
    q::check(exact_builder.publish(sum));
    const auto logical = q::take(std::move(exact_builder).finish());
    const std::array point{g::ShapeBinding{rows, 4}}, wrong_point{g::ShapeBinding{rows, 3}};
    const std::array variants{g::ExactShapeVariant{point}}, wrong_variants{g::ExactShapeVariant{wrong_point}};
    const auto finite = q::take(backend.compile(logical, {},
        {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants}));
    i::PreparedProgramFamily family;
    const std::array<g::NodeId, 1> invalid_resources{};
    const auto invalid = family.prepare(finite, logical, variants, {.extra_stores = invalid_resources});
    require(invalid.IsInvalidArgument() && !family.size() && !family.parent().executable && !family.parent().report,
        "invalid exact resource node published a partial cold family");
    const auto mismatch = family.prepare(finite, logical, wrong_variants);
    require(mismatch.IsFailedPrecondition() && !family.size() && !family.parent().executable && !family.parent().report &&
        finite.executable && finite.report && finite.executable.exact_variant_count() == 1,
        "wrong exact signature published a family or consumed its original compiled route");
    q::check(family.prepare(finite, logical, variants));
    require(family.size() == 1 && &family.child(0).resource_graph() == &logical &&
        family.child(0).output(sum).description == description && finite.executable && finite.report,
        "exact family retry lost its original output facts or invented a resource graph for empty bindings");
    const auto &child = family.child(0);
    i::ProgramIo child_io;
    i::ProgramCarrier<Scope> child_carrier;
    q::check(child_io.prepare(child.compiled, 0, 2, 0, 0));
    const auto non_input = child_io.add_input(sum), foreign_input = child_io.add_input(first);
    require(first.ordinal() == input.ordinal() && first != input && !non_input && !foreign_input &&
        non_input.error().IsNotFound() && foreign_input.error().IsNotFound(),
        "cold input registration accepted a noninput or matched only a foreign ValueId ordinal");
    const auto child_input = q::take(child_io.add_input(input));
    const auto unused_input = q::take(child_io.add_input(unused)); q::check(child_io.seal());
    require(child_io.input_facts(child_input).value == input && child_io.input_facts(child_input).description == description &&
        child_io.input_facts(unused_input).value == unused &&
        child_io.input_facts(unused_input).description == padded_description && padded_description != description,
        "cold exact input facts lost their concrete shape, original unused input or full strided description");
    q::check(child_carrier.prepare(child.compiled, backend.queue_bindings()));
    q::check(child_carrier.initialize({.io = child_io, .invocation = {.resources = g::ResourceOptions{}}}));
    const auto *child_run = q::take(child_carrier.borrow());
    require(child_run->specialization() == child.compiled.executable.specialization() &&
        child_run->exact_variant_index() == std::optional<std::size_t>{0},
        "cold child installation lost its original complete signature or selected family variant");
  }
  // Original cold carriers and IO only. An invalid count must not start the
  // recipe; a later-slot failure must not publish the earlier prepared slot.
  {
    ProgramSlots rejected;
    std::size_t visited = 0;
    const auto prepare_empty = [&](std::size_t index, ProgramSlots::Slot &slot) noexcept -> hyle::Status {
      ++visited;
      if (index == 1) return hyle::Status::InvalidArgument("deliberate second-slot cold IO refusal");
      HYLE_RETURN_IF_ERROR(slot.io.prepare(compiled, 0, 0, 0, 0));
      return slot.io.seal();
    };
    const auto zero_count = rejected.prepare(compiled, backend.queue_bindings(), 0, {}, prepare_empty);
    const auto oversized = rejected.prepare(compiled, backend.queue_bindings(), 3, {}, prepare_empty);
    require(zero_count.IsInvalidArgument() && oversized.IsInvalidArgument() && !visited &&
        !rejected.size() && rejected.slots().empty() && !rejected.pending(),
        "invalid slot count began cold preparation or published a carrier");
    const auto partial_group = rejected.prepare(compiled, backend.queue_bindings(), 2, {}, prepare_empty);
    const auto partial_initialize = rejected.initialize_all({});
    const auto partial_retry = rejected.prepare(compiled, backend.queue_bindings(), 2, {}, prepare_empty);
    require(partial_group.IsInvalidArgument() && partial_initialize.IsFailedPrecondition() &&
        partial_retry.IsFailedPrecondition() && visited == 2 && !rejected.size() &&
        rejected.slots().empty() && !rejected.pending(),
        "partial cold slot group published or retried its original nonmovable inventory");
  }
  // Same executable/report handles, but a distinct borrowed program object.
  // IO must belong to the exact stable CompiledGraph selected by its carrier.
  const auto foreign_compiled = compiled;
  auto a_scope = inputs(backend, {1, 2, 3, 4}, {10, 20, 30, 40});
  auto b_scope = inputs(backend, {-1, -2, -3, -4}, {7, 11, 17, 23});
  const auto view = [&](const c::ReadLease &owner, const auto &shape) {
    return q::take(c::ConstTensorView::create(shape, q::take(owner.view())));
  };
  ProgramSlots programs;
  q::check(programs.prepare(compiled, backend.queue_bindings(), 2, {},
      [&](std::size_t, ProgramSlots::Slot &slot) noexcept -> hyle::Status {
        HYLE_RETURN_IF_ERROR(slot.io.prepare(compiled, 0, 2, 1, 0));
        HYLE_ASSIGN_OR_RETURN(slot.data.first, slot.io.add_input(first));
        HYLE_ASSIGN_OR_RETURN(slot.data.second, slot.io.add_input(second));
        HYLE_ASSIGN_OR_RETURN(slot.data.readback, slot.io.add_readback(output, backend.storage_domains()));
        return slot.io.seal();
      }));
  require(programs.size() == 2 && programs.slots().size() == 2 && !programs.pending(),
      "cold slot group did not publish its complete original carrier inventory");
  auto &a_slot = programs[0], &b_slot = programs[1];
  auto &a_io = a_slot.io, &b_io = b_slot.io;
  auto &a = a_slot.carrier, &b = b_slot.carrier;
  const auto a_first = a_slot.data.first, a_second = a_slot.data.second, b_first = b_slot.data.first,
      b_second = b_slot.data.second;
  const auto a_readback = a_slot.data.readback, b_readback = b_slot.data.readback;
  i::ProgramIo partial_io, unsealed_io, foreign_io;
  // Install the original readback first, then one valid input and the wrong
  // second shape. Failure must return that actual caller writer, if applicable.
  q::check(partial_io.prepare(compiled, 2, 0, 1, 0));
  const auto partial_readback = q::take(partial_io.add_readback(output, backend.storage_domains()));
  q::check(partial_io.add_fixed(first, view(a_scope->first, description)));
  q::check(partial_io.add_fixed(second, view(a_scope->second, bad_description)));
  q::check(partial_io.seal());
  auto *partial_output = q::take(partial_io.readback(partial_readback));
  const bool caller_writer = partial_output->facts.caller_owned();
  require(caller_writer ? partial_output->pool().valid() && partial_output->pool().stats().live_blocks == 1
                        : !partial_output->pool().valid(),
      "partial-installation readback did not retain its actual output ownership");
  q::check(unsealed_io.prepare(compiled, 0, 0, 0, 0));
  q::check(foreign_io.prepare(foreign_compiled, 0, 0, 0, 0)); q::check(foreign_io.seal());
  const g::InvocationOptions options{.resources = g::ResourceOptions{}};
  const auto unsealed = a.initialize({.io = unsealed_io, .invocation = options});
  const auto foreign = a.initialize({.io = foreign_io, .invocation = options});
  require(unsealed.IsFailedPrecondition() && foreign.IsFailedPrecondition() &&
      !a.initialized() && !a.pending() && !b.initialized(),
      "unsealed or foreign IO published a carrier");
  const auto partial = a.initialize({.io = partial_io, .invocation = options});
  require(partial.IsInvalidArgument() && !a.initialized() && !a.pending() && !b.initialized(),
      "cold bind failure published a half-initialized carrier or touched its peer");
  require(caller_writer ? partial_output->pool().valid() && partial_output->pool().stats().live_blocks == 0
                        : !partial_output->pool().valid(),
      "failed IO installation retained the original caller writer or invented a native-output pool");
  // Move the one reclaimed readback owner to the valid inventory. Its next
  // bind must reacquire the same pool, and the two original submits below must
  // produce the original correct outputs. No extra native work is submitted.
  auto *a_output = q::take(a_io.readback(a_readback));
  *a_output = std::move(*partial_output);
  require(!partial_output->pool().valid() &&
      (caller_writer ? a_output->pool().valid() && a_output->pool().stats().live_blocks == 0
                     : !a_output->pool().valid()),
      "readback recovery duplicated its pool or lost the original output ownership");
  const auto missing = a.borrow();
  require(!missing && missing.error().IsFailedPrecondition(), "failed cold carrier admitted an original invocation");
  std::array<std::size_t, 2> initialization_calls{};
  const auto initialize_a = [&] {
    ++initialization_calls[0];
    return ProgramSlots::Initialization{.invocation = options};
  };
  const auto initialize_b = [&] {
    ++initialization_calls[1];
    return ProgramSlots::Initialization{.invocation = options};
  };
  std::size_t invalid_initialization_calls = 0;
  const auto outside = programs.initialize_with(2, [&] {
    ++invalid_initialization_calls;
    return ProgramSlots::Initialization{.invocation = options};
  });
  require(outside.IsOutOfRange() && !invalid_initialization_calls && !a.initialized() && !b.initialized(),
      "out-of-range initialization evaluated a recipe or installed an original carrier");
  q::check(programs.initialize_with(1, initialize_b));
  require(b.initialized() && !a.initialized() && initialization_calls == std::array<std::size_t, 2>{0, 1},
      "initializing B repaired failed A or evaluated another slot's installation recipe");
  q::check(programs.initialize_with(0, initialize_a));
  require(initialization_calls == std::array<std::size_t, 2>{1, 1},
      "initial slot installation did not evaluate each original recipe exactly once");
  require(!caller_writer || a_output->pool().stats().live_blocks == 1,
      "valid IO retry did not reacquire the failed installation's original readback writer");
  auto *a_run = q::take(a.borrow()); auto *b_run = q::take(b.borrow());
  require(a_run != b_run && a_run->valid() && b_run->valid(), "independent carriers share one original mutable invocation");
  require(a_run->specialization() == compiled.executable.specialization() &&
      b_run->specialization() == compiled.executable.specialization() &&
      !a_run->exact_variant_index() && !b_run->exact_variant_index(),
      "standalone installation changed its original signature or invented a family variant");
  // Preserve Graph's original checked read owners after the retained Scope
  // retires; these are dynamic bindings, not dangling cold borrowed views.
  q::check(a_io.bind(*a_run, a_first, a_scope->first));
  q::check(a_io.bind(*a_run, a_second, a_scope->second));
  q::check(b_io.bind(*b_run, b_first, b_scope->first));
  q::check(b_io.bind(*b_run, b_second, b_scope->second));
  // Both views are valid and have the original shape. Their different stride
  // must reject without replacing the previously bound dense read permission;
  // the original native submissions below still check every output element.
  const auto typed_stride = a_io.bind(*a_run, a_first, view(a_scope->first, broadcast_description));
  const auto explicit_stride = a_io.bind(*a_run, a_first, broadcast_description, a_scope->first);
  require(typed_stride.IsInvalidArgument() && explicit_stride.IsInvalidArgument(),
      "program IO replaced actual typed strides with its cold expected description");
  const auto duplicate = a.initialize({.io = a_io, .invocation = options});
  require(duplicate.IsFailedPrecondition() && q::take(a.borrow()) == a_run && q::take(b.borrow()) == b_run,
      "already initialized carrier replaced its live original invocation or touched its peer");
  a_slot.planned_uses = 5; a_slot.preparation_uses = 3;
  b_slot.planned_uses = 7; b_slot.preparation_uses = 4;
  q::check(programs.initialize_with(0, initialize_a));
  q::check(programs.initialize_with(1, initialize_b));
  q::check(programs.initialize_all({.invocation = options}));
  q::check(programs.initialize_all_with([&] {
    ++invalid_initialization_calls;
    return ProgramSlots::Initialization{.invocation = options};
  }));
  require(q::take(a.borrow()) == a_run && q::take(b.borrow()) == b_run &&
      a_slot.planned_uses == 5 && a_slot.preparation_uses == 3 &&
      b_slot.planned_uses == 7 && b_slot.preparation_uses == 4 &&
      initialization_calls == std::array<std::size_t, 2>{1, 1} && !invalid_initialization_calls,
      "idempotent slot-group initialization evaluated its recipe or replaced an original run or replay counter");
  q::check(a_run->prepare_replay(a.queues(), {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  q::check(b_run->prepare_replay(b.queues(), {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  // A missing scope must reject before native submit, preserving the actual
  // prepared run. It is not evidence of an asynchronously pending device.
  const auto no_scope = a.submit({});
  require(!no_scope && no_scope.error().IsUnavailable() && !a.pending() && q::take(a.borrow()) == a_run,
      "rejected resource reservation submitted work or lost its original prepared run");
  std::weak_ptr<Scope> a_lifetime = a_scope, b_lifetime = b_scope;
  auto a_done = q::take(a.submit(a_scope)); a_scope.reset();
  bool pending_observed = a.pending() || programs.pending();
  const auto a_admission = a.borrow();
  require(a_admission ? !a.pending() && !a_run->in_flight() : a_admission.error().IsUnavailable(),
      "carrier admitted reuse before original native retirement");
  auto b_done = q::take(b.submit(b_scope)); b_scope.reset();
  pending_observed = pending_observed || b.pending() || programs.pending();
  const auto b_admission = b.borrow();
  require(b_admission ? !b.pending() && !b_run->in_flight() : b_admission.error().IsUnavailable(),
      "peer carrier admitted reuse before its own original native retirement");
  q::check(a_done.wait()); q::check(b_done.wait()); q::check(backend.collect_retired());
  require(!a.pending() && !b.pending() && !programs.pending() && a_lifetime.expired() && b_lifetime.expired(),
      "successful real Completions did not release their own original retained scopes");
  std::array<float, 4> actual_a{}, actual_b{};
  q::check(a_io.read(*a_run, a_readback, std::as_writable_bytes(std::span(actual_a))));
  q::check(b_io.read(*b_run, b_readback, std::as_writable_bytes(std::span(actual_b))));
  require(actual_a == std::array<float, 4>{11, 22, 33, 44} && actual_b == std::array<float, 4>{6, 9, 14, 19},
      "two native carriers mixed fixed inputs or their prepared readbacks");
  a_done = {}; b_done = {};
  // Keep the actual Graph output pin. Carrier metadata grants no extra right
  // to unbind the original Graph while an observation still owns its slot.
  auto pinned = q::take(a_run->result(output)); const std::array release{first};
  const auto pinned_release = a.release_inputs(release);
  require(pinned_release.IsUnavailable() && pinned.valid(), "program carrier bypassed an original Graph result pin");
  pinned.reset();
  q::check(a.release_inputs(release));
  const auto retired_a = a.retire(backend); q::check(retired_a.status);
  require(retired_a.retired && a.initialized() && a.closing() && !b.closing() && q::take(b.borrow()) == b_run,
      "actual A retirement discarded its original handle early or closed independent B");
  const auto closed = a.borrow(); require(!closed && closed.error().IsFailedPrecondition(), "closed truthy carrier was admitted for reuse");
  const auto closed_initialize = programs.initialize_with(0, initialize_a);
  require(closed_initialize.IsFailedPrecondition() && initialization_calls == std::array<std::size_t, 2>{1, 1},
      "closed truthy carrier evaluated its installation recipe or was admitted as an idempotent live run");
  q::check(a.reset_retired()); require(!a.initialized() && !a.closing(), "actually retired carrier did not reset its original run");
  const auto retired_b = programs.retire_all(backend); q::check(retired_b.status);
  require(retired_b.retired && !programs.pending(), "slot group did not actually retire its original B");
  q::check(b.reset_retired());
  programs.release_invocation_roots();
  const auto teardown_prepare = a.prepare(compiled, backend.queue_bindings());
  const auto teardown_initialize = a.initialize({.io = a_io, .invocation = options});
  const auto teardown_reset = a.reset_retired();
  require(teardown_prepare.IsFailedPrecondition() && teardown_initialize.IsFailedPrecondition() &&
      teardown_reset.IsFailedPrecondition() && !a.initialized() && a.closing() && !a.pending(),
      "irreversible teardown handoff admitted cold initialization or reset its original ownership gate");
  const auto peer_teardown_prepare = b.prepare(compiled, backend.queue_bindings());
  const auto peer_teardown_initialize = programs.initialize_with(1, initialize_b);
  const auto peer_teardown_reset = b.reset_retired();
  require(peer_teardown_prepare.IsFailedPrecondition() && peer_teardown_initialize.IsFailedPrecondition() &&
      peer_teardown_reset.IsFailedPrecondition() && !b.initialized() && b.closing() && !programs.pending(),
      "slot-group teardown left its peer eligible for initialization or ownership reset");
  require(initialization_calls == std::array<std::size_t, 2>{1, 1} && !invalid_initialization_calls,
      "irreversible slot-group handoff evaluated a retired carrier's initialization recipe");
  std::cout << "{\"status\":\"PASS\",\"case\":\"program-carrier\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"models\":0,\"native_submissions\":2,\"carriers\":2,\"cold_failure_atomic\":true,"
         "\"independent_fixed_bindings\":true,\"real_completion_retirement\":true,\"GraphOutput_pin_preserved\":true,"
         "\"typed_io_installation\":true,\"common_slot_inventory\":true,\"group_cold_failure_unpublished\":true,"
         "\"group_initialization_idempotent\":true,\"group_initialization_recipe_lazy\":true,"
         "\"group_retirement_original\":true,\"unsealed_io_rejected\":true,\"foreign_io_rejected\":true,"
         "\"standalone_signature_preserved\":true,\"finite_child_signature_preserved\":true,"
         "\"unused_input_cold_registered\":true,\"full_input_description_preserved\":true,\"foreign_input_identity_rejected\":true,"
         "\"actual_stride_rejected_original_permission_preserved\":true,"
         "\"finite_child_variant_preserved\":true,\"partial_installation_failed\":true,\"partial_caller_writer_tested\":"
      << (caller_writer ? "true" : "false") << ",\"partial_caller_writer_recovered\":" << (caller_writer ? "true" : "null")
      << ",\"valid_io_retry_output_checked\":true,"
         "\"exact_resource_rejection_atomic\":true,\"exact_resource_original_graph_borrowed\":true,"
         "\"pending_observed\":" << (pending_observed ? "true" : "false")
      << ",\"closed_handle_not_reused\":true,\"teardown_handoff_irreversible\":true}\n";
  return 0;
}
} // namespace inference_program_carrier_case
