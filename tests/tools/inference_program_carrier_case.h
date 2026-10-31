#pragma once

#include "../../engine/inference/program_slots.h"
#include "../../engine/inference/prepared_program_family.h"
#include "../../engine/inference/execution_episode.h"

// Two public Graph carriers and, on CUDA, a three-stage Source chain from the
// same concrete program. No Model, synthetic Completion or replacement owner.
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
using Episode = i::ExecutionEpisode<Scope, i::SubmissionViews, 0>;
struct FinishException final { int marker; };

inline std::shared_ptr<Scope> inputs(i::Backend &backend,
    const std::array<float, 4> &first, const std::array<float, 4> &second) {
  const auto ready = [&](const auto &values) {
    auto buffer = q::take(q::mem::Buffer::create(backend.device(), sizeof(values), 256));
    q::check(backend.write(buffer.view(), std::as_bytes(std::span(values))));
    return q::take(backend.adopt_buffer(std::move(buffer)));
  };
  return std::make_shared<Scope>(Scope{ready(first), ready(second)});
}

// The middle original Graph is both a Source consumer and producer. Reuse
// exactly the cold physical tuples; no host result is observed between submits.
inline std::size_t finish_source_chain(i::Backend &backend, const g::CompiledGraph &compiled,
    const g::Graph &graph, g::ValueId first, g::ValueId second, g::ValueId output,
    const c::TensorDesc &description) {
  if (backend.kind() != i::BackendKind::kCuda) return 0;
  require(q::take(i::prepare_output(compiled, output)).caller_owned(), "Source chain needs its original caller output");
  auto scope = inputs(backend, {1, 2, 3, 4}, {10, 20, 30, 40});
  const auto &storage = backend.storage_owner();
  const auto bytes = q::take(description.required_bytes());
  auto pool = q::take(c::LeasePool::create(backend.device(), 3 * 256,
      {.max_blocks = 3, .backing_alignment = 256}));
  std::array<c::ReadLease, 3> cold_ready, retained;
  std::array<q::mem::ConstBufferView, 3> physical;
  for (std::size_t index = 0; index < 3; ++index) {
    auto writer = q::take(pool.allocate(bytes, 256));
    physical[index] = q::take(writer.view());
    q::check(q::take(writer.view()).fill(std::byte{}));
    cold_ready[index] = q::take(writer.publish());
  }
  std::array<i::ProgramIo, 3> io;
  std::array<i::ProgramCarrier<Scope>, 3> carriers;
  std::array<std::array<i::ProgramIo::Input, 2>, 3> ports;
  std::array<i::ProgramIo::Output, 3> outputs;
  std::array<std::shared_ptr<Episode>, 3> episodes;
  struct Cleanup final {
    i::Backend &backend;
    std::array<std::shared_ptr<Episode>, 3> &episodes;
    bool active = true;
    ~Cleanup() {
      if (!active) return;
      for (std::size_t index = episodes.size(); index--;) if (episodes[index] && episodes[index]->run) {
        (void)episodes[index]->retire(backend);
        (void)episodes[index]->recover([]() noexcept {});
      }
    }
  } cleanup{backend, episodes};
  for (std::size_t index = 0; index < 3; ++index) {
    q::check(io[index].prepare(compiled, 0, 2, 0, 1));
    ports[index][0] = q::take(io[index].add_input(first));
    ports[index][1] = q::take(io[index].add_input(second));
    outputs[index] = q::take(io[index].add_output(output)); q::check(io[index].seal());
    q::check(carriers[index].prepare(compiled, backend.queue_bindings()));
    q::check(carriers[index].initialize({.io = io[index], .invocation = {.resources = g::ResourceOptions{}}}));
    episodes[index] = std::make_shared<Episode>();
    episodes[index]->run = q::take(carriers[index].borrow()); episodes[index]->carrier = &carriers[index];
    q::check(io[index].bind(*episodes[index]->run, ports[index][1], scope->second));
  }
  const auto reclaim = [&](std::size_t index) {
    auto writer = q::take(cold_ready[index].try_write());
    const auto actual = q::take(writer.view());
    require(actual.resource() == physical[index].resource() && actual.native_handle() == physical[index].native_handle() &&
        actual.offset() == physical[index].offset() && actual.size() == physical[index].size(),
        "cold Source handoff replaced its original physical writer");
    return writer;
  };
  for (std::size_t index = 3; index--;) {
    auto &episode = *episodes[index];
    q::check(io[index].bind(*episode.run, ports[index][0], index ? cold_ready[index - 1] : scope->first));
    q::check(episode.run->bind(output, description, reclaim(index)));
    q::check(episode.prepare_replay({.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}));
    const auto report = q::take(episode.run->replay_report());
    require(report.route == g::ReplayRoute::kCaptured && report.recorded_capture_range_count &&
        !report.host_wait_dependency_count, "Source chain did not cold-capture its complete tuple");
    if (index) q::check(episode.run->unbind(first));
  }
  for (std::size_t index = 0; index < 3; ++index) {
    auto &episode = *episodes[index];
    if (index) {
      episode.incoming = std::make_unique<i::IncomingExecutionSources<i::SubmissionViews>>();
      episode.incoming->owner = episodes[index - 1]->sources;
      episode.incoming->bindings.push_back(episode.incoming->owner->primary_output.reader_binding(first));
      episode.predecessor_episode = episodes[index - 1];
      q::check(episode.bind_consumers(graph)); q::check(episode.prepare_replay());
      const auto report = q::take(episode.run->replay_report());
      const auto dependency = q::take(episode.run->input_source_report());
      require(report.route == g::ReplayRoute::kCaptured && !report.recorded_capture_range_count &&
          report.reused_capture_range_count && !report.host_wait_dependency_count &&
          dependency.source_count == 1 && !dependency.host_wait_count,
          "Source chain recorded or host-waited on an accepted predecessor");
    }
    if (index != 2) {
      episode.sources = std::make_shared<i::SubmissionViews>();
      q::check(episode.sources->prepare(*episode.run, std::span<const i::SubmissionViews::StoreCapture>{},
          std::span<const i::SubmissionViews::PageCapture>{}, {}));
    }
    q::check(episode.submit(scope));
    if (episode.sources) require(q::take(episode.sources->capture(*episode.run, episode.completion, output, storage, pool)),
        "Source chain did not capture its actual accepted output");
  }
  std::array<std::size_t, 3> completed{}, observed{}, interpreted{};
  const std::array ordinary_second{second};
  for (std::size_t index = 3; index-- > 1;) {
    auto &episode = *episodes[index];
    q::check(episode.wait());
    require(q::take(episode.poll()) == c::CompletionPoll::kComplete &&
        q::take(episode.poll()) == c::CompletionPoll::kComplete && !completed[index] &&
        episode.incoming && episode.incoming->bound && episode.predecessor_episode == episodes[index - 1],
        "repeated Source completion observation invoked callbacks or withdrew its original consumer owners");
    require(q::take(episode.try_finish_success(0, 0, ordinary_second,
        [&]() noexcept { ++completed[index]; },
        [&]() -> hyle::Status {
          require(completed[index] == 1 && !interpreted[index], "Source output observation preceded completion");
          retained[index] = q::take(io[index].retain(*episode.run, outputs[index])); ++observed[index];
          if (index == 1) {
            const auto refused = episode.release_source_bindings();
            require(refused.IsUnavailable() &&
                refused.message() == "source bundle still has native work, results, observations or preparation" &&
                episode.incoming && episode.incoming->bound && episode.predecessor_episode == episodes[0],
                "middle stage released incoming Source before its own producer export detached");
          }
          return {};
        },
        [&]() -> hyle::Status {
          require(observed[index] == 1 && episode.incoming && episode.incoming->bound,
              "Source stage interpreted transactions after withdrawing its original incoming bundle");
          if (index == 1) {
            const auto closed = g::OutputSource::create(episode.sources->submission, output, storage, pool);
            require(!closed && closed.error().IsFailedPrecondition() && closed.error().message() == "submission source export is closed",
                "middle stage interpretation preceded its own producer export detach");
          }
          ++interpreted[index]; return {};
        })), "ready Source stage did not finish through its original nonblocking observation");
    require(!episode.incoming && !episode.predecessor_episode && carriers[index].initialized(),
        "successful Source stage lost its original carrier or retained incoming readers");
  }
  bool original_exception = false;
  try {
    (void)episodes[0]->try_finish_success(0, 0, std::array{first, second},
        [&]() noexcept { ++completed[0]; },
        [&]() -> hyle::Status {
          retained[0] = q::take(io[0].retain(*episodes[0]->run, outputs[0])); ++observed[0];
          throw FinishException{73};
        },
        [&]() -> hyle::Status { ++interpreted[0]; return {}; });
  } catch (const FinishException &failure) { original_exception = failure.marker == 73; }
  require(original_exception && completed == std::array<std::size_t, 3>{1, 1, 1} &&
      observed == std::array<std::size_t, 3>{1, 1, 1} && interpreted == std::array<std::size_t, 3>{0, 1, 1} &&
      episodes[0]->sources && episodes[0]->sources->submission.valid() && retained[0].valid(),
      "model exception changed type or consumed the original unfinished Source resources");
  require(q::take(episodes[0]->recover([]() noexcept {})) && !carriers[0].initialized(),
      "original recovery could not return a stage after its output observer threw");
  for (std::size_t index = 0; index < 3; ++index) {
    std::array<float, 4> actual{}; const auto view = q::take(retained[index].view());
    q::check(view.read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t element = 0; element < actual.size(); ++element)
      require(actual[element] == float(element + 1) * float(11 + 10 * index),
          "Source chain lost actual output values after stage withdrawal or recovery");
    require(view.native_handle() == physical[index].native_handle() && view.offset() == physical[index].offset(),
        "Source chain retained another output backing");
  }
  for (std::size_t index = 3; index--;) if (episodes[index]->run) {
    const auto retired = episodes[index]->retire(backend); q::check(retired.status);
    require(retired.retired && q::take(episodes[index]->recover([]() noexcept {})),
        "Source chain did not recover its actual retired carrier");
  }
  retained = {}; cleanup.active = false; q::check(backend.collect_retired());
  require(pool.stats().live_blocks == 0, "Source chain retained an original output block after cleanup");
  return 3;
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
  Episode a_episode, b_episode;
  a_episode.run = a_run; a_episode.carrier = &a;
  b_episode.run = b_run; b_episode.carrier = &b;
  const std::array ordinary_inputs{first, second};
  std::array<std::size_t, 2> completion_callbacks{}, observations{}, completions{};
  const auto on_a_completion = [&]() noexcept { ++completion_callbacks[0]; };
  const auto observe_a = [&]() -> hyle::Status {
    require(completion_callbacks[0] == 1 && !completions[0],
        "stage observed outputs before original completion or after complete");
    ++observations[0];
    return hyle::Status::InvalidArgument("original stage observation refusal");
  };
  const auto complete_a = [&]() -> hyle::Status { ++completions[0]; return {}; };
  const auto no_completion = a_episode.finish_success(0, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  const auto no_try_completion = a_episode.try_finish_success(0, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  require(no_completion.IsFailedPrecondition() && !no_try_completion &&
      no_try_completion.error().IsFailedPrecondition() && !completion_callbacks[0] && !observations[0] && !completions[0],
      "default success without a Completion entered stage callbacks");
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
  a_episode.completion = std::move(a_done); b_episode.completion = std::move(b_done);
  const auto outside_appends = a_episode.finish_success(1, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  const auto outside_try_appends = a_episode.try_finish_success(1, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  require(outside_appends.IsOutOfRange() && a_episode.completion.valid() &&
      !outside_try_appends && outside_try_appends.error().IsOutOfRange() &&
      !completion_callbacks[0] && !observations[0] && !completions[0],
      "invalid stage append range waited or entered a model callback");
  q::check(a_episode.wait());
  require(q::take(a_episode.poll()) == c::CompletionPoll::kComplete &&
      q::take(a_episode.poll()) == c::CompletionPoll::kComplete && a_episode.completion.valid() &&
      !completion_callbacks[0] && !observations[0] && !completions[0],
      "repeated original Completion observation invoked stage callbacks or discarded its owner");
  const auto try_observation_refused = a_episode.try_finish_success(0, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  require(!try_observation_refused, "nonblocking finish swallowed its original observer refusal");
  const auto observation_refused = try_observation_refused.error();
  const auto repeated_try_refusal = a_episode.try_finish_success(0, 0, ordinary_inputs,
      on_a_completion, observe_a, complete_a);
  require(!repeated_try_refusal && repeated_try_refusal.error().IsFailedPrecondition() &&
      completion_callbacks[0] == 1 && observations[0] == 1 && !completions[0],
      "nonblocking finish retried callbacks after its original partial failure");
  const auto completion_refused = b_episode.finish_success(0, 0, ordinary_inputs,
      [&]() noexcept { ++completion_callbacks[1]; },
      [&]() -> hyle::Status {
        require(completion_callbacks[1] == 1 && !completions[1],
            "stage output observation did not follow original completion");
        ++observations[1]; return {};
      },
      [&]() -> hyle::Status {
        require(observations[1] == 1 && !completions[1],
            "stage completion did not follow its successful output observation");
        ++completions[1];
        return hyle::Status::Unavailable("original stage completion refusal");
      });
  require(observation_refused.IsInvalidArgument() &&
      observation_refused.message() == "original stage observation refusal" && !observation_refused.has_native_error() &&
      completion_refused.IsUnavailable() && completion_refused.message() == "original stage completion refusal" &&
      !completion_refused.has_native_error() && completion_callbacks == std::array<std::size_t, 2>{1, 1} &&
      observations == std::array<std::size_t, 2>{1, 1} && completions == std::array<std::size_t, 2>{0, 1},
      "stage completion replaced a callback Status, repeated a callback or continued after refusal");
  q::check(backend.collect_retired());
  require(!a.pending() && !b.pending() && !programs.pending() && a_lifetime.expired() && b_lifetime.expired(),
      "successful real Completions did not release their own original retained scopes");
  std::array<float, 4> actual_a{}, actual_b{};
  q::check(a_io.read(*a_run, a_readback, std::as_writable_bytes(std::span(actual_a))));
  q::check(b_io.read(*b_run, b_readback, std::as_writable_bytes(std::span(actual_b))));
  require(actual_a == std::array<float, 4>{11, 22, 33, 44} && actual_b == std::array<float, 4>{6, 9, 14, 19},
      "two native carriers mixed fixed inputs or their prepared readbacks");
  a_done = {}; b_done = {};
  // Reuse B's actual bound inputs and original Invocation for one successful
  // try-finish. Natural CUDA scheduling may already have completed this tiny
  // graph; only a real pending observation exercises the no-callback branch.
  auto try_scope = std::make_shared<Scope>();
  std::weak_ptr<Scope> try_lifetime = try_scope;
  std::size_t try_admission = 1, try_pending_observations = 0;
  std::array<std::size_t, 3> try_callbacks{};
  b_episode.candidate_admission = &try_admission;
  q::check(b_episode.prepare_replay({.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  q::check(b_episode.submit(try_scope)); try_scope.reset();
  const auto on_try_completion = [&]() noexcept { ++try_callbacks[0]; };
  const auto observe_try = [&]() -> hyle::Status {
    require(try_callbacks == std::array<std::size_t, 3>{1, 0, 0} && try_admission == 1,
        "nonblocking output observation preceded completion or released original admission");
    q::check(b_io.read(*b_run, b_readback, std::as_writable_bytes(std::span(actual_b))));
    ++try_callbacks[1]; return {};
  };
  const auto complete_try = [&]() -> hyle::Status {
    require(try_callbacks == std::array<std::size_t, 3>{1, 1, 0},
        "nonblocking completion preceded its original output observation");
    ++try_callbacks[2]; return {};
  };
  const auto try_finish = [&] {
    return q::take(b_episode.try_finish_success(0, 0, ordinary_inputs,
        on_try_completion, observe_try, complete_try));
  };
  bool try_finished = try_finish();
  if (!try_finished) {
    ++try_pending_observations;
    require(try_callbacks == std::array<std::size_t, 3>{} && try_admission == 1 &&
        !try_lifetime.expired() && b_episode.completion.valid() &&
        b_episode.run == b_run && b_episode.carrier == &b,
        "pending nonblocking finish invoked callbacks or released its original execution owners");
    try_finished = try_finish();
    if (!try_finished) {
      ++try_pending_observations;
      require(try_callbacks == std::array<std::size_t, 3>{} && try_admission == 1 &&
          !try_lifetime.expired() && b_episode.completion.valid(),
          "repeated pending finish changed callbacks, admission or its original retained owner");
      q::check(b_episode.wait());
      try_finished = try_finish();
    }
  }
  const auto repeated_try_finish = b_episode.try_finish_success(0, 0, ordinary_inputs,
      on_try_completion, observe_try, complete_try);
  require(try_finished && !repeated_try_finish && repeated_try_finish.error().IsFailedPrecondition() &&
      try_callbacks == std::array<std::size_t, 3>{1, 1, 1} && !b_episode.has_completion() &&
      try_admission == 1 && try_lifetime.expired() && b.retired() &&
      actual_b == std::array<float, 4>{6, 9, 14, 19},
      "nonblocking finish repeated a callback, lost output values or changed original retirement/admission");
  b_episode.release_admission();
  require(!try_admission, "nonblocking finish lost its model-owned admission release");
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
  const auto source_submissions = finish_source_chain(backend, compiled, graph, first, second, output, description);
  std::cout << "{\"status\":\"PASS\",\"case\":\"program-carrier\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"models\":0,\"native_submissions\":" << 3 + source_submissions << ",\"carriers\":" << 2 + source_submissions
      << ",\"cold_failure_atomic\":true,"
         "\"independent_fixed_bindings\":true,\"real_completion_retirement\":true,\"GraphOutput_pin_preserved\":true,"
         "\"typed_io_installation\":true,\"common_slot_inventory\":true,\"group_cold_failure_unpublished\":true,"
         "\"group_initialization_idempotent\":true,\"group_initialization_recipe_lazy\":true,"
         "\"group_retirement_original\":true,\"unsealed_io_rejected\":true,\"foreign_io_rejected\":true,"
         "\"standalone_signature_preserved\":true,\"finite_child_signature_preserved\":true,"
         "\"unused_input_cold_registered\":true,\"full_input_description_preserved\":true,\"foreign_input_identity_rejected\":true,"
         "\"actual_stride_rejected_original_permission_preserved\":true,"
         "\"finish_no_completion_rejected_before_callbacks\":true,\"finish_append_range_preflight\":true,"
         "\"finish_completion_observe_complete_order\":true,\"finish_callback_status_resources_preserved\":true,"
         "\"try_finish_original_completion_reobserved\":true,\"try_finish_preflight_before_observation\":true,"
         "\"try_finish_callback_failure_preserved\":true,\"try_finish_callback_order\":true,"
         "\"try_finish_once_without_success_cache\":true,\"try_finish_pending_observations\":" << try_pending_observations << ','
      << "\"finish_source_chain_submissions\":" << source_submissions
      << ",\"finish_middle_incoming_before_own_detach_rejected\":" << (source_submissions ? "true" : "null")
      << ",\"finish_middle_detach_then_release_succeeded\":" << (source_submissions ? "true" : "null")
      << ",\"finish_model_exception_preserved_and_recovered\":" << (source_submissions ? "true" : "null")
      << ","
         "\"finite_child_variant_preserved\":true,\"partial_installation_failed\":true,\"partial_caller_writer_tested\":"
      << (caller_writer ? "true" : "false") << ",\"partial_caller_writer_recovered\":" << (caller_writer ? "true" : "null")
      << ",\"valid_io_retry_output_checked\":true,"
         "\"exact_resource_rejection_atomic\":true,\"exact_resource_original_graph_borrowed\":true,"
         "\"pending_observed\":" << (pending_observed ? "true" : "false")
      << ",\"closed_handle_not_reused\":true,\"teardown_handoff_irreversible\":true}\n";
  return 0;
}
} // namespace inference_program_carrier_case
