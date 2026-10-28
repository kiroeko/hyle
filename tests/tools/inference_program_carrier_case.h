#pragma once

#include "../../engine/inference/program_carrier.h"

// Two public Graph invocations from one concrete program. No Model, synthetic
// Completion, replacement resource owner or test-only execution route.
namespace inference_program_carrier_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace g = hyle::graph;
namespace c = hyle::compute;
using qwen_host_image_case::require;
struct Scope final { c::ReadLease first, second; };

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
  auto builder = q::take(g::GraphBuilder::create());
  const auto first = q::take(builder.input(description));
  const auto second = q::take(builder.input(description));
  const auto output = q::take(builder.add(c::Binary{.kind = c::BinaryKind::kAdd}, first, second));
  q::check(builder.publish(output));
  const auto graph = q::take(std::move(builder).finish());
  const auto compiled = q::take(backend.compile(graph, {}));
  auto a_output = q::take(i::PreparedReadback::prepare(compiled, output, backend));
  auto b_output = q::take(i::PreparedReadback::prepare(compiled, output, backend));
  auto a_scope = inputs(backend, {1, 2, 3, 4}, {10, 20, 30, 40});
  auto b_scope = inputs(backend, {-1, -2, -3, -4}, {7, 11, 17, 23});
  i::ProgramCarrier<Scope> a, b;
  q::check(a.prepare(compiled, backend.queue_bindings())); q::check(b.prepare(compiled, backend.queue_bindings()));
  const g::InvocationOptions options{.resources = g::ResourceOptions{}};
  const auto partial = a.initialize(options, [&](g::Invocation &candidate) -> hyle::Status {
    HYLE_RETURN_IF_ERROR(candidate.bind(first, description, a_scope->first));
    return candidate.bind(second, bad_description, a_scope->second);
  });
  require(partial.IsInvalidArgument() && !a.initialized() && !a.pending() && !b.initialized(),
      "cold bind failure published a half-initialized carrier or touched its peer");
  const auto missing = a.borrow();
  require(!missing && missing.error().IsFailedPrecondition(), "failed cold carrier admitted an original invocation");
  q::check(b.initialize(options, [&](g::Invocation &candidate) -> hyle::Status {
    HYLE_RETURN_IF_ERROR(candidate.bind(first, description, b_scope->first));
    HYLE_RETURN_IF_ERROR(candidate.bind(second, description, b_scope->second));
    return b_output.bind(candidate);
  }));
  require(b.initialized() && !a.initialized(), "initializing B repaired or published failed A");
  q::check(a.initialize(options, [&](g::Invocation &candidate) -> hyle::Status {
    HYLE_RETURN_IF_ERROR(candidate.bind(first, description, a_scope->first));
    HYLE_RETURN_IF_ERROR(candidate.bind(second, description, a_scope->second));
    return a_output.bind(candidate);
  }));
  auto *a_run = q::take(a.borrow()); auto *b_run = q::take(b.borrow());
  require(a_run != b_run && a_run->valid() && b_run->valid(), "independent carriers share one original mutable invocation");
  bool overwritten = false;
  const auto duplicate = a.initialize(options, [&](g::Invocation &) -> hyle::Status { overwritten = true; return hyle::Status::Ok(); });
  require(duplicate.IsFailedPrecondition() && !overwritten && q::take(a.borrow()) == a_run,
      "already initialized carrier called a binder or replaced its live original invocation");
  q::check(a_run->prepare_replay(a.queues(), {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  q::check(b_run->prepare_replay(b.queues(), {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  // A missing scope must reject before native submit, preserving the actual
  // prepared run. It is not evidence of an asynchronously pending device.
  const auto no_scope = a.submit({});
  require(!no_scope && no_scope.error().IsUnavailable() && !a.pending() && q::take(a.borrow()) == a_run,
      "rejected resource reservation submitted work or lost its original prepared run");
  std::weak_ptr<Scope> a_lifetime = a_scope, b_lifetime = b_scope;
  auto a_done = q::take(a.submit(a_scope)); a_scope.reset();
  bool pending_observed = a.pending();
  const auto a_admission = a.borrow();
  require(a_admission ? !a.pending() && !a_run->in_flight() : a_admission.error().IsUnavailable(),
      "carrier admitted reuse before original native retirement");
  auto b_done = q::take(b.submit(b_scope)); b_scope.reset();
  pending_observed = pending_observed || b.pending();
  const auto b_admission = b.borrow();
  require(b_admission ? !b.pending() && !b_run->in_flight() : b_admission.error().IsUnavailable(),
      "peer carrier admitted reuse before its own original native retirement");
  q::check(a_done.wait()); q::check(b_done.wait()); q::check(backend.collect_retired());
  require(!a.pending() && !b.pending() && a_lifetime.expired() && b_lifetime.expired(),
      "successful real Completions did not release their own original retained scopes");
  std::array<float, 4> actual_a{}, actual_b{};
  q::check(a_output.read(backend, *a_run, std::as_writable_bytes(std::span(actual_a))));
  q::check(b_output.read(backend, *b_run, std::as_writable_bytes(std::span(actual_b))));
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
  q::check(a.reset_retired()); require(!a.initialized() && !a.closing(), "actually retired carrier did not reset its original run");
  const auto retired_b = b.retire(backend); q::check(retired_b.status); require(retired_b.retired, "B did not actually retire");
  q::check(b.reset_retired());
  a.release_invocation_root();
  bool teardown_binder_called = false;
  const auto teardown_prepare = a.prepare(compiled, backend.queue_bindings());
  const auto teardown_initialize = a.initialize(options, [&](g::Invocation &) -> hyle::Status {
    teardown_binder_called = true; return hyle::Status::Ok();
  });
  const auto teardown_reset = a.reset_retired();
  require(teardown_prepare.IsFailedPrecondition() && teardown_initialize.IsFailedPrecondition() &&
      teardown_reset.IsFailedPrecondition() && !teardown_binder_called && !a.initialized() && a.closing() && !a.pending(),
      "irreversible teardown handoff admitted cold initialization or reset its original ownership gate");
  std::cout << "{\"status\":\"PASS\",\"case\":\"program-carrier\",\"backend\":\"" << qwen_host_image_case::backend_name(kind)
      << "\",\"models\":0,\"native_submissions\":2,\"carriers\":2,\"cold_failure_atomic\":true,"
         "\"independent_fixed_bindings\":true,\"real_completion_retirement\":true,\"GraphOutput_pin_preserved\":true,"
         "\"pending_observed\":" << (pending_observed ? "true" : "false")
      << ",\"closed_handle_not_reused\":true,\"teardown_handoff_irreversible\":true}\n";
  return 0;
}
} // namespace inference_program_carrier_case
