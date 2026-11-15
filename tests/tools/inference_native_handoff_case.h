#pragma once

#include "../../engine/inference/backend.h"
#include "../../engine/inference/prepared_output.h"
#include "hyle/graph/source.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace inference_native_handoff_case {
namespace i = hyle::inference;
namespace g = hyle::graph;
namespace c = hyle::compute;
namespace m = hyle::mem;

inline void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
inline void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) check(result.error());
  return std::move(*result);
}

// Two real original native Graphs, with an exact caller-owned output feeding
// the next Graph. No model fixture, host token roundtrip or synthetic Queue.
inline int run(i::BackendKind kind) {
  auto backend = take(i::Backend::create({.primary = kind}));
  require(backend.queue_bindings().empty(), "native handoff unexpectedly has a caller Queue");
  const auto owner = backend.storage_owner();
  require(owner && owner.endpoint_identity() && !owner.context_identity(),
      "native storage must retain its actual endpoint without a Compute Context");
  const auto output_owner = take(backend.storage_owner_for(backend.staging()));
  require(output_owner && output_owner.endpoint_identity() == owner.endpoint_identity(),
      "native staging must retain the same actual endpoint");
  g::StorageDomainId output_storage;
  const auto domains = backend.storage_domains();
  for (std::size_t index = 0; index < domains.size(); ++index)
    if (domains[index].resource == output_owner.ref()) output_storage = g::StorageDomainId{static_cast<std::uint32_t>(index)};
  require(output_storage.valid(), "original native output allocation domain is missing");
  const auto description = take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}));
  auto builder = take(g::GraphBuilder::create());
  const auto first = take(builder.input(description));
  const auto second = take(builder.input(description));
  const auto sum = take(builder.add(c::Binary{.kind = c::BinaryKind::kAdd}, first, second));
  const auto output = take(builder.add(c::Copy{}, sum));
  check(builder.publish(output));
  const auto graph = take(std::move(builder).finish());
  // The original native transfer capability materializes only this small
  // external result. Its math stays in the primary GPU allocation domain.
  const std::array external{g::ExternalResourceContract{output, output_storage},
      g::ExternalResourceContract{first, output_storage}};
  const auto compiled = take(backend.compile(graph, {}, {.external_resources = external}));
  require(!compiled.executable.required_queue_count() &&
      take(i::prepare_output(compiled, output)).caller_owned(),
      "native Source requires its original queue-free Graph and full caller writer");
  auto producer = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  auto consumer = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
  const auto ready = [&](const c::StorageOwner &selected, const std::array<float, 4> &values) {
    auto bytes = take(m::Buffer::create(selected.ref(), sizeof(values), 256));
    check(bytes.view().write(std::as_bytes(std::span(values))));
    return take(selected.adopt_buffer(std::move(bytes)));
  };
  auto left = ready(output_owner, {1, 2, 3, 4}), right = ready(owner, {10, 20, 30, 40});
  auto pool = take(c::LeasePool::create(output_owner.ref(), 512, {.max_blocks = 2, .backing_alignment = 256}));
  auto first_writer = take(pool.allocate(16, 256)), second_writer = take(pool.allocate(16, 256));
  const auto original = take(first_writer.view());
  check(producer.bind(first, description, left));
  check(producer.bind(second, description, right));
  check(consumer.bind(second, description, right));
  check(producer.bind(output, description, std::move(first_writer)));
  check(consumer.bind(output, description, std::move(second_writer)));
  check(producer.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto submitted = take(producer.submit());
  const auto no_pending_dependency = g::SubmissionSource::create(producer, std::move(submitted), owner);
  require(!no_pending_dependency && no_pending_dependency.error().IsUnimplemented() && submitted.valid(),
      "native runtime masqueraded as a shared-Queue pending dependency");
  auto foreign = take(i::Backend::create({.primary = kind}));
  const auto wrong_owner = g::SubmissionSource::create_handoff(producer, std::move(submitted), foreign.storage_owner());
  require(!wrong_owner && submitted.valid(), "native handoff accepted a foreign metadata resource or consumed on rejection");
  auto source = take(g::SubmissionSource::create_handoff(producer, std::move(submitted), owner));
  require(!g::OutputSource::create(source, output, owner, pool),
      "native output accepted the allocation owner of another resource");
  auto view = take(g::OutputSource::create(source, output, output_owner, pool));
  const std::array incoming{view.reader_binding(first)};
  check(g::bind_sources(consumer, incoming));
  check(g::prepare_sources(consumer, {}));
  const auto report = take(consumer.input_source_report());
  require(report.source_count == 1 && report.host_wait_count == 1 && !report.device_wait_count,
      "native handoff did not report its actual whole-producer dependency");
  // Ordinary submit has no authority for this pending writer. Reject before
  // accepting native work rather than silently treating its address as ready.
  require(!consumer.submit(), "ordinary submit bypassed native source permission admission");
  auto consumed = take(g::submit_sources(consumer, {}));
  require(!source.detach().ok(), "native handoff detached a still-borrowed original writer");
  check(consumed.wait());
  consumed = {};
  auto result = take(consumer.result(output));
  auto retained = take(result.storage_lease());
  result.reset();
  std::array<float, 4> actual{};
  check(take(retained.view()).read(std::as_writable_bytes(std::span(actual))));
  require(actual == std::array<float, 4>{21, 42, 63, 84}, "native handoff lost original GPU output values");
  check(view.detach());
  check(g::release_sources(consumer));
  check(source.detach());
  require(take(source.poll()) == c::CompletionPoll::kComplete && producer.replay_route() == g::ReplayRoute::kCommands,
      "native handoff changed its original whole verdict or execution route");
  auto first_result = take(producer.result(output));
  const auto first_ready = take(first_result.storage_lease());
  const auto address = take(first_ready.view());
  require(address.resource() == original.resource() && address.native_handle() == original.native_handle() &&
      address.offset() == original.offset() && address.size() == original.size(),
      "native handoff copied or replaced the original producer storage");

  // A real checked gather fails before the tail transfer may publish bytes.
  // Retain the same original whole verdict, then reuse the finite program
  // with a fresh Invocation, as required by original Graph failure recovery.
  const auto index_description = take(c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}));
  const auto gathered_description = take(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{1}));
  auto failure_builder = take(g::GraphBuilder::create());
  const auto data = take(failure_builder.input(description));
  const auto indices = take(failure_builder.input(index_description));
  const auto gathered = take(failure_builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, data, indices));
  check(failure_builder.publish(gathered));
  const auto failure_graph = take(std::move(failure_builder).finish());
  const std::array failure_external{g::ExternalResourceContract{gathered, output_storage}};
  const auto failure_program = take(backend.compile(failure_graph, {}, {.external_resources = failure_external}));
  auto failed = take(failure_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  auto failure_pool = take(c::LeasePool::create(output_owner.ref(), 256, {.max_blocks = 1}));
  const auto index = [&](std::int32_t value) {
    auto bytes = take(m::Buffer::create(owner.ref(), sizeof(value), 256));
    check(bytes.view().write(std::as_bytes(std::span(&value, 1))));
    return take(owner.adopt_buffer(std::move(bytes)));
  };
  auto bad_index = index(-1);
  check(failed.bind(data, description, right));
  check(failed.bind(indices, index_description, bad_index));
  check(failed.bind(gathered, gathered_description, take(failure_pool.allocate(sizeof(float)))));
  check(failed.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto failed_completion = take(failed.submit());
  auto failed_source = take(g::SubmissionSource::create_handoff(failed, std::move(failed_completion), owner));
  auto failed_view = g::OutputSource::create(failed_source, gathered, output_owner, failure_pool);
  const auto failure = failed_source.wait();
  require(failure.IsOutOfRange() || failure.IsInvalidArgument(), "checked native gather lost its original failure");
  require(!failed.result(gathered), "failed native Graph published its incomplete output");
  if (failed_view) {
    require(!failed_view->detach().ok(), "failed native output became a ready input");
    failed_view = g::OutputSource{};
  }
  check(failed_source.detach());
  const auto retained_failure = failed_source.wait();
  require(retained_failure.code() == failure.code() && retained_failure.message() == failure.message(),
      "native Source detach replaced the original whole failure verdict");
  require(!failed.prepare_replay({}, {.mode = g::ReplayMode::kCommands}).ok(),
      "failed original Invocation was silently reused");
  failed_source = {};
  failed = {};
  auto retry = take(failure_program.executable.create_invocation({.resources = g::ResourceOptions{}}));
  check(retry.bind(data, description, right));
  check(retry.bind(gathered, gathered_description, take(failure_pool.allocate(sizeof(float)))));
  check(retry.bind(indices, index_description, index(0)));
  check(retry.prepare_replay({}, {.mode = g::ReplayMode::kCommands}));
  auto retried = take(retry.submit());
  check(retried.wait());
  retried = {};
  auto retry_output = take(retry.result(gathered));
  auto retry_ready = take(retry_output.storage_lease());
  float retry_value = 0;
  check(take(retry_ready.view()).read(std::as_writable_bytes(std::span(&retry_value, 1))));
  require(retry_value == 10, "retired native failure did not recover its original finite program");
  std::cout << "PASS native_handoff caller_queues=0 original_graphs=2 original_output_storage=1"
      " whole_success_and_retirement=1 ordinary_submit_rejected=1 values=21,42,63,84"
      " checked_native_failure=1 failed_output_not_ready=1 same_program_retry=1\n" << std::flush;
  return 0;
}
} // namespace inference_native_handoff_case
