#pragma once

#include "../../engine/inference/program_io.h"
#include "../../engine/inference/readback_batch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HYLE_COMPUTE_CUDA)
#include "hyle/compute/cuda.h"
#endif

// Two original Graph results, two dtypes, one prepared CopyBatch phase. This
// component consumer needs no model or replacement execution/completion owner.
namespace inference_readback_batch_case {
namespace q = qwen35;
namespace i = hyle::inference;
namespace g = hyle::graph;
namespace c = hyle::compute;
using qwen_host_image_case::require;

inline auto report_fields(const c::CopyBatchReport &r) noexcept {
  return std::tuple{r.resources, r.copies, r.active_copies, r.phases, r.no_op_copies,
      r.logical_bytes, r.physical_bytes, r.staging_bytes, r.metadata_bytes,
      r.native_calls, r.native_waits, r.phase_completions, r.host_waits,
      r.native_metadata_unknown};
}

inline void write_report(const c::CopyBatchReport &r) {
  std::cout << "{\"resources\":" << r.resources << ",\"copies\":" << r.copies
      << ",\"active_copies\":" << r.active_copies << ",\"phases\":" << r.phases
      << ",\"no_op_copies\":" << r.no_op_copies << ",\"logical_bytes\":" << r.logical_bytes
      << ",\"physical_bytes\":" << r.physical_bytes << ",\"staging_bytes\":" << r.staging_bytes
      << ",\"metadata_bytes\":" << r.metadata_bytes << ",\"native_calls\":" << r.native_calls
      << ",\"native_waits\":" << r.native_waits << ",\"phase_completions\":" << r.phase_completions
      << ",\"host_waits\":" << r.host_waits << ",\"native_metadata_unknown\":"
      << (r.native_metadata_unknown ? "true" : "false") << '}';
}

struct Destination final {
  static constexpr std::byte canary{0xa5};
  std::array<std::byte, sizeof(std::int32_t) + 8> id;
  std::array<std::byte, 4 * sizeof(float) + 8> predicate;

  void clear() noexcept { id.fill(canary); predicate.fill(canary); }
  std::array<std::span<std::byte>, 2> spans() noexcept {
    return {std::span(id).subspan(4, sizeof(std::int32_t)),
        std::span(predicate).subspan(4, 4 * sizeof(float))};
  }
  bool untouched() const noexcept {
    const auto same = [](std::byte b) { return b == canary; };
    return std::ranges::all_of(id, same) && std::ranges::all_of(predicate, same);
  }
  void check(std::int32_t expected_id, const std::array<float, 4> &expected_predicate) const {
    const auto guard = [](const auto &bytes) {
      for (std::size_t j = 0; j < 4; ++j)
        if (bytes[j] != canary || bytes[bytes.size() - 1 - j] != canary) return false;
      return true;
    };
    require(guard(id) && guard(predicate), "batched readback overwrote an exact output range");
    std::int32_t actual_id = 0;
    std::array<float, 4> actual_predicate{};
    std::memcpy(&actual_id, id.data() + 4, sizeof(actual_id));
    std::memcpy(actual_predicate.data(), predicate.data() + 4, sizeof(actual_predicate));
    require(actual_id == expected_id && actual_predicate == expected_predicate,
        "batched readback changed an I32/F32 result or reused a previous replay's data");
  }
};

#if defined(HYLE_COMPUTE_CUDA)
// Same original-Queue host-callback instrument as batch_probe. It contributes
// no Graph Completion or retirement verdict. The watchdog opens a failed test
// after 30 seconds; every normal and exceptional exit releases before waiting.
class NativeGate final {
public:
  NativeGate() = default;
  NativeGate(const NativeGate &) = delete;
  NativeGate &operator=(const NativeGate &) = delete;
  ~NativeGate() {
    release();
    if (watchdog_.joinable()) watchdog_.join();
    if (completion_.valid()) completion_.wait().IgnoreError();
  }
  void arm(i::Backend &backend) {
    arm(*q::take(backend.common_queue()));
  }
  void arm(c::Queue &queue) {
    watchdog_ = std::thread([this] {
      std::unique_lock lock(mutex_);
      if (!changed_.wait_for(lock, std::chrono::seconds(30), [&] { return open_; })) {
        expired_ = true; open_ = true; changed_.notify_all();
      }
    });
    completion_ = q::take(c::cuda::submit(queue,
        [&](cudaStream_t stream) noexcept { return cudaLaunchHostFunc(stream, &wait, this); }));
    std::unique_lock lock(mutex_);
    require(changed_.wait_for(lock, std::chrono::seconds(30), [&] { return entered_ || expired_; }) &&
        entered_ && !expired_, "readback Queue did not enter its original native gate");
  }
  void release() noexcept {
    { std::lock_guard lock(mutex_); open_ = true; }
    changed_.notify_all();
  }
  void finish() {
    release(); q::check(completion_.wait());
    std::lock_guard lock(mutex_);
    require(!expired_, "readback native gate watchdog opened during an assertion");
  }
  bool blocked() noexcept {
    std::lock_guard lock(mutex_);
    return entered_ && !open_ && !expired_;
  }
private:
  static void CUDART_CB wait(void *opaque) {
    auto &self = *static_cast<NativeGate *>(opaque);
    std::unique_lock lock(self.mutex_);
    self.entered_ = true; self.changed_.notify_all();
    self.changed_.wait(lock, [&] { return self.open_; }); // No CUDA calls here.
  }
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread watchdog_;
  c::Completion completion_;
  bool entered_ = false, open_ = false, expired_ = false;
};

#if defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
struct ObservationStatusSeam final {
  NativeGate gate;
  c::Queue *original_queue = nullptr;
  std::size_t submit_hooks = 0, wait_hooks = 0;
  hyle::Status failure = hyle::Status::FromNative(hyle::StatusCode::Unavailable,
      {hyle::NativeErrorDomain::Unknown, 207}, "injected readback observation status; native copy still pending");

  static hyle::Status before_submit(void *opaque, c::Queue &queue) noexcept {
    auto &self = *static_cast<ObservationStatusSeam *>(opaque);
    try {
      ++self.submit_hooks; self.original_queue = &queue;
      self.gate.arm(queue);
      return {};
    } catch (...) {
      self.gate.release();
      return hyle::Status::Internal("readback observation seam could not arm its original native Queue");
    }
  }
  static hyle::Status observe(void *opaque, c::CopyBatchUpdate &update) noexcept {
    auto &self = *static_cast<ObservationStatusSeam *>(opaque);
    ++self.wait_hooks;
    if (!self.gate.blocked() || !update.in_flight())
      return hyle::Status::Internal("readback observation seam lost its real pending native copy");
    // Deliberately do not call update.wait(). This is an observation-status
    // seam on a real pending copy, never a physical CUDA driver failure.
    return self.failure;
  }
};
#endif
#endif

inline int run(i::BackendKind kind) {
  require(kind == i::BackendKind::kCpu || kind == i::BackendKind::kCuda,
      "readback batch component case requires CPU or CUDA");
  auto backend = q::take(i::Backend::create({.primary = kind}));
  const auto table_desc = q::dense(q::Type::kInt32, {4});
  const auto index_desc = q::dense(q::Type::kInt32, {1});
  const auto predicate_desc = q::dense(q::Type::kFloat32, {4});
  auto table_storage = q::take(q::mem::Buffer::create(backend.device(), 4 * sizeof(std::int32_t), 256));
  auto index_storage = q::take(q::mem::Buffer::create(backend.device(), sizeof(std::int32_t), 256));
  auto predicate_storage = q::take(q::mem::Buffer::create(backend.device(), 4 * sizeof(float), 256));
  auto builder = q::take(g::GraphBuilder::create());
  const auto table = q::take(builder.input(table_desc));
  const auto index = q::take(builder.input(index_desc));
  const auto predicate_input = q::take(builder.input(predicate_desc));
  const auto selected = q::take(builder.add(c::Gather{.axis = 0}, table, index));
  const auto predicate = q::take(builder.add(c::Unary{.kind = c::UnaryKind::kAbs}, predicate_input));
  q::check(builder.publish(selected)); q::check(builder.publish(predicate));
  const auto graph = q::take(std::move(builder).finish());
  const auto compiled = q::take(backend.compile(graph, {}));
  i::ProgramIo io;
  q::check(io.prepare(compiled, 3, 0, 2, 0));
  q::check(io.add_fixed(table, q::take(c::ConstTensorView::create(table_desc, table_storage.view()))));
  q::check(io.add_fixed(index, q::take(c::ConstTensorView::create(index_desc, index_storage.view()))));
  q::check(io.add_fixed(predicate_input, q::take(c::ConstTensorView::create(predicate_desc, predicate_storage.view()))));
  const auto id_port = q::take(io.add_readback(selected, backend.storage_domains()));
  const auto predicate_port = q::take(io.add_readback(predicate, backend.storage_domains()));
  q::check(io.seal());
  const std::array<const i::PreparedReadback *, 2> ports{
      q::take(io.readback(id_port)), q::take(io.readback(predicate_port))};
  require(ports[0]->facts.caller_owned() && ports[1]->facts.caller_owned() &&
      ports[0]->facts.description.data_type() == q::Type::kInt32 &&
      ports[1]->facts.description.data_type() == q::Type::kFloat32,
      "readback batch fixture did not retain two original caller-owned dtypes");
  auto batch = q::take(i::PreparedReadbackBatch::prepare(backend, ports));
  require(batch.valid() && !batch.in_flight(), "cold readback batch has no reusable original copy slot");
  auto invocation = q::take(compiled.executable.create_invocation());
  q::check(io.bind_fixed(invocation));
  struct Cleanup final {
    i::Backend &backend;
    i::PreparedReadbackBatch &batch;
    g::Invocation &invocation;
    ~Cleanup() {
      (void)batch.recover();
      invocation.drain().IgnoreError();
      backend.collect_retired().IgnoreError();
    }
  } cleanup{backend, batch, invocation};

  Destination destination; destination.clear();
  auto destinations = destination.spans();
  const auto before_unsubmitted = batch.report();
  const auto unsubmitted = batch.read(invocation, destinations);
  require(!unsubmitted.ok() && destination.untouched() &&
      batch.report().native_calls == before_unsubmitted.native_calls &&
      batch.report().phase_completions == before_unsubmitted.phase_completions,
      "readback copied or published a Graph result without original successful completion");
  auto recovered = batch.recover();
  require(recovered.retired && recovered.status == unsubmitted && !batch.in_flight(),
      "unsubmitted readback refusal did not preserve its original error and retire its copy slot");

  std::uint64_t native_calls = 0, native_waits = 0, phase_completions = 0, host_waits = 0;
  std::vector<c::CopyBatchReport> copy_reports;
  std::size_t graph_submissions = 0, batch_reads = 0, invalid_targets = 0, pending_rejections = 0;
  std::size_t prime_graph_submissions = 0;
  for (std::size_t replay = 0; replay < 2; ++replay) {
    const std::array<std::int32_t, 4> values = replay
        ? std::array<std::int32_t, 4>{99, -20'000'001, -2'147'483'600, 7}
        : std::array<std::int32_t, 4>{16'777'217, 2'147'483'600, -19, 88};
    const std::array<std::int32_t, 1> chosen{replay ? 2 : 1};
    const std::array<float, 4> input = replay ? std::array<float, 4>{-1, 0, 1, 0}
                                            : std::array<float, 4>{0, -1, 0, 1};
    const std::array<float, 4> expected = replay ? std::array<float, 4>{1, 0, 1, 0}
                                               : std::array<float, 4>{0, 1, 0, 1};
    q::check(backend.write(table_storage.view(), std::as_bytes(std::span(values))));
    q::check(backend.write(index_storage.view(), std::as_bytes(std::span(chosen))));
    q::check(backend.write(predicate_storage.view(), std::as_bytes(std::span(input))));
    q::check(invocation.prepare_replay(backend.queue_bindings(), {.mode = g::ReplayMode::kCommands}));

#if defined(HYLE_COMPUTE_CUDA)
    if (kind == i::BackendKind::kCuda && replay == 0) {
      // The first actual producer submission may materialize native provider
      // state. Finish that work before closing this Context's Queue gate; its
      // real submission is included in the reported Graph count.
      auto prime = q::take(invocation.submit(backend.queue_bindings()));
      ++graph_submissions; ++prime_graph_submissions;
      q::check(prime.wait()); prime = {};
      q::check(invocation.prepare_replay(backend.queue_bindings(), {.mode = g::ReplayMode::kCommands}));
    }
    NativeGate gate;
    if (kind == i::BackendKind::kCuda) gate.arm(backend);
#endif
    auto completion = q::take(invocation.submit(backend.queue_bindings())); ++graph_submissions;
#if defined(HYLE_COMPUTE_CUDA)
    if (kind == i::BackendKind::kCuda) {
      const auto observed = q::take(completion.poll());
      require(observed == c::CompletionPoll::kPending && invocation.in_flight(),
          "readback gate did not preserve real pending original Graph work");
      destination.clear(); const auto before = batch.report();
      const auto status = batch.read(invocation, destinations);
      require(!status.ok() && destination.untouched() && batch.report().native_calls == before.native_calls &&
          batch.report().phase_completions == before.phase_completions &&
          batch.report().host_waits == before.host_waits && invocation.in_flight(),
          "pending readback waited, copied, published or retired original Graph work");
      ++pending_rejections;
      recovered = batch.recover();
      require(recovered.retired && recovered.status == status && !batch.in_flight(),
          "pending Graph readback refusal changed its original error or retained a copy reader");
      gate.finish();
    }
#endif
    q::check(completion.wait()); completion = {};
    require(!invocation.in_flight(), "readback producer did not really retire after successful wait");

    // All destination validation precedes native copy setup or any host write.
    for (std::size_t invalid = 0; invalid < 3; ++invalid) {
      destination.clear(); auto wrong = destinations;
      if (invalid == 0) wrong[0] = wrong[0].first(sizeof(std::int32_t) - 1);
      if (invalid == 1) wrong[1] = std::span(destination.predicate).subspan(4, 4 * sizeof(float) + 1);
      const auto before = batch.report();
      const auto target = invalid == 2 ? std::span(wrong).first(1) : std::span(wrong);
      const auto status = batch.read(invocation, target);
      require(status.IsInvalidArgument() && destination.untouched() && !batch.in_flight() &&
          report_fields(batch.report()) == report_fields(before),
          "invalid readback extent/count reached native copy or changed the original report");
      ++invalid_targets;
    }
    // A repeated observation explicitly performs another original batch; the
    // report counts it rather than implying a result cache or free transport.
    for (std::size_t observation = 0; observation < (replay ? 1u : 2u); ++observation) {
      destination.clear(); q::check(batch.read(invocation, destinations)); ++batch_reads;
      destination.check(values[static_cast<std::size_t>(chosen[0])], expected);
      require(!batch.in_flight(), "successful readback kept its source CopyBatch readers in flight");
      const auto report = batch.report();
      require(report.copies == 2 && report.active_copies == 2 && report.phases == 1 &&
          report.phase_completions == 1 && report.logical_bytes == sizeof(std::int32_t) + 4 * sizeof(float) &&
          report.native_calls >= 1 && report.host_waits <= 1,
          "readback report does not describe its two real copies and one completion phase");
      copy_reports.push_back(report);
      native_calls += report.native_calls; native_waits += report.native_waits;
      phase_completions += report.phase_completions; host_waits += report.host_waits;
      recovered = batch.recover(); q::check(recovered.status);
      require(recovered.retired && !batch.in_flight(), "completed readback recovery changed real retirement");
    }
    // The next replay reuses Graph's original output writers. An escaped
    // CopyBatch source reader would reject their promotion before submission.
  }
  q::check(backend.collect_retired());
  require(graph_submissions == 2 + prime_graph_submissions &&
      prime_graph_submissions == (kind == i::BackendKind::kCuda ? 1u : 0u) &&
      batch_reads == 3 && invalid_targets == 6 &&
      pending_rejections == (kind == i::BackendKind::kCuda ? 2u : 0u),
      "readback finite consumer changed the number of original operations");

  std::size_t observation_status_failures = 0, late_retirements = 0, pending_abandons = 0;
  std::size_t source_writer_refusals = 0, late_graph_replays = 0;
#if defined(HYLE_COMPUTE_CUDA) && defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
  if (kind == i::BackendKind::kCuda) {
    const std::array limits{c::ResourceBudgetLimit{backend.device(), 8u << 20},
        c::ResourceBudgetLimit{backend.staging(), 8u << 20},
        c::ResourceBudgetLimit{q::mem::system_resource(), 8u << 20}};
    auto budget = q::take(c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
    const auto metadata_live = [&] { return q::take(budget.usage(q::mem::system_resource())).live_bytes; };
    const auto budget_control = metadata_live();
    for (const bool abandon : {false, true}) {
      auto late = q::take(i::PreparedReadbackBatch::prepare(backend, ports, budget));
      const auto cold_metadata = metadata_live();
      require(cold_metadata > budget_control, "late readback scope/copy metadata was not tracked by its original budget");
      ObservationStatusSeam seam;
      destination.clear();
      hyle::Status observed;
      {
        i::testing::ReadbackBatchHookScope hook({&seam,
            &ObservationStatusSeam::before_submit, &ObservationStatusSeam::observe});
        observed = late.read(invocation, destinations);
      }
      require(observed == seam.failure && observed.has_native_error() && destination.untouched() &&
          seam.submit_hooks == 1 && seam.wait_hooks == 1 && seam.original_queue && seam.gate.blocked() &&
          late.in_flight(), "observation-status refusal changed its error or published a genuinely pending copy");
      ++observation_status_failures;
      const auto pending_report = late.report();
      copy_reports.push_back(pending_report);
      native_calls += pending_report.native_calls; native_waits += pending_report.native_waits;
      phase_completions += pending_report.phase_completions; host_waits += pending_report.host_waits;
      const auto pending = late.recover();
      require(!pending.retired && pending.status == seam.failure && late.in_flight() &&
          destination.untouched() && seam.gate.blocked(),
          "observation-status error was treated as native retirement or host publication");
      auto *original_id = q::take(io.readback(id_port));
      const auto writer_busy = original_id->reclaim(invocation);
      require(writer_busy.IsUnavailable(), "pending readback allowed the original Graph output writer to be reused");
      ++source_writer_refusals;

      if (abandon) {
        // The gate remains shut while the application drops its entire batch.
        // Returning before the watchdog opens proves abandonment did not wait;
        // live charged metadata proves original native work still pins Scope.
        late = {};
        require(!late.valid() && seam.gate.blocked() && metadata_live() >= cold_metadata,
            "pending batch destruction waited or released its original Scope/Queue metadata early");
        ++pending_abandons;
      }
      seam.gate.finish();
      q::check(q::take(backend.common_context())->drain());
      q::check(backend.collect_retired());
      if (!abandon) {
        const auto retired = late.recover();
        require(retired.retired && retired.status == seam.failure && !late.in_flight() && destination.untouched(),
            "late original copy retirement changed its observation error or published rejected host data");
        late = {};
      }
      // Abandoned completions use CUDA's existing reaper. Stream drain proves
      // native completion; allow its separate host callback to release Scope.
      const auto release_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (metadata_live() != budget_control && std::chrono::steady_clock::now() < release_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      require(metadata_live() == budget_control,
          "late original copy retirement retained its abandoned Scope/Queue or charged metadata");
      ++late_retirements;
      q::check(original_id->reclaim(invocation)); q::check(original_id->bind(invocation));

      // After late retirement, promote the exact original Graph writer and
      // produce different contents. This also proves its Copy source readers
      // ended, rather than merely hiding the failed batch's observation root.
      const std::array<std::int32_t, 4> values{abandon ? -77 : 55, 19, 23, 29};
      const std::array<std::int32_t, 1> chosen{0};
      const std::array<float, 4> input = abandon ? std::array<float, 4>{0, -1, 0, -1}
                                               : std::array<float, 4>{-1, -1, 0, 0};
      const std::array<float, 4> expected = abandon ? std::array<float, 4>{0, 1, 0, 1}
                                                  : std::array<float, 4>{1, 1, 0, 0};
      q::check(backend.write(table_storage.view(), std::as_bytes(std::span(values))));
      q::check(backend.write(index_storage.view(), std::as_bytes(std::span(chosen))));
      q::check(backend.write(predicate_storage.view(), std::as_bytes(std::span(input))));
      q::check(invocation.prepare_replay(backend.queue_bindings(), {.mode = g::ReplayMode::kCommands}));
      auto completion = q::take(invocation.submit(backend.queue_bindings())); q::check(completion.wait());
      completion = {}; ++late_graph_replays; ++graph_submissions;
      destination.clear(); q::check(batch.read(invocation, destinations)); destination.check(values[0], expected);
      ++batch_reads;
      const auto replay_report = batch.report();
      copy_reports.push_back(replay_report);
      native_calls += replay_report.native_calls; native_waits += replay_report.native_waits;
      phase_completions += replay_report.phase_completions; host_waits += replay_report.host_waits;
    }
    require(observation_status_failures == 2 && late_retirements == 2 && pending_abandons == 1 &&
        source_writer_refusals == 2 && late_graph_replays == 2,
        "late readback consumer changed its finite failure/recovery/abandonment coverage");
  }
#endif
  q::check(invocation.drain()); q::check(backend.collect_retired());
  std::cout << "{\"status\":\"PASS\",\"case\":\"readback-batch\",\"backend\":\""
      << qwen_host_image_case::backend_name(kind)
      << "\",\"models\":0,\"graphs\":1,\"graph_submissions\":" << graph_submissions
      << ",\"prime_graph_submissions\":" << prime_graph_submissions
      << ",\"successful_batch_reads\":" << batch_reads << ",\"copy_batches\":" << batch_reads + observation_status_failures
      << ",\"logical_copies\":" << 2 * (batch_reads + observation_status_failures)
      << ",\"logical_bytes\":" << (batch_reads + observation_status_failures) * (sizeof(std::int32_t) + 4 * sizeof(float))
      << ",\"native_calls\":" << native_calls << ",\"native_waits\":" << native_waits
      << ",\"phase_completions\":" << phase_completions << ",\"host_waits\":" << host_waits
      << ",\"invalid_target_rejections\":" << invalid_targets
      << ",\"unsubmitted_result_rejections\":1,\"native_pending_rejections\":" << pending_rejections
      << ",\"observation_status_seam_failures\":" << observation_status_failures
      << ",\"late_native_retirements\":" << late_retirements << ",\"pending_batch_abandons\":" << pending_abandons
      << ",\"pending_source_writer_refusals\":" << source_writer_refusals
      << ",\"late_graph_replays\":" << late_graph_replays
      << ",\"copy_reports\":[";
  for (std::size_t index = 0; index < copy_reports.size(); ++index) {
    if (index) std::cout << ',';
    write_report(copy_reports[index]);
  }
  std::cout << "],\"two_dtypes_exact\":true,\"destination_canaries_checked\":true,"
         "\"different_replay_data_checked\":true,\"repeated_read_checked\":true,"
         "\"original_graph_reuse_checked\":true,\"no_escaped_copy_readers\":true,"
         "\"physical_driver_fault_injection\":false,\"performance_claim\":false}\n";
  return 0;
}

// Optional cached tiny-model consumer. The zero-model component test above
// remains independently runnable; this path reads only the supplied fixture.
inline int run_model_handoff(const std::filesystem::path &input, i::BackendKind kind) {
#if defined(HYLE_COMPUTE_CUDA) && defined(HYLE_TEST_INFERENCE_WARM_COST) && HYLE_TEST_INFERENCE_WARM_COST
  if (kind != i::BackendKind::kCuda) return 0;
  std::size_t models = 0, graph_submissions = 0, observation_failures = 0;
  std::size_t retained_sessions = 0, dropped_sessions = 0;
  std::array<std::size_t, 2> budget_controls{};
  for (const bool retain_session : {true, false}) {
    auto source = q::prepare_weight_source(input / "weights.tsv", false);
    const auto geometry = source.roles.geometry;
    require(geometry.hidden == 64 && geometry.vocabulary == 64 && geometry.context_limit >= 16 &&
        geometry.recurrent_layers() && geometry.attention_layers(),
        "readback Model handoff requires its cached untrained GA/H64/V64/C16 fixture");
    const auto plan = q::cache::plan(geometry, source.cache_row_bytes, 16, 1, 3,
        q::recurrent_bytes(q::state_schema_for(geometry, 16), geometry), 1, 1, false, true);
    auto model = std::make_unique<q::Model>(std::move(source), geometry, 16, plan,
        std::filesystem::path{}, false, q::ReplayPolicy::kCommands, SIZE_MAX,
        q::Model::Sampling{.enabled = true, .policy = {1.F, 0, 1.F}, .seed = 123},
        std::size_t{0}, i::BackendSelection{.primary = kind});
    ++models;
    auto budget = model->execution_budget;
    require(budget.valid(), "readback Model handoff lacks its original execution metadata budget");
    const auto metadata_live = [&] { return q::take(budget.usage(q::mem::system_resource())).live_bytes; };
    const auto budget_control = metadata_live();
    budget_controls[retain_session ? 0 : 1] = budget_control;
    const std::array rows{std::size_t{2}};
    model->prepare_family(rows); model->plan_replay_uses(2, 1, 2); model->prepare_execution(rows);
    auto session = model->create_session(rows, {}, q::GraphRole::kText, q::cache::SamplingCursor{123, 7, 0});
    const auto *const original_head = session->head.get();
    const auto original_epoch = session->head.epoch();
    const auto original_length = session->head->length;
    const auto original_sampling = session->head->sampling;
    const auto original_next_id = session->head->next_id;
    std::weak_ptr<q::ColdResources> cold = model->resources_;
    std::weak_ptr<q::Session> session_lifetime = session;
    require(metadata_live() > budget_control, "readback Model handoff did not track its prepared execution metadata");
    const std::array<std::int32_t, 2> tokens{1, 2};
    model->prepare_evaluation(*session, tokens); model->submit_evaluation(*session); ++graph_submissions;
    require(model->active_step && model->active_step->phase && model->active_step->submitted &&
        model->active_step->phase->slots[model->active_step->state_slot].data.small_results.valid(),
        "readback Model handoff did not select its original ordinary small-result copy");

    ObservationStatusSeam seam;
    bool caught = false, copy_pending = false;
    {
      i::testing::ReadbackBatchHookScope hook({&seam,
          &ObservationStatusSeam::before_submit, &ObservationStatusSeam::observe});
      try { (void)q::finish_evaluation(*model, *session, "readback-model-handoff", {}, false); }
      catch (const i::EvaluationFailure &failure) {
        caught = failure.status() == seam.failure && failure.retirement_observed() && !failure.retirement().retired;
      }
    }
    if (model->active_step && model->active_step->phase)
      copy_pending = model->active_step->phase->slots[model->active_step->state_slot].data.small_results.in_flight();
    require(caught && copy_pending && seam.submit_hooks == 1 && seam.wait_hooks == 1 && seam.gate.blocked() &&
        session->head.get() == original_head && session->head.epoch() == original_epoch &&
        session->head->length == original_length && session->head->sampling == original_sampling &&
        session->head->next_id == original_next_id,
        "failed original readback published a Model head/RNG boundary or lost its pending copy");
    ++observation_failures;

    if (!retain_session) {
      session.reset();
      require(!session_lifetime.expired(), "unpublished original Step did not retain its Session before Model handoff");
    }
    model.reset();
    require(seam.gate.blocked() && !cold.expired() && metadata_live() > budget_control,
        "Model destruction waited for pending copy or released its original cold resources early");
    if (retain_session) {
      require(session->stopping && !session->pending,
          "Model handoff did not withdraw the original Session's application links");
      const auto pending = session->try_retire();
      require(!pending.retired && pending.status == seam.failure && seam.gate.blocked() &&
          session->head.get() == original_head && session->head.epoch() == original_epoch &&
          session->head->sampling == original_sampling,
          "Session handoff inferred retirement from an idle Graph while its original copy remained pending");
      ++retained_sessions;
    } else {
      require(session_lifetime.expired(), "Model handoff left a Session owner in the original copy's source scope");
      ++dropped_sessions;
    }
    seam.gate.finish();
    const auto release_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    if (retain_session) {
      // Abandonment handed observation to the original CUDA reaper. A Session
      // drain may still see that host callback pending after the gate opens.
      auto retired = session->drain();
      require(retired.status == seam.failure, "late Session drain changed its original observation diagnosis");
      while (!retired.retired && std::chrono::steady_clock::now() < release_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        retired = session->try_retire();
        require(retired.status == seam.failure, "late Session retry changed its original observation diagnosis");
      }
      require(retired.retired && !session->head,
          "Session retry did not observe real late retirement or retained a failed unpublished boundary");
      session.reset();
    }
    while ((!cold.expired() || metadata_live() != budget_control) && std::chrono::steady_clock::now() < release_deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(cold.expired() && session_lifetime.expired() && metadata_live() == budget_control,
        "late Model/Session retirement leaked its original source scope or tracked execution metadata");
  }
  require(models == 2 && graph_submissions == 2 && observation_failures == 2 &&
      retained_sessions == 1 && dropped_sessions == 1, "Model handoff changed its finite two-arm coverage");
  std::cout << "{\"status\":\"PASS\",\"case\":\"readback-model-handoff\",\"backend\":\"cuda\","
      "\"models\":2,\"graph_submissions\":2,\"observation_status_seam_failures\":2,"
      "\"retained_session_arms\":1,\"dropped_external_session_arms\":1,"
      "\"unpublished_head_rng_unchanged\":true,\"model_destruction_nonblocking\":true,"
      "\"session_pending_after_model_handoff\":true,\"late_native_retirement\":true,"
      "\"late_source_lifetime_without_session\":true,"
      "\"original_error_preserved\":true,\"final_cold_resources\":0,\"final_extra_execution_metadata_bytes\":0,"
      "\"retained_budget_control_bytes\":[" << budget_controls[0] << ',' << budget_controls[1]
      << "],\"physical_driver_fault_injection\":false,\"performance_claim\":false}\n";
#else
  (void)input; (void)kind;
#endif
  return 0;
}
} // namespace inference_readback_batch_case
