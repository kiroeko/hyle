#define HYLE_G01_PAGE_RISK_APPLE 1
#include "dirty_page_native_risk_fixture.h"
#include "native_apple_failure_fixture.h"
#include "../../src/compute_route_apple/apple_file_stream_access.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "hyle/compute/detail/apple_plan.h"

#include <optional>
#include <thread>
#include <utility>

namespace {
namespace r = hyle_g03_dirty_risk;
namespace n = r::n;
namespace mx = mlx::core;
using hyle::Status;

using CopyOwner = n::CopyBatchUpdate;
static_assert(sizeof(CopyOwner) == sizeof(n::CopyBatchInvocation) +
                                 sizeof(n::CopyBatchSubmission));

Status episode(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto state, r::prepare(context, device, staging));
  const std::array foreign_specs{r::c::CopyResourceSpec{device.ref(), 256}};
  HYLE_ASSIGN_OR_RETURN(auto foreign_program, n::CopyBatchProgram::prepare(context, foreign_specs));
  HYLE_ASSIGN_OR_RETURN(auto foreign, n::CopyBatchInvocation::create(foreign_program));
  // The final independent empty phase is inline even while the prior native
  // dirty write remains pending. It cannot erase the update's real prefix.
  const std::array<r::c::CopyPhase,3> phases{{{}, {.dependency_mask=1}, {}}};
  const std::array<r::c::CopyRange,1> ranges{{{{0},{1},{1},0,256,256}}};
  HYLE_RETURN_IF_ERROR(state.copy.prepare_replay(phases, ranges));
  auto gate = std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Close final {
    decltype(gate) value;
    n::Context &context;
    ~Close() { value->release(); context.close().IgnoreError(); }
  } close{gate, context};
  const std::array specs{n::InputSpec{{4}, mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker, context.compile([gate](const auto &inputs, mx::Stream stream) {
    auto prefix = mx::square(inputs[0], stream);
    return std::vector{mx::array(prefix.shape(), prefix.dtype(),
        std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream, gate), {prefix})};
  }, specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking, blocker.create_invocation());
  const std::array inputs{mx::array({1.F, 2.F, 3.F, 4.F})};
  HYLE_RETURN_IF_ERROR(blocking.bind(inputs));
  { std::lock_guard lock(gate->mutex); gate->armed = true; gate->fail = mode == 2; }
  HYLE_ASSIGN_OR_RETURN(auto prefix, blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(), "owner probe lacked a real pending GPU checkpoint"));
  HYLE_ASSIGN_OR_RETURN(auto submission, state.copy.submit());
  const auto bytes = state.copy.report().metadata_bytes;
  const auto rejected = CopyOwner::take(std::move(foreign), std::move(submission));
  HYLE_RETURN_IF_ERROR(r::require(!rejected && rejected.error().IsInvalidArgument() &&
      foreign.valid() && submission.valid() && state.copy.valid(),
      "foreign update controls were accepted or consumed"));
  if (mode == 4) state.copy.close().IgnoreError();
  HYLE_ASSIGN_OR_RETURN(auto owner, CopyOwner::take(std::move(state.copy), std::move(submission)));
  HYLE_RETURN_IF_ERROR(r::require(!state.copy.valid() && !submission.valid() &&
      owner.in_flight() && state.row_pool.stats().active_uses == 1 &&
      owner.observe().IsUnavailable() && !owner.take_mutable({1}),
      "move handoff changed pending authority or left another public observer"));
  HYLE_ASSIGN_OR_RETURN(auto update_input, n::detail::CopyStreamAccess::input(owner, {1}));
  HYLE_ASSIGN_OR_RETURN(auto dependency, n::detail::CopyStreamAccess::dependency(owner));
  HYLE_RETURN_IF_ERROR(r::require(update_input.writer != nullptr && update_input.root != nullptr &&
      update_input.writer->generation() == state.generation &&
      update_input.view.size() == state.data.keys.page_bytes() &&
      update_input.physical.capacity >= update_input.view.size() &&
      update_input.last_write.value == 1 && dependency.pending() &&
      !n::detail::CopyStreamAccess::input(owner, {0}) &&
      !n::detail::CopyStreamAccess::input(owner, {2}) &&
      !update_input.writer->view(),
      "typed update input lost original geometry/permission or erased pending native prefix"));
  if (mode == 1) HYLE_RETURN_IF_ERROR(owner.request_cancel());

  std::optional<CopyOwner> returned;
  Status terminal;
  // The fixture thread proves handoff, not a proposed product execution owner.
  std::thread observer([owned = std::move(owner), &returned, &terminal, mode]() mutable {
    if (mode == 3) return; // Destroy both original controls without a hidden wait.
    terminal = owned.wait();
    if (terminal.ok()) {
      returned.emplace(std::move(owned));
    }
  });
  if (mode == 3) {
    observer.join();
    const auto still_pending = state.row_pool.stats().active_uses == 1;
    gate->release();
    HYLE_RETURN_IF_ERROR(context.close());
    HYLE_RETURN_IF_ERROR(r::require(still_pending && state.row_pool.stats().active_uses == 0,
        "abandoned moved controls did not retain until actual native retirement"));
    HYLE_RETURN_IF_ERROR(r::require(!state.data.keys.acquire(0),
        "abandoned consumed writer fabricated a reusable arena page"));
  } else {
    gate->release();
    observer.join();
    HYLE_RETURN_IF_ERROR(r::require(mode == 1 ? terminal.IsCancelled() :
        mode == 2 ? terminal.IsInternal() : terminal.ok(),
        "handed-off original observer lost cancellation or prefix failure"));
    HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses == 0,
                                   "actual retired copy kept source uses"));
    if (mode == 1 || mode == 2) {
      HYLE_RETURN_IF_ERROR(r::require(!returned && !state.data.keys.acquire(0),
                                    "failed handoff fabricated ready/reusable page"));
    } else {
      HYLE_RETURN_IF_ERROR(r::require(returned.has_value() &&
          returned->report().metadata_bytes == bytes,
          "handoff rebuilt or lost the original cold copy slot"));
      if (mode == 4) HYLE_RETURN_IF_ERROR(r::require(!returned->take_mutable({1}),
                                                   "closed handed-off slot returned writer"));
      else {
        HYLE_ASSIGN_OR_RETURN(state.writer.lease(), returned->take_mutable({1}));
        HYLE_RETURN_IF_ERROR(r::require(state.writer.lease().generation() == state.generation,
                                       "update owner changed original page generation"));
        HYLE_ASSIGN_OR_RETURN(state.data.key_reads[0], state.writer.publish(state.data.schema.capacity().page_size));
        HYLE_ASSIGN_OR_RETURN(auto frame, state.data.keys.read(state.data.layout, state.data.key_reads));
        HYLE_RETURN_IF_ERROR(r::require(frame.page_count() == 2, "returned update writer broke exact frame"));
      }
      HYLE_ASSIGN_OR_RETURN(state.copy, returned->release_invocation());
      HYLE_RETURN_IF_ERROR(r::require(!returned->valid() && state.copy.valid() &&
          state.copy.report().metadata_bytes == bytes, "original invocation recovery rebuilt its cold slot"));
    }
  }
  std::fprintf(stderr, "actual Apple original copy invocation+submission move handoff mode=%u PASS; "
      "one observer, original retirement/guards/failure, 16-byte local owner; explicit fixture wait, "
      "not pending attention qualification\n", mode);
  return Status::Ok();
}
struct ReadPreflight final {
  CopyOwner *source = nullptr;
  unsigned calls = 0;
  static Status check(void *opaque) noexcept {
    auto &self = *static_cast<ReadPreflight *>(opaque);
    ++self.calls;
    // The original FIFO reaches this after the original producer's native
    // retirement. This is the sole source observer until the suffix retires.
    return self.source->observe();
  }
};
Status successor(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto foreign_context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto state, r::prepare(context, device, staging));
  const std::array<r::c::CopyPhase,3> phases{{{}, {.dependency_mask=1}, {}}};
  const std::array<r::c::CopyRange,1> ranges{{{{0},{1},{1},0,256,256}}};
  HYLE_RETURN_IF_ERROR(state.copy.prepare_replay(phases, ranges));
  HYLE_ASSIGN_OR_RETURN(auto destination, hyle::mem::Buffer::create(device.ref(), 256));
  HYLE_ASSIGN_OR_RETURN(auto recording, n::detail::prepare_copy_recording(context, 1));
  HYLE_ASSIGN_OR_RETURN(auto uses, r::c::LeaseUsePool::create());
  auto gate = std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Close final {
    decltype(gate) value;
    n::Context &context;
    ~Close() { value->release(); context.close().IgnoreError(); }
  } close{gate, context};
  const std::array specs{n::InputSpec{{4}, mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker, context.compile([gate](const auto &inputs, mx::Stream stream) {
    auto prefix = mx::square(inputs[0], stream);
    return std::vector{mx::array(prefix.shape(), prefix.dtype(),
        std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream, gate), {prefix})};
  }, specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking, blocker.create_invocation());
  const std::array inputs{mx::array({1.F, 2.F, 3.F, 4.F})};
  HYLE_RETURN_IF_ERROR(blocking.bind(inputs));
  { std::lock_guard lock(gate->mutex); gate->armed = true; gate->fail = mode == 2; }
  HYLE_ASSIGN_OR_RETURN(auto prefix, blocking.submit());
  HYLE_RETURN_IF_ERROR(r::require(gate->await(), "successor lacked a real pending native prefix"));
  HYLE_ASSIGN_OR_RETURN(auto submission, state.copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto owner, CopyOwner::take(std::move(state.copy), std::move(submission)));
  HYLE_ASSIGN_OR_RETURN(auto dependency, n::detail::CopyStreamAccess::dependency(owner));
  HYLE_RETURN_IF_ERROR(r::require(dependency.pending(), "inline final phase erased the real pending prefix"));
  HYLE_RETURN_IF_ERROR(r::require(n::detail::qualify_recording_dependency(
      n::detail::ContextAccess::runtime(foreign_context), dependency).IsInvalidArgument(),
      "foreign native endpoint admitted the update dependency"));
  HYLE_RETURN_IF_ERROR(n::detail::qualify_recording_dependency(
      n::detail::ContextAccess::runtime(context), dependency));
  HYLE_ASSIGN_OR_RETURN(auto input, n::detail::CopyStreamAccess::input(owner, {1}));
  HYLE_ASSIGN_OR_RETURN(auto row, input.view.subview(256, 256));
  HYLE_ASSIGN_OR_RETURN(auto reading, uses.begin());
  HYLE_RETURN_IF_ERROR(n::detail::CopyStreamAccess::admit_read(reading, owner, {1}));
  const std::array<hyle::compute::detail::CopyBackendRange,1> copies{{{destination.view(), row}}};
  HYLE_RETURN_IF_ERROR(n::detail::bind_copy_recording(*recording, copies));
  ReadPreflight preflight{&owner};
  HYLE_RETURN_IF_ERROR(n::detail::set_copy_preflight(*recording, &ReadPreflight::check, &preflight));
  if (mode == 1) HYLE_RETURN_IF_ERROR(owner.request_cancel());
  HYLE_ASSIGN_OR_RETURN(auto suffix, n::detail::submit_copy_recording(*recording));
  HYLE_RETURN_IF_ERROR(reading.attach(suffix));
  HYLE_ASSIGN_OR_RETURN(auto pending, suffix.poll());
  HYLE_RETURN_IF_ERROR(r::require(pending == r::c::CompletionPoll::kPending &&
      uses.active_batches() == 1 && !owner.take_mutable({1}),
      "suffix was ready or released the original page before actual native retirement"));
  gate->release();
  const auto terminal = suffix.wait(); // The only producer/copy/suffix host wait.
  HYLE_RETURN_IF_ERROR(r::require(mode == 1 ? terminal.IsCancelled() :
      mode == 2 ? terminal.IsInternal() : terminal.ok(),
      "ordered native suffix lost the original source's terminal status"));
  HYLE_RETURN_IF_ERROR(r::require(uses.active_batches() == 0,
      "suffix completion did not retire its original source read guard"));
  const auto report = n::detail::copy_recording_report(*recording);
  if (mode == 0) {
    std::array<std::byte,256> bytes;
    HYLE_RETURN_IF_ERROR(destination.view().read(bytes));
    for (auto byte : bytes) HYLE_RETURN_IF_ERROR(r::require(byte == std::byte{31},
        "ordered successor read stale dirty-page bytes"));
    HYLE_RETURN_IF_ERROR(r::require(preflight.calls == 1 && report.native_calls == 1,
        "successor did not use the original permission predicate and native copy"));
    HYLE_ASSIGN_OR_RETURN(state.writer.lease(), owner.take_mutable({1}));
    HYLE_RETURN_IF_ERROR(r::require(state.writer.lease().generation() == state.generation,
        "ordered successor changed the original page generation"));
    HYLE_ASSIGN_OR_RETURN(state.data.key_reads[0], state.writer.publish(state.data.schema.capacity().page_size));
  } else {
    HYLE_RETURN_IF_ERROR(r::require(report.native_calls == 0 && !owner.take_mutable({1}),
        "failed prefix executed suffix or published a ready original page"));
  }
  std::fprintf(stderr, "actual Apple typed update -> original native copy suffix mode=%u PASS; "
      "foreign endpoint rejected, original ordered read guard, terminal propagation; "
      "only final suffix host wait; source seam proof, not pending attention\n", mode);
  return Status::Ok();
}
Status run() {
  for (unsigned mode = 0; mode < 5; ++mode) HYLE_RETURN_IF_ERROR(episode(mode));
  for (unsigned mode = 0; mode < 3; ++mode) HYLE_RETURN_IF_ERROR(successor(mode));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "pending copy owner: %.*s\n",
                                int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
