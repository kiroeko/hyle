#define HYLE_G01_PAGE_RISK_APPLE 1
#include "dirty_page_native_risk_fixture.h"
#include "native_apple_failure_fixture.h"

#include <optional>
#include <thread>
#include <utility>

namespace {
namespace r = hyle_g03_dirty_risk;
namespace n = r::n;
namespace mx = mlx::core;
using hyle::Status;

// Prototype only: transfer both original controls to one observation owner.
// It adds no queue, Completion state or field to either original control.
struct CopyOwner final {
  n::CopyBatchInvocation invocation;
  n::CopyBatchSubmission submission;
};
static_assert(sizeof(CopyOwner) == sizeof(n::CopyBatchInvocation) +
                                 sizeof(n::CopyBatchSubmission));

Status episode(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto state, r::prepare(context, device, staging));
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
  CopyOwner owner{std::move(state.copy), std::move(submission)};
  HYLE_RETURN_IF_ERROR(r::require(!state.copy.valid() && !submission.valid() &&
      owner.invocation.in_flight() && state.row_pool.stats().active_uses == 1 &&
      !owner.invocation.unbind_mutable({1}) && !owner.submission.output({1}, {1}, 256, 256),
      "move handoff changed pending authority or left another public observer"));
  if (mode == 1) HYLE_RETURN_IF_ERROR(owner.submission.request_cancel());
  if (mode == 4) owner.invocation.close().IgnoreError();

  std::optional<CopyOwner> returned;
  Status terminal;
  // The fixture thread proves handoff, not a proposed product execution owner.
  std::thread observer([owned = std::move(owner), &returned, &terminal, mode]() mutable {
    if (mode == 3) return; // Destroy both original controls without a hidden wait.
    terminal = owned.submission.wait();
    if (terminal.ok()) {
      owned.submission = {};
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
          returned->invocation.report().metadata_bytes == bytes,
          "handoff rebuilt or lost the original cold copy slot"));
      state.copy = std::move(returned->invocation);
      if (mode == 4) HYLE_RETURN_IF_ERROR(r::require(!state.copy.unbind_mutable({1}),
                                                   "closed handed-off slot returned writer"));
      else HYLE_RETURN_IF_ERROR(r::success(state));
    }
  }
  std::fprintf(stderr, "actual Apple original copy invocation+submission move handoff mode=%u PASS; "
      "one observer, original retirement/guards/failure, 16-byte local owner; explicit fixture wait, "
      "not pending attention qualification\n", mode);
  return Status::Ok();
}
Status run() {
  for (unsigned mode = 0; mode < 5; ++mode) HYLE_RETURN_IF_ERROR(episode(mode));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "pending copy owner: %.*s\n",
                                int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
