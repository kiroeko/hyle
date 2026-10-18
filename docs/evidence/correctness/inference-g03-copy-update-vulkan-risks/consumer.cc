#include "dirty_page_native_risk_fixture.h"
#include "native_vulkan_failure_fixture.h"
#include "../../src/compute_route_vulkan/vulkan_file_stream_access.h"

namespace {
namespace r = hyle_g03_dirty_risk;
namespace n = r::n;
using hyle::Status;
std::atomic<bool> unknown_retirement{false};
std::atomic<unsigned> confirmed_fences{0};
Status episode(unsigned mode) {
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_RETURN_IF_ERROR(r::require(!context.environment().software_device, "actual Vulkan GPU required"));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(context, n::MemoryKind::kStaging));
  struct Close final { n::Context &context; ~Close() { context.close().IgnoreError(); } } close{context};
  HYLE_ASSIGN_OR_RETURN(auto state, r::prepare(context, device, staging));
  hyle_f5_vk_failure_fixture::Boundary boundary(context);
  boundary.fail_at = mode == 1 ? 1 : 0;
  confirmed_fences = 0;
  unknown_retirement = mode == 5;
  boundary.device.fn.GetFenceStatus = [](VkDevice device, VkFence fence) -> VkResult {
    auto &hook = *hyle_f5_vk_failure_fixture::Boundary::active;
    ++hook.polls;
    if (hook.fail_poll.exchange(false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto result = hook.original.GetFenceStatus(device, fence);
    // Mask a positive observation only. Actual GPU work/fence/wait remain real;
    // this tests unknown retirement, not a fictitious long-running shader.
    if (unknown_retirement.load()) return VK_NOT_READY;
    if (result == VK_SUCCESS) ++confirmed_fences;
    return result;
  };
  HYLE_ASSIGN_OR_RETURN(auto submitted, state.copy.submit());
  if (mode == 3) state.copy.close().IgnoreError();
  HYLE_ASSIGN_OR_RETURN(auto owner, n::CopyBatchUpdate::take(std::move(state.copy), std::move(submitted)));
  if (mode == 1) {
    HYLE_RETURN_IF_ERROR(r::require(owner.wait().IsResourceExhausted() &&
        !n::detail::CopyStreamAccess::input(owner, {1}) &&
        !n::detail::CopyStreamAccess::dependency(owner) && !owner.take_mutable({1}) &&
        boundary.submits == 1 && boundary.accepted == 0,
        "rejected native suffix fabricated an update input or writer"));
  } else {
    HYLE_RETURN_IF_ERROR(r::require(boundary.accepted == 1 && state.row_pool.stats().active_uses == 1,
        "update risk had no real native copy admission"));
    HYLE_ASSIGN_OR_RETURN(auto input, n::detail::CopyStreamAccess::input(owner, {1}));
    HYLE_ASSIGN_OR_RETURN(auto dependency, n::detail::CopyStreamAccess::dependency(owner));
    HYLE_RETURN_IF_ERROR(r::require(input.writer != nullptr && input.root != nullptr &&
        input.last_write.value == 1 && dependency.pending() &&
        !n::detail::CopyStreamAccess::input(owner, {0}) &&
        !n::detail::CopyStreamAccess::input(owner, {2}),
        "update source admission lost original identity or accepted a non-destination"));
    if (mode == 2) {
      unknown_retirement = true;
      boundary.fail_poll = true;
      const auto status = owner.observe();
      // Original Run keeps an observation error while retirement is unknown.
      // Its original Completion remains pending until an actual fence proof;
      // typed geometry/dependency are still borrowed, never a ready read.
      HYLE_ASSIGN_OR_RETURN(auto unknown_input, n::detail::CopyStreamAccess::input(owner, {1}));
      HYLE_ASSIGN_OR_RETURN(auto unknown_prefix, n::detail::CopyStreamAccess::dependency(owner));
      HYLE_RETURN_IF_ERROR(r::require(status.IsUnavailable() && !boundary.fail_poll.load() &&
          owner.in_flight() && state.row_pool.stats().active_uses == 1 &&
          unknown_input.writer != nullptr && !unknown_input.writer->view() &&
          unknown_prefix.pending() && !owner.take_mutable({1}),
          "unknown observation fabricated native retirement or ready update permission"));
      unknown_retirement = false;
    }
    if (mode == 4) HYLE_RETURN_IF_ERROR(owner.request_cancel());
    if (mode == 5) {
      HYLE_RETURN_IF_ERROR(r::require(owner.observe().IsUnavailable(), "masked retirement was treated as ready"));
      owner = {};
      HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses == 1,
          "abandoned unknown update released its original native guards"));
      unknown_retirement = false;
      HYLE_RETURN_IF_ERROR(context.close());
      HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses == 0 &&
          !state.data.keys.acquire(0), "abandonment fabricated a published reusable page"));
    } else {
      const auto terminal = owner.wait();
      HYLE_RETURN_IF_ERROR(r::require(mode == 2 ? terminal.IsResourceExhausted() :
          mode == 4 ? terminal.IsCancelled() : terminal.ok(), "update wait lost original terminal status"));
      HYLE_RETURN_IF_ERROR(r::require(state.row_pool.stats().active_uses == 0 &&
          (boundary.waits > 0 || confirmed_fences > 0), "update lacked actual native fence retirement"));
      if (mode == 2) HYLE_RETURN_IF_ERROR(r::require(
          !n::detail::CopyStreamAccess::input(owner, {1}) &&
          !n::detail::CopyStreamAccess::dependency(owner),
          "known retired failure admitted typed update input or prefix"));
      if (mode == 2 || mode == 3 || mode == 4) {
        HYLE_RETURN_IF_ERROR(r::require(!owner.take_mutable({1}), "failed/closed/cancelled update returned writer"));
      } else {
        HYLE_ASSIGN_OR_RETURN(state.writer.lease(), owner.take_mutable({1}));
        HYLE_RETURN_IF_ERROR(r::require(state.writer.lease().generation() == state.generation,
            "update returned a different page generation"));
        HYLE_ASSIGN_OR_RETURN(state.data.key_reads[0], state.writer.publish());
        HYLE_ASSIGN_OR_RETURN(auto frame, state.data.keys.read(state.data.layout, state.data.key_reads));
        HYLE_RETURN_IF_ERROR(r::require(frame.page_count() == 2, "update lost original exact frame"));
      }
    }
  }
  if (owner.valid()) {
    HYLE_ASSIGN_OR_RETURN(state.copy, owner.release_invocation());
    state.copy.drain().IgnoreError();
  }
  std::fprintf(stderr, "actual Vulkan CopyBatchUpdate mode=%u PASS: native submits=%u accepted=%u "
      "waits=%u confirmed=%u; original permissions/terminal/retirement retained\n",
      mode, boundary.submits, boundary.accepted, boundary.waits.load(), confirmed_fences.load());
  return Status::Ok();
}
Status run() {
  for (unsigned mode = 0; mode < 6; ++mode) HYLE_RETURN_IF_ERROR(episode(mode));
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "update risks: %.*s\n", int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
