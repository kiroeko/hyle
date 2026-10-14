#include <hyle/compute/batch_transfer.h>
#if defined(HYLE_COPY_UPDATE_APPLE)
#include <hyle/compute/apple_memory.h>
#elif defined(HYLE_COPY_UPDATE_VULKAN)
#include <hyle/compute/vulkan_memory.h>
#elif defined(HYLE_COPY_UPDATE_CUDA)
#include <hyle/compute/backends/cuda.h>
#else
#include <hyle/compute/backends/cpu.h>
#endif
#include <algorithm>
#include <array>
#include <cstdio>
#include <type_traits>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
#if defined(HYLE_COPY_UPDATE_APPLE)
namespace n = c::apple;
#elif defined(HYLE_COPY_UPDATE_VULKAN)
namespace n = c::vulkan;
#else
namespace n = c;
#endif
using hyle::Status;
static_assert(!std::is_copy_constructible_v<n::CopyBatchUpdate>);
static_assert(std::is_nothrow_move_constructible_v<n::CopyBatchUpdate>);
static_assert(sizeof(n::CopyBatchUpdate) == sizeof(n::CopyBatchInvocation) + sizeof(n::CopyBatchSubmission));
Status require(bool good, const char *message) {
  return good ? Status::Ok() : Status::Internal(message);
}
Status run() {
#if defined(HYLE_COPY_UPDATE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto host, n::MemoryResource::create(context));
  const auto destination_resource = device.ref(), source_resource = host.ref();
#elif defined(HYLE_COPY_UPDATE_VULKAN)
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_RETURN_IF_ERROR(require(!context.environment().software_device, "actual Vulkan device required"));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, n::MemoryResource::create(context, n::MemoryKind::kStaging));
  const auto destination_resource = device.ref(), source_resource = host.ref();
#elif defined(HYLE_COPY_UPDATE_CUDA)
  HYLE_ASSIGN_OR_RETURN(auto device_id, c::cuda::device(0));
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(device_id));
  HYLE_ASSIGN_OR_RETURN(auto device, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto host, c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  const auto destination_resource = device.ref(), source_resource = host.ref();
#else
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(c::cpu::device()));
  const auto destination_resource = m::system_resource(), source_resource = m::system_resource();
#endif
#if !defined(HYLE_COPY_UPDATE_APPLE) && !defined(HYLE_COPY_UPDATE_VULKAN)
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  struct Drain final { c::Queue &queue; ~Drain() { queue.drain().IgnoreError(); } } drain{queue};
#endif
  constexpr std::size_t page_bytes = 16384, row_bytes = 1024;
  HYLE_ASSIGN_OR_RETURN(auto source_pool, c::LeasePool::create(source_resource, page_bytes + 512));
  HYLE_ASSIGN_OR_RETURN(auto destination_pool, c::LeasePool::create(destination_resource, page_bytes + 512));
  HYLE_ASSIGN_OR_RETURN(auto incoming, source_pool.allocate(page_bytes));
  HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte{17}));
  HYLE_ASSIGN_OR_RETURN(auto source, incoming.publish());
  HYLE_ASSIGN_OR_RETURN(auto page, destination_pool.allocate(page_bytes));
  const auto generation = page.generation();
  const std::array specs{c::CopyResourceSpec{source_resource, page_bytes},
                         c::CopyResourceSpec{destination_resource, page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program, n::CopyBatchProgram::prepare(context, specs));
#if defined(HYLE_COPY_UPDATE_APPLE) || defined(HYLE_COPY_UPDATE_VULKAN)
  HYLE_ASSIGN_OR_RETURN(auto copy, n::CopyBatchInvocation::create(program));
  HYLE_ASSIGN_OR_RETURN(auto foreign, n::CopyBatchInvocation::create(program));
#else
  HYLE_ASSIGN_OR_RETURN(auto copy, n::CopyBatchInvocation::create(program, queue));
  HYLE_ASSIGN_OR_RETURN(auto foreign, n::CopyBatchInvocation::create(program, queue));
#endif
  HYLE_ASSIGN_OR_RETURN(auto readback_pool, c::LeasePool::create(source_resource, page_bytes + 512));
  const std::array down_specs{c::CopyResourceSpec{destination_resource, page_bytes},
                              c::CopyResourceSpec{source_resource, page_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto down_program, n::CopyBatchProgram::prepare(context, down_specs));
#if defined(HYLE_COPY_UPDATE_APPLE) || defined(HYLE_COPY_UPDATE_VULKAN)
  HYLE_ASSIGN_OR_RETURN(auto down, n::CopyBatchInvocation::create(down_program));
#else
  HYLE_ASSIGN_OR_RETURN(auto down, n::CopyBatchInvocation::create(down_program, queue));
#endif
  const auto metadata = copy.report().metadata_bytes;
  const std::array<c::CopyPhase,3> phases{{{}, {.dependency_mask=1}, {}}};
  std::array<std::byte,page_bytes> expected, actual;
  expected.fill(std::byte{17});
  for (unsigned step = 0; step < 65; ++step) {
    if (step != 0) {
      HYLE_ASSIGN_OR_RETURN(incoming, source.try_write());
      HYLE_RETURN_IF_ERROR(incoming.view()->fill(std::byte(23 + step)));
      HYLE_ASSIGN_OR_RETURN(source, incoming.publish());
    }
    const auto offset = step == 0 ? 0 : ((step - 1) % 16) * row_bytes;
    const auto bytes = step == 0 ? page_bytes : row_bytes;
    HYLE_RETURN_IF_ERROR(copy.bind_source({0}, source));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(page)));
    const std::array<c::CopyRange,1> ranges{{{{0},{1},{1},0,offset,bytes}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted, copy.submit());
    const auto rejected = n::CopyBatchUpdate::take(std::move(foreign), std::move(submitted));
    HYLE_RETURN_IF_ERROR(require(!rejected && rejected.error().IsInvalidArgument() &&
        foreign.valid() && submitted.valid() && copy.valid(), "foreign controls were consumed or accepted"));
    HYLE_ASSIGN_OR_RETURN(auto update, n::CopyBatchUpdate::take(std::move(copy), std::move(submitted)));
    HYLE_RETURN_IF_ERROR(require(!copy.valid() && !submitted.valid() && update.valid(),
                                 "update did not move both original controls"));
    HYLE_RETURN_IF_ERROR(update.wait());
    HYLE_RETURN_IF_ERROR(require(update.report().logical_bytes == bytes &&
        update.report().metadata_bytes == metadata, "update rebuilt its cold slot or copied history"));
    HYLE_ASSIGN_OR_RETURN(page, update.take_mutable({1}));
    HYLE_RETURN_IF_ERROR(require(!update.take_mutable({1}) && page.generation() == generation,
                                 "update returned writer twice or changed original generation"));
    HYLE_ASSIGN_OR_RETURN(copy, update.release_invocation());
    HYLE_RETURN_IF_ERROR(require(!update.valid() && copy.valid(), "original invocation was not recovered"));
    HYLE_ASSIGN_OR_RETURN(auto ready, page.publish());
    if (step != 0) std::fill_n(expected.begin() + offset, row_bytes, std::byte(23 + step));
    // Read device content through the original staging copy route. Device
    // resources do not become host-readable just because the GPU uses UMA.
    HYLE_ASSIGN_OR_RETURN(auto readback_writer, readback_pool.allocate(page_bytes));
    HYLE_RETURN_IF_ERROR(down.bind_source({0}, ready));
    HYLE_RETURN_IF_ERROR(down.bind_mutable({1}, std::move(readback_writer)));
    const std::array<c::CopyPhase,1> down_phases{{{}}};
    const std::array<c::CopyRange,1> down_ranges{{{{0},{1},{0},0,0,page_bytes}}};
    HYLE_RETURN_IF_ERROR(down.prepare_replay(down_phases, down_ranges));
    c::ReadLease readback;
    {
      HYLE_ASSIGN_OR_RETURN(auto downloaded, down.submit());
      HYLE_RETURN_IF_ERROR(downloaded.wait());
      HYLE_ASSIGN_OR_RETURN(readback, downloaded.output({0}, {1}, 0, page_bytes));
    }
    HYLE_RETURN_IF_ERROR(readback.view()->read(actual));
    HYLE_RETURN_IF_ERROR(down.reset());
    readback = {};
    HYLE_RETURN_IF_ERROR(require(actual == expected, "dirty data or untouched history differs"));
    HYLE_ASSIGN_OR_RETURN(page, ready.try_write());
    HYLE_RETURN_IF_ERROR(copy.reset());
  }
  HYLE_RETURN_IF_ERROR(require(source_pool.stats().active_uses == 0 &&
      destination_pool.stats().active_uses == 0, "retired update retained native permission"));
  std::fprintf(stderr, "public CopyBatchUpdate PASS: initialization + 64 dirty updates, "
      "1024/16384 bytes, original generation/cold slot, foreign rejection and control recovery\n");
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "copy update: %.*s\n", int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
