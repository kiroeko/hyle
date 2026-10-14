#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/detail/copy_stream_access.h>
#include <hyle/compute/transfer.h>
#include "owned_outcome_completion_fixture.h"

#include <array>
#include <cstdio>
#include <utility>

namespace {
namespace c = hyle::compute;
using hyle::Status;
Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(c::cpu::device()));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(hyle::mem::system_resource(), 4096));
  HYLE_ASSIGN_OR_RETURN(auto source, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(1024));
  HYLE_ASSIGN_OR_RETURN(auto receiver, pool.allocate(1024));
  HYLE_RETURN_IF_ERROR(source.view()->fill(std::byte{71}));
  HYLE_ASSIGN_OR_RETURN(auto ready_source, source.publish());
  const std::array specs{c::CopyResourceSpec{hyle::mem::system_resource(), 1024},
                         c::CopyResourceSpec{hyle::mem::system_resource(), 1024}};
  HYLE_ASSIGN_OR_RETURN(auto program, c::CopyBatchProgram::prepare(context, specs));
  HYLE_ASSIGN_OR_RETURN(auto copy, c::CopyBatchInvocation::create(program, queue));
  HYLE_RETURN_IF_ERROR(copy.bind_source({0}, ready_source));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(destination)));
  HYLE_RETURN_IF_ERROR(copy.bind_producer({0}, c::Completion::completed()));
  const std::array<c::CopyPhase, 2> phases{{{}, {}}};
  const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 1024}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
  HYLE_ASSIGN_OR_RETURN(auto submitted, copy.submit());
  HYLE_ASSIGN_OR_RETURN(auto update, c::CopyBatchUpdate::take(std::move(copy), std::move(submitted)));
  HYLE_ASSIGN_OR_RETURN(auto pending, c::detail::CopyStreamAccess::input(update, {1}));
  HYLE_ASSIGN_OR_RETURN(auto uses, c::LeaseUsePool::create());
  HYLE_ASSIGN_OR_RETURN(auto reading, uses.begin());
  HYLE_RETURN_IF_ERROR(c::detail::CopyStreamAccess::admit_read(reading, update, {1}));
  HYLE_ASSIGN_OR_RETURN(auto receiver_view, receiver.view());
  HYLE_RETURN_IF_ERROR(reading.write(receiver));
  HYLE_ASSIGN_OR_RETURN(auto slot, c::detail::ResultSlot<c::WriteLease>::create(hyle::mem::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(auto write, slot.begin());
  HYLE_RETURN_IF_ERROR(write.emplace(std::move(receiver)));
  HYLE_ASSIGN_OR_RETURN(auto suffix, c::copy_async(queue, receiver_view, pending.view));
  HYLE_RETURN_IF_ERROR(reading.attach(suffix));
  HYLE_ASSIGN_OR_RETURN(auto result, write.publish(std::move(suffix)));
  using Aggregate = hyle_g03_outcome_probe::Outcome<c::WriteLease, c::CopyBatchUpdate>;
  HYLE_ASSIGN_OR_RETURN(auto aggregate, Aggregate::take<c::detail::CopyStreamAccess>(
      std::move(result), std::move(update)));
  HYLE_RETURN_IF_ERROR(aggregate.completion().wait());
  HYLE_ASSIGN_OR_RETURN(auto lease, aggregate.wait());
  HYLE_ASSIGN_OR_RETURN(auto repeated, aggregate.result());
  HYLE_RETURN_IF_ERROR(require(repeated.get() == lease.get() && !slot.begin(),
      "CPU aggregate changed result identity or old slot ownership"));
  std::array<std::byte, 1024> bytes{};
  HYLE_RETURN_IF_ERROR(lease->view()->read(bytes));
  for (auto byte : bytes)
    HYLE_RETURN_IF_ERROR(require(byte == std::byte{71}, "CPU original copy/suffix oracle differs"));
  HYLE_RETURN_IF_ERROR(require(uses.active_batches() == 0, "CPU original guard did not retire"));
  std::fprintf(stderr, "CPU original producer/F2/suffix aggregate PASS:1024 bytes, inline phase, same result/slot\n");
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
