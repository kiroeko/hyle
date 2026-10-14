#include <hyle/compute/detail/dependent_completion.h>
#include <hyle/compute/detail/generated_result.h>
#include <hyle/compute/detail/copy_outcome_source.h>
#include <hyle/compute/detail/copy_stream_access.h>
#include <hyle/compute/backends/cpu.h>
#include <hyle/compute/transfer.h>
#include <hyle/mem/quota_allocator.h>

#include <array>
#include <cstdio>
#include <utility>

namespace {
namespace c = hyle::compute;
struct InlineSource final {
  hyle::Status terminal;
  unsigned releases = 0;
  c::detail::OutcomeSources protocol() noexcept {
    return {this,
      [](void *opaque) noexcept -> hyle::Result<c::CompletionPoll> {
        const auto &status = static_cast<InlineSource *>(opaque)->terminal;
        if (!status.ok()) return std::unexpected(status);
        return c::CompletionPoll::kComplete;
      },
      [](void *opaque) noexcept { return static_cast<InlineSource *>(opaque)->terminal; },
      [](void *) noexcept { return hyle::Status::Ok(); },
      [](void *) noexcept {},
      [](void *opaque, c::detail::CompletionLifetime lifetime) noexcept {
        lifetime.terminal(lifetime.state, static_cast<InlineSource *>(opaque)->terminal);
        lifetime.release(lifetime.state);
      },
      [](void *opaque) noexcept { ++static_cast<InlineSource *>(opaque)->releases; }};
  }
};
hyle::Status require(bool condition, const char *message) noexcept {
  return condition ? hyle::Status::Ok() : hyle::Status::Internal(message);
}
hyle::Status copy_source() {
  HYLE_ASSIGN_OR_RETURN(auto context, c::Context::create(c::cpu::device()));
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  const auto resource = hyle::mem::system_resource();
  const std::array specs{c::CopyResourceSpec{resource, 16}, c::CopyResourceSpec{resource, 16}};
  HYLE_ASSIGN_OR_RETURN(auto program, c::CopyBatchProgram::prepare(context, specs, {.allocator = metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto invocation, c::CopyBatchInvocation::create(program, queue));
  using Source = c::detail::CopyOutcomeSource<c::CopyBatchUpdate, c::detail::CopyStreamAccess>;
  HYLE_ASSIGN_OR_RETURN(auto source, Source::create(2, {.allocator = metadata.get()}));
  HYLE_ASSIGN_OR_RETURN(auto observer, c::detail::DependentCompletion::create(metadata.get()));
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(resource, 1024));
  HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(16));
  HYLE_RETURN_IF_ERROR(writer.view()->fill(std::byte{41}));
  HYLE_ASSIGN_OR_RETURN(auto ready, writer.publish());
  HYLE_ASSIGN_OR_RETURN(auto output, pool.allocate(16));
  HYLE_ASSIGN_OR_RETURN(auto uses, c::LeaseUsePool::create());
  const auto cold_bytes = metadata.used();
  for (unsigned episode = 0; episode < 64; ++episode) {
    HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(16));
    HYLE_RETURN_IF_ERROR(invocation.bind_source({0}, ready));
    HYLE_RETURN_IF_ERROR(invocation.bind_mutable({1}, std::move(destination)));
    HYLE_RETURN_IF_ERROR(invocation.bind_producer({0}, c::Completion::completed()));
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 16}}};
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(phases, ranges));
    HYLE_ASSIGN_OR_RETURN(auto submitted, invocation.submit());
    HYLE_ASSIGN_OR_RETURN(auto update, c::CopyBatchUpdate::take(std::move(invocation), std::move(submitted)));
    HYLE_ASSIGN_OR_RETURN(auto protocol, source.take(std::move(update)));
    HYLE_ASSIGN_OR_RETURN(auto input, c::detail::CopyStreamAccess::input(source.update(), {1}));
    HYLE_ASSIGN_OR_RETURN(auto output_view, output.view());
    HYLE_ASSIGN_OR_RETURN(auto reading, uses.begin());
    HYLE_RETURN_IF_ERROR(c::detail::CopyStreamAccess::admit_read(reading, source.update(), {1}));
    HYLE_RETURN_IF_ERROR(reading.write(output));
    HYLE_ASSIGN_OR_RETURN(auto suffix, c::copy_async(queue, output_view, input.view));
    HYLE_RETURN_IF_ERROR(reading.attach(suffix));
    auto completion = observer.publish(std::move(suffix), protocol);
    HYLE_RETURN_IF_ERROR(completion.wait());
    std::array<std::byte, 16> actual{};
    HYLE_RETURN_IF_ERROR(output.view()->read(actual));
    for (auto value : actual) HYLE_RETURN_IF_ERROR(require(value == std::byte{41}, "SDK source/suffix oracle differs"));
    HYLE_ASSIGN_OR_RETURN(invocation, source.update().release_invocation());
    HYLE_RETURN_IF_ERROR(invocation.reset());
    HYLE_RETURN_IF_ERROR(require(source.available() && metadata.used() == cold_bytes,
        "SDK cold source lost original invocation or metadata"));
  }
  std::fprintf(stderr, "cold source SDK: original CPU source/F2/suffix, original invocation reused64 PASS; metadata=%zu\n", cold_bytes);
  return hyle::Status::Ok();
}
hyle::Status run() {
  HYLE_RETURN_IF_ERROR(copy_source());
  hyle::mem::QuotaAllocator metadata(hyle::mem::system_allocator());
  HYLE_ASSIGN_OR_RETURN(auto observer, c::detail::DependentCompletion::create(metadata.get()));
  HYLE_ASSIGN_OR_RETURN(auto slot, c::detail::ResultSlot<int>::create(metadata.get()));
  const auto cold = metadata.used();
  for (int episode = 0; episode < 64; ++episode) {
    HYLE_ASSIGN_OR_RETURN(auto write, slot.begin());
    HYLE_RETURN_IF_ERROR(write.emplace(int{episode}));
    HYLE_ASSIGN_OR_RETURN(auto result, write.publish(c::Completion::completed()));
    InlineSource source{episode & 1 ? hyle::Status::Cancelled("source cancelled")
                                  : hyle::Status::Ok()};
    auto &suffix = c::detail::ResultAccess::completion(result);
    auto original = c::detail::ResultAccess::replace_completion(
        result, observer.publish(std::move(suffix), source.protocol()));
    HYLE_RETURN_IF_ERROR(require(!original.valid(), "original completion duplicated"));
    const auto completion_status = result.completion().wait();
    if (episode & 1) {
      HYLE_RETURN_IF_ERROR(require(completion_status.IsCancelled() &&
          result.wait().error().IsCancelled() && result.result().error().IsCancelled(),
          "source cancellation verdict differs between original result methods"));
    } else {
      HYLE_RETURN_IF_ERROR(completion_status);
      HYLE_ASSIGN_OR_RETURN(auto lease, result.wait());
      HYLE_ASSIGN_OR_RETURN(auto repeated, result.result());
      HYLE_RETURN_IF_ERROR(require(*lease == episode && repeated.get() == lease.get(),
          "original payload identity changed"));
      result = {};
      HYLE_RETURN_IF_ERROR(require(!slot.begin(), "old result permits slot overwrite"));
    }
    result = {};
    HYLE_RETURN_IF_ERROR(require(observer.available() && source.releases == 1 &&
        metadata.used() == cold, "source reference or cold metadata leaked"));
  }
  std::fprintf(stderr, "owned outcome SDK: original completion/result/wait, same slot, "
      "64 episodes PASS; cold observer+result bytes=%zu, Completion=%zu, OwnedResult<int>=%zu\n",
      cold, sizeof(c::Completion), sizeof(c::OwnedResult<int>));
  return hyle::Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
