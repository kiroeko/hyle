#include <hyle/compute/detail/dependent_completion.h>
#include <hyle/compute/detail/generated_result.h>
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
hyle::Status run() {
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
