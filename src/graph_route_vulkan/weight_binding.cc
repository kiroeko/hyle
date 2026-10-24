#include "hyle/graph/vulkan.h"

#include "hyle/graph/detail/input_source.h"
#include "hyle/container/small_vector.h"
#include "hyle/util/ref_counted.h"
#include "../compute_route_vulkan/vulkan_weights_internal.h"

namespace hyle::graph::vulkan {
namespace {
namespace vd = compute::vulkan::detail;
namespace gd = graph::detail;
struct WeightSource final : util::RefCounted<WeightSource> {
  explicit WeightSource(vd::WeightBinding value) noexcept : binding(std::move(value)) {}
  void destroy() noexcept {
    const auto allocator = binding.allocator;
    allocator.destroy(this);
  }
  vd::WeightBinding binding;
};
const gd::InputSourceVTable kWeightSource{
    [](void *state) noexcept { static_cast<WeightSource *>(state)->incref(); },
    [](void *state) noexcept { static_cast<WeightSource *>(state)->decref(); },
    [](void *, const compute::detail::QueueRef *) noexcept
        -> Result<gd::InputDependencyRoute> {
      // The native Vulkan upload has no admitted Compute QueueSequence. Do
      // not invent one or drop its terminal status by treating queue order as
      // a complete imported dependency. The original Graph wait policy owns
      // this explicit host-ready frontier, including optional host progress.
      return gd::InputDependencyRoute::kHostWait;
    },
    [](void *state) noexcept {
      return vd::WeightAccess::poll(static_cast<WeightSource *>(state)->binding.owner.get());
    },
    [](void *state) noexcept {
      return vd::WeightAccess::wait(static_cast<WeightSource *>(state)->binding.owner.get());
    },
    [](void *, compute::detail::QueueSequence &) noexcept {
      return compute::detail::EnqueueResult::rejected(
          Status::Unimplemented("Vulkan weight source has no shared Queue import"));
    },
    [](void *) noexcept { return Status::Ok(); }};

Result<gd::InputSourceBinding> input_binding(ValueId value,
    vd::WeightBinding binding) noexcept {
  const auto tensor = binding.tensor;
  const auto allocator = binding.allocator;
  auto *owner = allocator.construct<WeightSource>(std::move(binding));
  if (!owner) return std::unexpected(Status::ResourceExhausted("Vulkan immutable input owner"));
  auto source = gd::InputSource::acquire(owner, &kWeightSource, owner->binding.owner.get());
  owner->decref(); // Drop construction, including a rejected acquire.
  if (!source) return std::unexpected(std::move(source).error());
  return gd::InputSourceBinding{value, tensor, std::move(*source)};
}
} // namespace

Status bind(Invocation &invocation, ValueId value,
    const compute::vulkan::WeightBatch &batch, std::uint64_t id) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto binding, vd::WeightAccess::binding(batch, id));
  HYLE_ASSIGN_OR_RETURN(auto input, input_binding(value, std::move(binding)));
  return gd::InputAccess::bind(invocation, value, input.tensor, std::move(input.source));
}

Status bind(Invocation &invocation, std::span<const graph::WeightBinding> selection,
    const compute::vulkan::WeightBatch &batch) noexcept {
  if (selection.empty())
    return Status::InvalidArgument("Vulkan weight binding bundle is empty");
  HYLE_ASSIGN_OR_RETURN(auto first, vd::WeightAccess::binding(batch, selection[0].id));
  container::SmallVector<gd::InputSourceBinding, 4, mem::AllocatorRef> bindings(first.allocator);
  if (!bindings.try_resize(selection.size()))
    return Status::ResourceExhausted("Vulkan weight binding bundle metadata");
  HYLE_ASSIGN_OR_RETURN(bindings[0], input_binding(selection[0].value, std::move(first)));
  for (std::size_t i = 1; i < selection.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto binding, vd::WeightAccess::binding(batch, selection[i].id));
    HYLE_ASSIGN_OR_RETURN(bindings[i], input_binding(selection[i].value, std::move(binding)));
  }
  return gd::InputAccess::bind(invocation,
      std::span<const gd::InputSourceBinding>(bindings.data(), bindings.size()));
}
} // namespace hyle::graph::vulkan
