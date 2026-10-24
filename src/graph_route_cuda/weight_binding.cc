#include "hyle/graph/cuda.h"

#include "hyle/graph/detail/input_source.h"
#include "hyle/container/small_vector.h"
#include "weights_internal.h"

namespace hyle::graph::cuda {
namespace {
namespace wd = compute::cuda::detail;
namespace gd = graph::detail;

const gd::InputSourceVTable kWeightSource{
    [](void *state) noexcept { wd::retain_weight(static_cast<wd::WeightState *>(state)); },
    [](void *state) noexcept { wd::release_weight(static_cast<wd::WeightState *>(state)); },
    [](void *state, const compute::detail::QueueRef *queue) noexcept
        -> Result<gd::InputDependencyRoute> {
      HYLE_ASSIGN_OR_RETURN(auto native, wd::weight_device_route(
          static_cast<wd::WeightState *>(state), queue));
      return native ? gd::InputDependencyRoute::kDeviceWait
                    : gd::InputDependencyRoute::kHostWait;
    },
    [](void *state) noexcept {
      return wd::poll_weight(static_cast<wd::WeightState *>(state));
    },
    [](void *state) noexcept {
      return wd::wait_weight(static_cast<wd::WeightState *>(state));
    },
    [](void *state, compute::detail::QueueSequence &sequence) noexcept {
      return wd::enqueue_weight(static_cast<wd::WeightState *>(state), sequence);
    },
    [](void *state) noexcept {
      return wd::consumed_weight_status(static_cast<wd::WeightState *>(state));
    }};
} // namespace

Status bind(Invocation &invocation, ValueId value,
    const compute::cuda::WeightBatch &batch, std::uint64_t id) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto tensor, batch.binding(id));
  HYLE_ASSIGN_OR_RETURN(auto source, gd::InputSource::acquire(
      wd::WeightAccess::state(batch), &kWeightSource, wd::WeightAccess::state(batch)));
  return gd::InputAccess::bind(invocation, value, tensor, std::move(source));
}

Status bind(Invocation &invocation, std::span<const graph::WeightBinding> selection,
    const compute::cuda::WeightBatch &batch) noexcept {
  if (selection.empty())
    return Status::InvalidArgument("CUDA weight binding bundle is empty");
  // Cold adapter metadata only. The common binding transaction uses the
  // Invocation's allocator and owns all actual mutation/rollback decisions.
  container::SmallVector<gd::InputSourceBinding, 4> bindings;
  if (!bindings.try_resize(selection.size()))
    return Status::ResourceExhausted("CUDA weight binding bundle metadata");
  for (std::size_t i = 0; i < selection.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto tensor, batch.binding(selection[i].id));
    HYLE_ASSIGN_OR_RETURN(auto source, gd::InputSource::acquire(
        wd::WeightAccess::state(batch), &kWeightSource, wd::WeightAccess::state(batch)));
    bindings[i] = {selection[i].value, tensor, std::move(source)};
  }
  return gd::InputAccess::bind(invocation,
      std::span<const gd::InputSourceBinding>(bindings.data(), bindings.size()));
}
} // namespace hyle::graph::cuda
