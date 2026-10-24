#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include "hyle/compute/cuda.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/tensor.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/compute/weight_assembly.h"
#include "hyle/mem/allocator.h"

namespace hyle::compute::cuda {
namespace detail {
struct WeightState;
struct WeightAccess;
}

struct WeightSpec final {
  std::uint64_t id = 0;
  compute::TensorDesc description{};
};

// Each source is host-accessible, immutable for upload()'s synchronous staging
// copy, and has the exact registered physical description. No quantization or
// conversion is performed here. Repeated IDs in one selection are rejected.
struct WeightFragment final {
  std::uint64_t id = 0;
  compute::ConstTensorView source{};
};

struct WeightSlotOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  // Optional scope shared with Graph. Its metadata allocator takes precedence
  // over allocator; both native backings are reserved before allocation.
  compute::ResourceBudget resource_budget{};
  // Only the sliced assembly API uses this limit. Whole-entry upload retains
  // its original catalog-sized selection capacity.
  std::size_t max_fragments = 64;
};

// A selected device batch and its real upload prefix. binding() may describe
// pending bytes: use order() or wait() before consuming them. Graph users call
// graph::cuda::bind(invocation, value, batch, id) from hyle/graph/cuda.h. It
// retains the selected immutable owner on the ordinary graph::Invocation and
// orders its upload at actual readers during submit(QueueBinding span).
// The source owner coexists with Graph page/resource bindings; reset/rebind
// releases it after the ordinary result and in-flight guards permit mutation.
// Direct consumers retain the batch through their final compute Completion.
class WeightBatch final {
public:
  WeightBatch() noexcept = default;
  WeightBatch(const WeightBatch &) = delete;
  WeightBatch &operator=(const WeightBatch &) = delete;
  HYLE_COMPUTE_ROUTE_CUDA_API WeightBatch(WeightBatch &&other) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API WeightBatch &operator=(WeightBatch &&other) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API ~WeightBatch();
  HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::ConstTensorView>
  binding(std::uint64_t id) const noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::CompletionPoll> poll() const noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Status wait() const noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::WeightUploadReport>
  report() const noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::Completion>
  order(compute::Queue &destination) const noexcept;

private:
  friend class WeightSlots;
  friend struct detail::WeightAccess;
  explicit WeightBatch(detail::WeightState *state) noexcept : state_(state) {}
  detail::WeightState *state_ = nullptr;
};

// Two preallocated pinned/device slots, each with capacity_bytes and a fixed
// catalog. IDs select fragments; the engine owns placement/cache/eviction.
// Slot reuse is nonblocking and rejected while an upload, batch or bound
// consumer is still live. upload() never grows metadata or allocation capacity.
// All handles are thread-compatible. Context/resources/allocator are borrowed
// through every batch and in-flight native prefix. close() drains uploads only.
class WeightSlots final {
public:
  WeightSlots() noexcept = default;
  WeightSlots(const WeightSlots &) = delete;
  WeightSlots &operator=(const WeightSlots &) = delete;
  HYLE_COMPUTE_ROUTE_CUDA_API WeightSlots(WeightSlots &&other) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API WeightSlots &operator=(WeightSlots &&other) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API ~WeightSlots();
  HYLE_COMPUTE_ROUTE_CUDA_API static Result<WeightSlots>
  create(const compute::Context &context,
         const compute::cuda::MemoryResource &host,
         const compute::cuda::MemoryResource &device,
         std::span<const WeightSpec> catalog, std::size_t capacity_bytes,
         WeightSlotOptions options = {}) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Result<WeightBatch>
  upload(std::size_t slot, std::span<const WeightFragment> selection,
         compute::Queue &transfer_queue) noexcept;
  // Cover each selected bank exactly, without holes or overlapping slices.
  // Plain banks use dense plain fragments of the registered scalar type;
  // encoded banks use dense plain U8 fragments of their original stored bytes.
  // The registered bank description is preserved. No conversion, repacking
  // or global/local ID remap occurs.
  HYLE_COMPUTE_ROUTE_CUDA_API Result<WeightBatch>
  upload_slices(std::size_t slot, std::span<const compute::WeightSlice> selection,
                compute::Queue &transfer_queue) noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::WeightUploadReport>
  report(std::size_t slot) const noexcept;
  HYLE_COMPUTE_ROUTE_CUDA_API Status close() noexcept;

private:
  detail::WeightState *states_[2]{};
  bool closed_ = false;
};

} // namespace hyle::compute::cuda
