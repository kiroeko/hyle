#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <utility>

#include "backend.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::inference {

// Shared cold storage for parked complete states, independent of Model and its
// device inventories. ResourceBudget is the only byte-admission authority;
// this owner adds only an independent image-count bound.
class HostStateStorage final : public std::enable_shared_from_this<HostStateStorage> {
  struct CreateKey final {};
  struct ImageCount final {
    explicit ImageCount(std::shared_ptr<HostStateStorage> retained) noexcept
        : storage(std::move(retained)) {}
    ~ImageCount() { storage->live_.fetch_sub(1, std::memory_order_release); }
    std::shared_ptr<HostStateStorage> storage;
  };
public:
  // The private key keeps construction on create(), while allowing the real
  // ResourceBudget metadata allocator to construct the shared control block.
  HostStateStorage(CreateKey, compute::StorageOwner storage, compute::ResourceBudget budget,
      std::size_t maximum_images) noexcept
      : storage_(std::move(storage)), budget_(std::move(budget)), maximum_images_(maximum_images) {}
  HostStateStorage(const HostStateStorage &) = delete;
  HostStateStorage &operator=(const HostStateStorage &) = delete;

  static Result<std::shared_ptr<HostStateStorage>> create(Backend &backend,
      std::size_t max_images, std::size_t max_bytes) noexcept {
    if (!max_images || !max_bytes)
      return std::unexpected(Status::InvalidArgument("host-state storage requires positive image and byte bounds"));
    HYLE_ASSIGN_OR_RETURN(auto storage, backend.create_host_storage());
    const auto resource = storage.ref();
    // Pinned CUDA host storage deliberately does not promise concurrent
    // host/device coherence. Original CopyBatch completion establishes its
    // visibility boundary; host reads use the resource's own read/map API.
    if (!resource.properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("host-state storage must expose host-readable bytes"));
    const auto metadata = mem::system_resource();
    std::array<compute::ResourceBudgetLimit, 2> limits{{{resource, max_bytes}, {metadata, max_bytes}}};
    std::array<mem::MemoryResourceRef, 2> nominated{resource, metadata};
    const auto count = resource == metadata ? std::size_t{1} : std::size_t{2};
    // Actual resource identities stay separate. CPU metadata shares the host
    // row; CUDA pinned payload and real system metadata share one existing
    // requested-bytes cap. max_bytes is not a promise about opaque driver RAM.
    HYLE_ASSIGN_OR_RETURN(auto budget, compute::ResourceBudget::create(
        {.limits = std::span(limits).first(count), .metadata_resource = metadata,
         .allocator = mem::system_allocator(), .allow_unattributed = true},
        std::span(nominated).first(count), max_bytes));
    HYLE_ASSIGN_OR_RETURN(auto allocator, budget.metadata_allocator());
    try {
      return std::allocate_shared<HostStateStorage>(mem::StdAllocator<HostStateStorage>(allocator),
          CreateKey{}, std::move(storage), std::move(budget), max_images);
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("host-state storage metadata"));
    }
  }

  Result<std::shared_ptr<const void>> reserve_image() noexcept {
    auto retained = weak_from_this().lock();
    if (!retained)
      return std::unexpected(Status::FailedPrecondition("host-state storage has no shared owner"));
    HYLE_ASSIGN_OR_RETURN(auto allocator, budget_.metadata_allocator());
    auto count = live_.load(std::memory_order_acquire);
    for (;;) {
      if (count >= maximum_images_)
        return std::unexpected(Status::ResourceExhausted("host-state image count is exhausted"));
      if (live_.compare_exchange_weak(count, count + 1,
          std::memory_order_acq_rel, std::memory_order_acquire)) break;
    }
    try {
      std::shared_ptr<const void> token = std::allocate_shared<ImageCount>(
          mem::StdAllocator<ImageCount>(allocator), std::move(retained));
      return token;
    } catch (const std::bad_alloc &) {
      live_.fetch_sub(1, std::memory_order_release);
      return std::unexpected(Status::ResourceExhausted("host-state image-count metadata"));
    }
  }

  std::size_t live_images() const noexcept { return live_.load(std::memory_order_acquire); }
  mem::MemoryResourceRef resource() const noexcept { return storage_.ref(); }
  const compute::ResourceBudget &budget() const noexcept { return budget_; }
  const void *context_identity() const noexcept { return storage_.context_identity(); }
  Status validate_backend(Backend &backend) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto *context, backend.common_context());
    HYLE_ASSIGN_OR_RETURN(auto original, compute::detail::ContextRef::acquire(*context));
    return original.identity_key() == context_identity() ? Status::Ok()
        : Status::InvalidArgument("host-state storage belongs to another execution Context");
  }
  Result<compute::ResourceBudgetUsage> usage() const noexcept { return budget_.usage(resource()); }

private:
  // Reverse destruction leaves the original resource/Context after accounting.
  compute::StorageOwner storage_;
  compute::ResourceBudget budget_;
  std::size_t maximum_images_;
  std::atomic<std::size_t> live_{0};
};

} // namespace hyle::inference
