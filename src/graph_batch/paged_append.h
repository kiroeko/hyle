#pragma once
#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_adapter.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/compute/detail/paged_append_backend.h"
#include "native_append_regions.h"

namespace hyle::graph::batch::detail {
// Exact original carrier -> logical token rows; no storage/content conversion.
HYLE_GRAPH_BATCH_API Result<std::uint32_t> append_query_rows(
    const PagedAppend&, const compute::TensorDesc&) noexcept;
struct AppendAdapter final {
  mem::MemoryResourceRef resource;
  compute::DeviceClass device_class=compute::DeviceClass::kCpu;
  compute::detail::BackendExtensionRef native;
  bool native_capture=false;
};
struct AppendFacts final {
  compute::BatchLayoutVersion layout;
  compute::BatchCacheRole role=compute::BatchCacheRole::kKey;
  compute::ConstTensorView capacity;
  compute::CacheStorage storage; // Actual bank interpretation, not layout provenance.
};
struct AppendQueueBinding final {
  ExecutionLaneId lane;
  compute::detail::QueueRef queue;
};
struct PagedAppendAccess final {
  static PagedAppendData* data(PagedAppendPages&p)noexcept{return p.data_;}
  static const RegionProviderVTable* provider()noexcept;
  static bool selected(const graph::detail::RegionAccessView&)noexcept;
  static bool needs_queue(const graph::detail::RegionAccessView&)noexcept;
  static Result<NativeAppendRegionRef> native(Invocation&,NodeId)noexcept;
  static Result<AppendFacts> facts(const PagedAppendPages&)noexcept;
  static Status bind(Invocation&,const Graph&,NodeId,ValueId,PagedAppendPages&&,
      std::span<const graph::detail::ProjectedReader>,const compute::detail::QueueRef&,
      mem::MemoryResourceRef host_workspace,ResourceOptions)noexcept;
  static Status bind(Invocation&,const Graph&,NodeId,PagedAppendPages&&,
      std::span<const AppendQueueBinding>,mem::MemoryResourceRef host_workspace,
      ResourceOptions)noexcept;
  static Result<PagedAppendPages> take(Invocation&,NodeId)noexcept;
  static Status reset(Invocation&)noexcept;
};
} // namespace hyle::graph::batch::detail
