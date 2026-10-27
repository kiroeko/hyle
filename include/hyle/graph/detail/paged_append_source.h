#pragma once

#include "hyle/graph/batch.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/resource_source.h"

namespace hyle::graph::batch::detail {
struct PagedAppendSourceData;
struct PagedAppendSourceAccess;
// Same-build typed view of one original append transaction. The common
// ResourceSource retains the sole submission verdict and actual permissions;
// this owner retains only its arena/layout/prefix interpretation.
class PagedAppendSource final {
public:
  PagedAppendSource() noexcept = default;
  PagedAppendSource(const PagedAppendSource &) = delete;
  PagedAppendSource &operator=(const PagedAppendSource &) = delete;
  HYLE_GRAPH_BATCH_API PagedAppendSource(PagedAppendSource &&) noexcept;
  HYLE_GRAPH_BATCH_API PagedAppendSource &operator=(PagedAppendSource &&) noexcept;
  HYLE_GRAPH_BATCH_API ~PagedAppendSource();
  // Cold fixed metadata, before accepting the producer. No writer is retained
  // or sealed. capture rechecks the actual selected transaction of that submit.
  HYLE_GRAPH_BATCH_API static Result<PagedAppendSource> prepare(Invocation &, NodeId,
      const compute::StorageOwner &, PagedAppendOptions = {}) noexcept;
  bool prepared() const noexcept { return data_ != nullptr; }
  HYLE_GRAPH_BATCH_API bool valid() const noexcept;
  HYLE_GRAPH_BATCH_API Status capture(const graph::detail::SubmissionSource &) noexcept;
  // No wait: whole success plus actual retirement and matching returned
  // owners authorize the original page/frontier publication. Retry preserves
  // already published guards. Fresh appendable conversion closes full-ticket
  // admission; existing readers may retire, later readers need real ready views.
  HYLE_GRAPH_BATCH_API Result<compute::PagedArenaRead> ready(PagedAppendPages &) const noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::PagedArenaDelta> ready_delta(PagedAppendPages &,
      compute::PagedArenaFootprint, compute::BatchLayoutVersion accepted) const noexcept;
private:
  friend struct PagedAppendSourceAccess;
  explicit PagedAppendSource(PagedAppendSourceData *data) noexcept : data_(data) {}
  PagedAppendSourceData *data_ = nullptr;
};
// Cold once per typed view, after B's ordinary batch::prepare. Requires the
// shared reader's explicit, owned ready metadata input. B's query
// metadata may differ from A; every referenced physical prefix must be covered.
// Returns a call descriptor whose spans remain owned by source. The caller
// combines it with token/Store views in the original single bind_outputs call.
HYLE_GRAPH_BATCH_API Result<graph::detail::OutputBinding> reader_binding(Invocation &,
    ValueId bank, PagedAppendSource &, compute::BatchLayoutVersion read_layout) noexcept;
} // namespace hyle::graph::batch::detail
