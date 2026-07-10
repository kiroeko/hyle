#include "hyle/graph/taskgraph.h"

#include <new>
#include <utility>

namespace hyle::graph {

namespace {
inline std::uint32_t idx(NodeId n) noexcept { return static_cast<std::uint32_t>(n); }
}  // namespace

// dep_off_ is the CSR row-pointer array: it holds num_nodes+1 entries, where node i's
// deps occupy dep_pool_[dep_off_[i] .. dep_off_[i+1]). Seed it with a single 0 (the
// start offset of the first node's deps) so add_node only ever appends the new END.
TaskGraph::TaskGraph(mem::Allocator alloc) noexcept
    : fn_(alloc), ctx_(alloc), dirty_(alloc), dep_pool_(alloc), dep_off_(alloc),
      alloc_(alloc) {
  dep_off_.push_back(0u);
}

TaskGraph::~TaskGraph() = default;

TaskGraph::TaskGraph(TaskGraph&& o) noexcept
    : fn_(std::move(o.fn_)), ctx_(std::move(o.ctx_)), dirty_(std::move(o.dirty_)),
      dep_pool_(std::move(o.dep_pool_)), dep_off_(std::move(o.dep_off_)),
      alloc_(o.alloc_) {
  // Restore the moved-from graph to a VALID EMPTY state, not a broken one. SmallVector's
  // move resets the source to size 0 -- which for dep_off_ means the seed `0` is gone,
  // violating the "dep_off_ has num_nodes+1 entries, always >= 1" invariant. A later
  // add_node on the moved-from graph would then read dep_off_ out of bounds. Re-seed so a
  // moved-from TaskGraph is reusable exactly like a fresh one (HYLE moved-from = valid-empty).
  o.dep_off_.push_back(0u);
}

TaskGraph& TaskGraph::operator=(TaskGraph&& o) noexcept {
  if (this != &o) {
    // SmallVector is move-CONSTRUCTIBLE but not move-ASSIGNABLE (copy deleted, no move=
    // defined). So rebuild in place: destroy this, then placement-move-construct from o.
    this->~TaskGraph();
    new (this) TaskGraph(std::move(o));
  }
  return *this;
}

NodeId TaskGraph::add_node(ComputeFn fn, void* ctx, const NodeId* deps,
                           std::size_t num_deps) noexcept {
  HYLE_FATAL_ASSERT(fn != nullptr);
  const std::uint32_t id = static_cast<std::uint32_t>(fn_.size());
  // decision ②: every dep must be an already-created (strictly smaller) id. This makes
  // insertion order a topological order -- no per-run sort, no cycle detection.
  for (std::size_t i = 0; i < num_deps; ++i) {
    HYLE_FATAL_ASSERT(idx(deps[i]) < id);
    dep_pool_.push_back(deps[i]);
  }
  fn_.push_back(fn);
  ctx_.push_back(ctx);
  dirty_.push_back(true);                                     // fresh node = never computed
  dep_off_.push_back(static_cast<std::uint32_t>(dep_pool_.size()));  // end of this node's deps
  return static_cast<NodeId>(id);
}

void TaskGraph::mark_dirty(NodeId n) noexcept {
  const std::uint32_t start = idx(n);
  HYLE_FATAL_ASSERT(start < fn_.size());
  // Eager downstream flood (decision ③). A node depends only on smaller ids, so a single
  // FORWARD pass from `start` suffices: any node j > start that (transitively) depends on
  // a now-dirty node is caught because its own deps -- all < j -- were visited first and
  // already carry the dirty flag. O(nodes + edges from start on), no compute.
  dirty_[start] = true;
  const std::uint32_t n_nodes = static_cast<std::uint32_t>(fn_.size());
  for (std::uint32_t j = start + 1; j < n_nodes; ++j) {
    if (dirty_[j]) continue;  // already dirty -> its subtree will be handled anyway
    for (std::uint32_t k = dep_off_[j]; k < dep_off_[j + 1]; ++k) {
      if (dirty_[idx(dep_pool_[k])]) {
        dirty_[j] = true;
        break;
      }
    }
  }
}

Status TaskGraph::evaluate() noexcept {
  const std::uint32_t n_nodes = static_cast<std::uint32_t>(fn_.size());
  // Single forward sweep = topological order (decision ②). Run only dirty nodes; a dirty
  // node's deps (smaller ids) were visited first, so its inputs are already fresh. Clean
  // nodes are skipped -- their cached output in user memory is reused (the incremental win).
  for (std::uint32_t i = 0; i < n_nodes; ++i) {
    if (!dirty_[i]) continue;
    Status s = fn_[i](ctx_[i]);
    if (!s.ok()) return s;  // abort sweep; node i stays dirty (retried next evaluate)
    dirty_[i] = false;
  }
  return Status();
}

Status TaskGraph::evaluate_node(NodeId target) noexcept {
  const std::uint32_t t = idx(target);
  HYLE_FATAL_ASSERT(t < fn_.size());

  // 1. Mark target's DEPENDENCY CONE (target + all transitive deps). Because a dep always
  //    has a SMALLER id (decision ②), one BACKWARD sweep from t down to 0 suffices: seed
  //    `needed[t]=true`, then for each i from t down, if i is needed, mark all of i's deps
  //    (ids < i, visited later in this descending sweep) needed too. No recursion, no
  //    revisits. `needed` is scratch (this call only), sized to t+1 (ids > t can't be in a
  //    cone whose root is t -- they'd have larger ids than target, i.e. downstream).
  container::SmallVector<bool, 64, mem::Allocator> needed(alloc_);
  needed.reserve(t + 1);
  for (std::uint32_t i = 0; i <= t; ++i) needed.push_back(false);
  needed[t] = true;
  for (std::uint32_t ri = 0; ri <= t; ++ri) {
    const std::uint32_t i = t - ri;  // descend t, t-1, ..., 0
    if (!needed[i]) continue;
    for (std::uint32_t k = dep_off_[i]; k < dep_off_[i + 1]; ++k) {
      needed[idx(dep_pool_[k])] = true;  // dep id < i, still ahead in the descending sweep
    }
  }

  // 2. Forward sweep 0..=t: run nodes that are BOTH dirty AND needed, in topo order. A run
  //    node's deps (smaller ids, already visited) that were dirty+needed already ran, so
  //    its inputs are fresh. Dirty nodes OUTSIDE the cone (needed[i]==false) are skipped
  //    and STAY dirty -- deferred until something that needs them is evaluated.
  for (std::uint32_t i = 0; i <= t; ++i) {
    if (!needed[i] || !dirty_[i]) continue;
    Status s = fn_[i](ctx_[i]);
    if (!s.ok()) return s;  // abort; node i stays dirty (retried next evaluate)
    dirty_[i] = false;
  }
  return Status();
}

std::size_t TaskGraph::num_nodes() const noexcept { return fn_.size(); }

bool TaskGraph::is_dirty(NodeId n) const noexcept {
  HYLE_FATAL_ASSERT(idx(n) < fn_.size());
  return dirty_[idx(n)];
}

}  // namespace hyle::graph
