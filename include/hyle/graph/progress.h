#pragma once

#include "hyle/error/status.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

namespace hyle::sched {
class Scheduler;
class BlockingPool;
} // namespace hyle::sched

namespace hyle::graph {
// Cold, optional host progress for this original Invocation. Replay still uses
// its compiled targets, lanes and Queue bindings. The Scheduler coordinates the
// existing dependency frontier; the BlockingPool runs actual blocking segments.
// Both runtimes must outlive the invocation's admitted progress and drain.
// Ordinary submit without this preparation keeps its direct execution path.
// Posting each run uses the Scheduler's existing detached task/stack allocator;
// it does not promise allocation-free task admission.
HYLE_GRAPH_PROGRESS_API Status prepare_progress(
    Invocation &, sched::Scheduler &, sched::BlockingPool &) noexcept;
} // namespace hyle::graph
