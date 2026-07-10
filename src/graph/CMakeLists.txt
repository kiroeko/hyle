# graph-01 (step-1): task-dependency DAG with dirty-flag incremental recompute.
# North-star ② (CAD/BIM/CAE parametric/simulation dependency recompute). Pure CPU,
# serial evaluate -- NO sched dependency yet (parallel node execution over the sched
# work-stealing pool is step-2). Depends only on error (Status) + mem (Allocator) +
# container (SmallVector) PUBLICLY (all appear in taskgraph.h). See
# docs/architecture/graph-design.md + docs/milestones/graph-01-incremental-recompute.md.
hyle_add_module(graph
  SOURCES taskgraph.cc
  PUBLIC_DEPS hyle_error hyle_mem hyle_container)
