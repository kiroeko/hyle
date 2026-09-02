# graph-01: task-dependency DAG with dirty-flag incremental recompute.
# North-star ② (CAD/BIM/CAE parametric/simulation dependency recompute). See
# docs/architecture/graph-design.md + docs/milestones/graph-01-incremental-recompute.md.
#
# step-1 (taskgraph.cc): the SERIAL incremental core -- depends only on error (Status) +
# mem (AllocatorRef) + container (SmallVector), builds ANYWHERE (no fiber asm needed).
# step-2 (parallel_eval.cc): PARALLEL evaluate over the sched work-stealing pool -- added
# only under HYLE_HAS_FCONTEXT (spawn_task/when_all/fiber suspend need sched), exactly as
# compute's gpu_poller.cc is. The current public graph root exposes Scheduler in
# parallel headers and therefore always selects sched; a serial-only product
# would require a future component/header split.
set(_graph_sources taskgraph.cc)

if(HYLE_HAS_FCONTEXT)
  # step-2 parallel evaluate: parallel_eval.cc uses sched spawn_task + JoinHandle + fiber
  # suspend. parallel.h includes it publicly (references sched::Scheduler), so a consumer
  # of that header needs sched -> hyle_sched is PUBLIC on this path.
  list(APPEND _graph_sources parallel_eval.cc)
  # graph-02: data_parallel.cc = parallel_for / parallel_reduce (recursive-split fork-join
  # over sched). Same sched gate; data_parallel.h references sched::Scheduler.
  list(APPEND _graph_sources data_parallel.cc)
endif()

hyle_add_module(graph
  SOURCES ${_graph_sources})
