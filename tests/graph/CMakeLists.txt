# graph-01 (step-1): task-dependency DAG incremental recompute. Pure CPU, serial evaluate
# -- no sched dep yet (parallel node execution is step-2).
hyle_add_test(graph SOURCES taskgraph_test.cc DEPS hyle_graph)
if(TARGET hyle_sched)
  # graph-01 step-2: parallel evaluate over the sched work-stealing pool (level-by-level
  # fork-join). Needs sched (spawn_task/when_all/fiber). Same gate as sched itself.
  hyle_add_test(graph NAME parallel SOURCES parallel_test.cc DEPS hyle_sched)
  # graph-02 step-1: data-parallel layer (parallel_for/parallel_reduce, recursive-split
  # fork-join over sched). Same sched gate.
  hyle_add_test(graph NAME data_parallel SOURCES data_parallel_test.cc DEPS hyle_sched)
endif()
