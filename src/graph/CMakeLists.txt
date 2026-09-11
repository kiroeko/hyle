# Graph v2 compiles a typed immutable DAG into shareable prepared Compute
# artifacts and a Mem lifetime plan. The plan remains internal; Invocation is
# the only mutable per-run state.
hyle_add_module(graph SOURCES graph.cc executable.cc)
