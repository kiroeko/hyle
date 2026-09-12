# Graph v3 compiles a typed immutable DAG into provider regions or explicit
# single-operation fallback steps plus one Mem lifetime plan. The plan remains
# internal; Invocation is the only mutable per-run state.
hyle_add_module(graph SOURCES graph.cc executable.cc)
