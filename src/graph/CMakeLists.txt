# Graph compiles a typed immutable DAG into a specialization-specific,
# multi-domain execution plan. The plan remains internal; Invocation owns the
# mutable bindings, workspace, replay routes and capture state for one lane.
hyle_add_module(graph SOURCES graph.cc effect_order.cc executable.cc)
