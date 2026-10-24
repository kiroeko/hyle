# Graph compiles a typed immutable DAG into a specialization-specific,
# multi-domain execution plan. The plan remains internal; Invocation owns the
# mutable bindings, workspace, replay routes and capture state for one lane.
hyle_add_module(graph TRACE TESTING SOURCES graph.cc effect_order.cc executable.cc submission.cc execution_source.cc view_domain.cc constant_artifact.cc resources.cc lease_submission.cc region_access.cc input_sources.cc resource_report.cc)
