# Graph v1 compiles one immutable logical RMSNorm node into a shareable Compute
# artifact. The plan remains internal; Invocation is the public run state.
hyle_add_module(graph SOURCES graph.cc executable.cc)
