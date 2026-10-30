# Model-owned cold implementation inventory. Each consumer compiles these
# sources with its existing Backend capability contract; this is not a shared
# precompiled object library or a registry of model implementations.
set(_hyle_qwen35_cold_sources
  "${CMAKE_CURRENT_LIST_DIR}/weights.cc"
  "${CMAKE_CURRENT_LIST_DIR}/graph_builder.cc"
  "${CMAKE_CURRENT_LIST_DIR}/target_graph.cc"
  "${CMAKE_CURRENT_LIST_DIR}/family_factory.cc"
  "${CMAKE_CURRENT_LIST_DIR}/phase_preparation.cc")
