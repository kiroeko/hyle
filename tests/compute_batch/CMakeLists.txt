hyle_add_header_self_contained_test(compute_batch HEADER_DIR compute
  HEADERS batch_layout.h batch_ops.h batch_execution.h detail/batch_provider.h detail/batch_graph_access.h batch_materialize.h)
hyle_add_test(compute_batch SOURCES batch_layout_test.cc batch_materialize_test.cc DEPS hyle_compute_state)
