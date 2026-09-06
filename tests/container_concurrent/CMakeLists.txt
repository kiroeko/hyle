hyle_add_header_self_contained_test(container_concurrent
  HEADER_DIR container
  HEADERS
    concurrent_flat_hash_map.h
    concurrent_flat_hash_set.h
    concurrent_hash_map.h
    concurrent_node_hash_map.h
    concurrent_node_hash_set.h
    concurrent_vector.h)
hyle_add_test(container_concurrent SOURCES
  adaptive_mutex_test.cc
  concurrent_hash_map_test.cc
  concurrent_node_hash_map_test.cc
  concurrent_vector_test.cc)
