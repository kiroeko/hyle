hyle_add_header_self_contained_test(container)
hyle_add_test(container SOURCES bounded_queue_test.cc spsc_queue_test.cc sharded_queue_test.cc work_stealing_deque_test.cc ws_fifo_queue_test.cc hash_test.cc pod_vector_test.cc small_vector_test.cc static_vector_test.cc probe_test.cc swiss_table_test.cc string_view_test.cc string_buf_test.cc concurrent_vector_test.cc concurrent_hash_map_test.cc concurrent_node_hash_map_test.cc string_algo_test.cc path_test.cc dynamic_bitset_test.cc sparse_set_test.cc ring_deque_test.cc circular_buffer_test.cc byte_ring_buffer_test.cc btree_test.cc flat_sorted_test.cc adaptive_mutex_test.cc DEPS hyle_platform)

# A fresh executable process must receive a different best-effort default
# StringHash seed. A CMake driver gives Windows and POSIX the same exec (rather
# than fork-only) contract and keeps this property independent from GTest's
# platform-specific death-test launcher.
add_executable(hyle_container_hash_seed_probe hash_seed_probe.cc)
target_link_libraries(hyle_container_hash_seed_probe PRIVATE hyle_container)
hyle_apply_warnings(hyle_container_hash_seed_probe)
hyle_apply_sanitizer(hyle_container_hash_seed_probe)
set_target_properties(hyle_container_hash_seed_probe PROPERTIES
  FOLDER "hyle/tests/container"
  HYLE_TEST_MODULE "container")
add_test(NAME hyle_container_hash_seed_process_test
  COMMAND "${CMAKE_COMMAND}"
          "-DPROBE=$<TARGET_FILE:hyle_container_hash_seed_probe>"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/hash_seed_process_test.cmake")
set_tests_properties(hyle_container_hash_seed_process_test PROPERTIES
  LABELS "container;hash")
hyle_ensure_module_test_target(container)
add_dependencies(hyle_container_tests hyle_container_hash_seed_probe)
