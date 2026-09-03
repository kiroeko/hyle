find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(mem)
hyle_add_test(mem SOURCES allocator_test.cc buffer_resource_test.cc buffer_pool_test.cc composite_buffer_test.cc heap_test.cc lifetime_plan_test.cc shared_pool_test.cc pool_stack_test.cc alignment_test.cc pmr_test.cc std_allocator_test.cc growable_pool_test.cc stats_test.cc
  DEPS Threads::Threads)
target_include_directories(hyle_mem_test PRIVATE "${PROJECT_SOURCE_DIR}/src/mem")
