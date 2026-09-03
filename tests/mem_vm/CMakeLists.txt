hyle_add_header_self_contained_test(mem_vm
  HEADER_DIR mem
  HEADERS fiber_stack.h hugepage.h mapping.h)
hyle_add_test(mem_vm SOURCES api_surface_test.cc mapping_test.cc
  file_mapping_test.cc fiber_stack_test.cc hugepage_test.cc numa_test.cc)
