hyle_add_header_self_contained_test(mem_vm
  HEADER_DIR mem
  HEADERS fiber_stack.h large_mapping.h mapping.h)
hyle_add_test(mem_vm SOURCES api_surface_test.cc mapping_test.cc
  file_mapping_test.cc fiber_stack_test.cc large_mapping_test.cc numa_test.cc)

if(TARGET hyle_mem_vm_test)
  add_executable(hyle_mem_vm_shared_mapping_peer shared_mapping_peer.cc)
  target_link_libraries(hyle_mem_vm_shared_mapping_peer PRIVATE hyle_mem_vm)
  set_target_properties(hyle_mem_vm_shared_mapping_peer PROPERTIES
    FOLDER "hyle/tests/mem_vm"
    HYLE_TEST_MODULE "mem_vm")
  hyle_apply_warnings(hyle_mem_vm_shared_mapping_peer)
  hyle_apply_sanitizer(hyle_mem_vm_shared_mapping_peer)
  target_compile_definitions(hyle_mem_vm_test PRIVATE
    HYLE_MEM_VM_SHARED_PEER_PATH="$<TARGET_FILE:hyle_mem_vm_shared_mapping_peer>")
  add_dependencies(hyle_mem_vm_test hyle_mem_vm_shared_mapping_peer)
  add_dependencies(hyle_mem_vm_tests hyle_mem_vm_shared_mapping_peer)
endif()
