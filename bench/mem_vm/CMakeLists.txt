if(HYLE_STATIC AND HYLE_BENCH_BOOST)
  function(_hyle_configure_mem_vm_mapping_bench target)
    target_link_libraries(${target} PRIVATE hyle_mem_vm nanobench
                          hyle_bench_boost_headers)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_vm_mapping_bench mapping_consumer_bench.cc)
  _hyle_configure_mem_vm_mapping_bench(hyle_mem_vm_mapping_bench)

  add_executable(hyle_mem_vm_mapping_bench_placement
    mapping_consumer_bench.cc)
  _hyle_configure_mem_vm_mapping_bench(
    hyle_mem_vm_mapping_bench_placement)
  if(MSVC)
    target_link_options(hyle_mem_vm_mapping_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_mem_vm_mapping_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "mem_vm mapping placement perturbation is unsupported")
  endif()
  hyle_register_benches(mem_vm
    hyle_mem_vm_mapping_bench
    hyle_mem_vm_mapping_bench_placement)
endif()

if(WIN32 AND HYLE_STATIC)
  function(_hyle_configure_mem_vm_residency_bench target)
    target_link_libraries(${target} PRIVATE hyle_mem_vm nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_vm_residency_bench residency_bench.cc)
  _hyle_configure_mem_vm_residency_bench(hyle_mem_vm_residency_bench)

  add_executable(hyle_mem_vm_residency_bench_placement residency_bench.cc)
  _hyle_configure_mem_vm_residency_bench(
    hyle_mem_vm_residency_bench_placement)
  if(MSVC)
    target_link_options(hyle_mem_vm_residency_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_mem_vm_residency_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "mem_vm residency placement perturbation is unsupported")
  endif()
  hyle_register_benches(mem_vm
    hyle_mem_vm_residency_bench
    hyle_mem_vm_residency_bench_placement)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND HYLE_STATIC)
  function(_hyle_configure_mem_vm_model_loader_bench target)
    target_link_libraries(${target} PRIVATE hyle_mem_vm nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_vm_model_loader_bench
    model_loader_mapping_bench.cc)
  _hyle_configure_mem_vm_model_loader_bench(
    hyle_mem_vm_model_loader_bench)

  add_executable(hyle_mem_vm_model_loader_bench_placement
    model_loader_mapping_bench.cc)
  _hyle_configure_mem_vm_model_loader_bench(
    hyle_mem_vm_model_loader_bench_placement)
  target_compile_options(hyle_mem_vm_model_loader_bench_placement PRIVATE
    -falign-functions=64)

  hyle_register_benches(mem_vm
    hyle_mem_vm_model_loader_bench
    hyle_mem_vm_model_loader_bench_placement)
endif()
