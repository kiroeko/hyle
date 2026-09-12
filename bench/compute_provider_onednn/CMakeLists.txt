if(HYLE_STATIC AND HYLE_COMPUTE_ONEDNN_ENABLED)
  function(_hyle_configure_compute_onednn_provider_bench target)
    # dnnl is linked explicitly because the peer arm calls the exact private
    # BRGEMM ukernel used by the provider. It remains benchmark-only and never
    # changes the public Compute dependency surface.
    target_link_libraries(${target} PRIVATE
      hyle_compute_provider_onednn hyle_compute_backend_cpu nanobench dnnl)
    set_property(TARGET ${target}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_compute_onednn_provider_bench
    provider_p1_bench.cc)
  _hyle_configure_compute_onednn_provider_bench(
    hyle_compute_onednn_provider_bench)

  add_executable(hyle_compute_onednn_provider_bench_placement
    provider_p1_bench.cc)
  _hyle_configure_compute_onednn_provider_bench(
    hyle_compute_onednn_provider_bench_placement)
  if(MSVC)
    target_link_options(hyle_compute_onednn_provider_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      hyle_compute_onednn_provider_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Compute oneDNN provider placement perturbation is unsupported")
  endif()

  hyle_register_benches(compute_provider_onednn
    hyle_compute_onednn_provider_bench
    hyle_compute_onednn_provider_bench_placement)
endif()
