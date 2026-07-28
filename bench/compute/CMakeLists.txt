# compute bench (compute-06 #16): CPU submit cost, launch (re-record) vs replay (reuse a
# recorded command buffer). Vulkan-only (record/replay is a Vulkan optimization); gate on it.
# Reuses the SAME embedded saxpy SPIR-V the test uses, regenerated here via hyle_embed_shader.
if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_embed_shader(saxpy.comp saxpy.spv.h kSaxpySpirv hyle_test_shaders)
  add_executable(hyle_compute_bench compute_bench.cc "${HYLE_SHADER_kSaxpySpirv}")
  target_link_libraries(hyle_compute_bench PRIVATE hyle_compute nanobench)
  target_include_directories(hyle_compute_bench PRIVATE "${HYLE_SHADER_DIR}")
  hyle_apply_warnings(hyle_compute_bench)
  hyle_register_benches(compute hyle_compute_bench)
endif()
