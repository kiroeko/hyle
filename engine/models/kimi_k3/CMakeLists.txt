# Model-private component; the default Qwen application does not link it.
if(NOT "cuda" IN_LIST _hyle_engine_routes)
  return()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/component_sources.cmake")
_hyle_add_kimi_k3_component(hyle::graph hyle::compute hyle::graph_progress hyle::mem)
