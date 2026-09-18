# A single-file, build-local correction to the locked oneDNN runtime. Preserve
# its Apache-2.0 header and source cache; no algorithm or primitive is forked.
function(hyle_onednn_runtime_overlay source_dir binary_dir)
  set(_original "${source_dir}/src/common/primitive_iface.cpp")
  file(SHA256 "${_original}" _sha)
  if(NOT _sha STREQUAL
      "5ab7581b718c109b010c67981c90fae327a91c59a7fbef194bfb7a53692a1707")
    message(FATAL_ERROR "oneDNN primitive runtime identity changed; re-audit C1-AA")
  endif()
  file(READ "${_original}" _content)
  set(_begin "    stream->before_exec_hook();")
  set(_end "    stream->after_exec_hook();")
  set(_guard [=[    // HYLE C1-AA: preserve threadpool state on nested calls and exceptions.
    struct execution_scope_t {
        stream_t *stream;
        dnnl::threadpool_interop::threadpool_iface *previous;
        ~execution_scope_t() {
            stream->after_exec_hook();
            threadpool_utils::activate_threadpool(previous);
        }
    } execution_scope {stream, threadpool_utils::get_active_threadpool()};
    threadpool_utils::deactivate_threadpool();
    stream->before_exec_hook();]=])
  string(REPLACE "#include <string>"
    "#include <string>\n#include \"common/dnnl_thread.hpp\"" _content "${_content}")
  # Remove the original tail first, so the guard retains its own hook call.
  string(REPLACE "${_end}" "" _content "${_content}")
  string(REPLACE "${_begin}" "${_guard}" _content "${_content}")
  set(_overlay "${binary_dir}/hyle-runtime/primitive_iface.cpp")
  file(CONFIGURE OUTPUT "${_overlay}" CONTENT "${_content}" @ONLY)
  get_target_property(_sources dnnl_common SOURCES)
  list(FIND _sources "${_original}" _index)
  if(_index LESS 0)
    message(FATAL_ERROR "oneDNN common target no longer owns primitive_iface.cpp")
  endif()
  list(REMOVE_ITEM _sources "${_original}")
  list(APPEND _sources "${_overlay}")
  set_property(TARGET dnnl_common PROPERTY SOURCES "${_sources}")
  target_include_directories(dnnl_common PRIVATE "${source_dir}/src/common")
endfunction()
