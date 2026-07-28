include_guard(GLOBAL)

# hyle_embed_shader(<comp> <out-header> <name> [<namespace>])
# Compile a GLSL compute shader (tools/shaders/<comp>) to SPIR-V with the fetched glslang
# binary, then embed it as a C header (`inline constexpr SpirvBlob <name>`) via
# tools/hyle_shader.py -- the SAME tool a HYLE user runs. Makes the SPIR-V headers
# REPRODUCIBLE from their .comp source instead of hand-pasted words. The output header lands in
# ${HYLE_BINARY_DIR}/generated/shaders; add that dir to a consumer target's include path.
# Requires HYLE_COMPUTE_VULKAN_ENABLED (the glslang-standalone target exists only then) +
# Python3. Sets the header's path in the parent-scope var HYLE_SHADER_<name>; the blob symbol
# is <namespace>::<name> (namespace optional, keeps multiple shaders from clashing).
function(hyle_embed_shader comp out_header name)
  set(ns "${ARGV3}")  # optional 4th arg
  set(src "${HYLE_SOURCE_DIR}/tools/shaders/${comp}")
  set(tool "${HYLE_SOURCE_DIR}/tools/hyle_shader.py")
  set(gen_dir "${HYLE_BINARY_DIR}/generated/shaders")
  set(spv "${gen_dir}/${name}.spv")
  set(hdr "${gen_dir}/${out_header}")
  file(MAKE_DIRECTORY "${gen_dir}")
  set(ns_arg "")
  if(ns)
    set(ns_arg --namespace ${ns})
  endif()
  # Declare the generating custom_command ONCE per output header. The gen dir is shared across
  # subdirectories (tests + bench both embed saxpy.spv.h into the same path); a second
  # add_custom_command for the same OUTPUT is a ninja "multiple rules generate" error. Track
  # emitted headers in a GLOBAL property and skip the duplicate rule (the file + its vars are
  # the same either way; consumers just add the include dir + list the header as a source).
  get_property(_emitted GLOBAL PROPERTY HYLE_EMBEDDED_SHADERS)
  if(NOT "${hdr}" IN_LIST _emitted)
    # One step: compile GLSL -> SPIR-V (via the fetched glslang) THEN embed -> header. Passing
    # --glslang $<TARGET_FILE:...> points the tool at our build's compiler (users pass their own
    # or rely on PATH). The tool is pure Python for embed; compile just shells out to glslang.
    add_custom_command(
      OUTPUT "${hdr}"
      COMMAND ${Python3_EXECUTABLE} "${tool}" compile "${src}" -o "${spv}"
              --glslang $<TARGET_FILE:glslang-standalone>
              --embed "${hdr}" --name ${name} ${ns_arg}
      DEPENDS "${src}" "${tool}" glslang-standalone
      COMMENT "Embedding SPIR-V ${comp} -> ${out_header} (${name})"
      VERBATIM)
    set_property(GLOBAL APPEND PROPERTY HYLE_EMBEDDED_SHADERS "${hdr}")
  endif()
  set(HYLE_SHADER_${name} "${hdr}" PARENT_SCOPE)
  set(HYLE_SHADER_DIR "${gen_dir}" PARENT_SCOPE)
endfunction()
