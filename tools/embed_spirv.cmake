# Embed a compiled SPIR-V module as a C header (run via `cmake -P`, no Python dependency).
#
# Args (via -D):  SPV=<input.spv>  OUT=<output.h>  ARRAY=<array-name>  SRC=<original.comp>
#
# Emits `static const unsigned <ARRAY>[] = { 0x..., };` -- the shape tests/compute/*_spirv.h
# use, so the Vulkan backend's length-prefixed load_ptx path consumes it unchanged. Invoked
# by hyle_embed_shader() as the second step after glslang compiles the .comp -> .spv, so the
# checked-in-style headers are REPRODUCIBLE from source (no hand-pasted SPIR-V words).
#
# SPIR-V is a little-endian uint32 stream. file(READ HEX) gives a lowercase hex string of the
# raw bytes; we regroup each 4 bytes (8 hex chars) LE -> one 0x-word so the `unsigned[]`
# initializer round-trips byte-for-byte (the backend reads uint32 words).

if(NOT DEFINED SPV OR NOT DEFINED OUT OR NOT DEFINED ARRAY)
  message(FATAL_ERROR "embed_spirv.cmake: need -DSPV= -DOUT= -DARRAY= (optional -DSRC=)")
endif()

file(READ "${SPV}" hex HEX)              # whole module as a flat lowercase hex string
string(LENGTH "${hex}" nchars)
math(EXPR nbytes "${nchars} / 2")
math(EXPR rem "${nbytes} % 4")
if(NOT rem EQUAL 0)
  message(FATAL_ERROR "embed_spirv.cmake: ${SPV} is ${nbytes} bytes, not a multiple of 4 (not SPIR-V?)")
endif()

set(body "")
set(col 0)
set(pos 0)
# Walk the hex string 8 chars (=4 bytes=1 word) at a time; swap byte order LE -> value.
while(pos LESS nchars)
  string(SUBSTRING "${hex}" ${pos} 8 w)           # b0 b1 b2 b3 (little-endian byte order)
  string(SUBSTRING "${w}" 0 2 b0)
  string(SUBSTRING "${w}" 2 2 b1)
  string(SUBSTRING "${w}" 4 2 b2)
  string(SUBSTRING "${w}" 6 2 b3)
  set(word "0x${b3}${b2}${b1}${b0}")              # reassemble MSB..LSB
  if(col EQUAL 0)
    string(APPEND body "  ${word},")
  else()
    string(APPEND body " ${word},")
  endif()
  math(EXPR col "(${col} + 1) % 8")
  if(col EQUAL 0)
    string(APPEND body "\n")
  endif()
  math(EXPR pos "${pos} + 8")
endwhile()

set(header "// Auto-generated from ${SRC} by tools/embed_spirv.cmake (glslang -V). DO NOT EDIT.\n")
string(APPEND header "// Regenerate via the hyle_embed_shader() CMake target, not by hand.\n")
string(APPEND header "static const unsigned ${ARRAY}[] = {\n")
string(APPEND header "${body}")
if(NOT col EQUAL 0)
  string(APPEND header "\n")
endif()
string(APPEND header "};\n")

file(WRITE "${OUT}" "${header}")
