# sched-01: fiber context switching on vendored fcontext asm.
# Only the Windows x64 MASM trio is wired here (this machine's toolchain). Other
# platforms' asm is vendored under third_party/fcontext/asm but not yet built --
# they get added (with ASM/GAS) when CI / those machines come online.
set(_fctx "${PROJECT_SOURCE_DIR}/third_party/fcontext/asm")

hyle_add_module(sched
  SOURCES
    fiber.cc
    "${_fctx}/jump_x86_64_ms_pe_masm.asm"
    "${_fctx}/make_x86_64_ms_pe_masm.asm"
    "${_fctx}/ontop_x86_64_ms_pe_masm.asm"
  PUBLIC_DEPS hyle_mem)

# The MASM files tag their PROCs with BOOST_CONTEXT_EXPORT (undefined in-file).
# Define it to EXPORT so jump/make/ontop_fcontext become PUBLIC and link from
# fiber.cc. Scoped to ASM_MASM so the C++ compiler never sees it.
target_compile_definitions(hyle_sched PRIVATE
  $<$<COMPILE_LANGUAGE:ASM_MASM>:BOOST_CONTEXT_EXPORT=EXPORT>)
