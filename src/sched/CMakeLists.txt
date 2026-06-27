# sched-01/02: fiber context switching on vendored fcontext asm + run-queue.
# Pick the asm flavor for this (platform, arch). x86_64 only for now; arm64 and
# macho asm are vendored under third_party/fcontext/asm but not yet wired (no
# test machine). The top-level gate (HYLE_HAS_FCONTEXT) guarantees we only get
# here on a supported target.
set(_fctx "${PROJECT_SOURCE_DIR}/third_party/fcontext/asm")

if(MSVC)
  # Windows x64: MASM (ml64). PROCs need BOOST_CONTEXT_EXPORT to be PUBLIC.
  set(_fctx_asm
    "${_fctx}/jump_x86_64_ms_pe_masm.asm"
    "${_fctx}/make_x86_64_ms_pe_masm.asm"
    "${_fctx}/ontop_x86_64_ms_pe_masm.asm")
else()
  # Linux x64: SysV/ELF GAS (.S via the C/ASM driver). Symbols are exported with
  # .globl directly -- no define needed.
  set(_fctx_asm
    "${_fctx}/jump_x86_64_sysv_elf_gas.S"
    "${_fctx}/make_x86_64_sysv_elf_gas.S"
    "${_fctx}/ontop_x86_64_sysv_elf_gas.S")
endif()

hyle_add_module(sched
  SOURCES
    fiber.cc
    worker.cc
    ${_fctx_asm}
  PUBLIC_DEPS hyle_mem)

if(MSVC)
  # The MASM files tag their PROCs with BOOST_CONTEXT_EXPORT (undefined in-file).
  # Define it to EXPORT so jump/make/ontop_fcontext become PUBLIC and link from
  # fiber.cc. Scoped to ASM_MASM so the C++ compiler never sees it.
  target_compile_definitions(hyle_sched PRIVATE
    $<$<COMPILE_LANGUAGE:ASM_MASM>:BOOST_CONTEXT_EXPORT=EXPORT>)
endif()
