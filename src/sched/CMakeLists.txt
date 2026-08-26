# sched-01/02: fiber context switching on vendored fcontext asm + run-queue.
# Pick the asm flavor for this (platform, arch). Vendored asm covers x86_64 (MS/PE MASM,
# SysV/ELF GAS, SysV/macho GAS) AND arm64 (AAPCS ELF/macho GAS). The top-level gate
# (HYLE_HAS_FCONTEXT) guarantees we only get here on a target we have asm for.
set(_fctx "${PROJECT_SOURCE_DIR}/third_party/fcontext/asm")

if(MSVC)
  # Windows x64: MASM (ml64). PROCs need BOOST_CONTEXT_EXPORT to be PUBLIC.
  set(_fctx_asm
    "${_fctx}/jump_x86_64_ms_pe_masm.asm"
    "${_fctx}/make_x86_64_ms_pe_masm.asm"
    "${_fctx}/ontop_x86_64_ms_pe_masm.asm")
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  # arm64 AAPCS GAS (.S). macho on Apple Silicon, ELF on arm64 Linux. Symbols exported via
  # .globl in-file -- no define needed (same as the x86 GAS flavors).
  if(APPLE)
    set(_fctx_asm
      "${_fctx}/jump_arm64_aapcs_macho_gas.S"
      "${_fctx}/make_arm64_aapcs_macho_gas.S"
      "${_fctx}/ontop_arm64_aapcs_macho_gas.S")
  else()
    set(_fctx_asm
      "${_fctx}/jump_arm64_aapcs_elf_gas.S"
      "${_fctx}/make_arm64_aapcs_elf_gas.S"
      "${_fctx}/ontop_arm64_aapcs_elf_gas.S")
  endif()
elseif(APPLE)
  # macOS x64: SysV/macho GAS (Mach-O object format differs from ELF -- different .S).
  set(_fctx_asm
    "${_fctx}/jump_x86_64_sysv_macho_gas.S"
    "${_fctx}/make_x86_64_sysv_macho_gas.S"
    "${_fctx}/ontop_x86_64_sysv_macho_gas.S")
else()
  # Linux x64: SysV/ELF GAS (.S via the C/ASM driver). Symbols are exported with
  # .globl directly -- no define needed.
  set(_fctx_asm
    "${_fctx}/jump_x86_64_sysv_elf_gas.S"
    "${_fctx}/make_x86_64_sysv_elf_gas.S"
    "${_fctx}/ontop_x86_64_sysv_elf_gas.S")
endif()

hyle_add_module(sched TRACE
  SOURCES
    fiber.cc
    worker.cc
    scheduler.cc
    sync.cc
    join.cc
    blocking.cc
    cancel.cc
    ${_fctx_asm})

if(MSVC)
  # The MASM files tag their PROCs with BOOST_CONTEXT_EXPORT (undefined in-file).
  # Define it to EXPORT so jump/make/ontop_fcontext become PUBLIC and link from
  # fiber.cc. Visual Studio's MASM integration consumes the target's
  # PreprocessorDefinitions property, while generic COMPILE_OPTIONS (including
  # a language generator expression) are not emitted into its MASM item. Keep
  # this target-private; the same unused definition on the C++ TUs is harmless.
  target_compile_definitions(hyle_sched PRIVATE BOOST_CONTEXT_EXPORT=EXPORT)
endif()

hyle_register_package_dependency(
  NAME fcontext
  OWNER sched
  LOCKED_DEPENDENCIES fcontext
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/fcontext/LICENSE_1_0.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|fcontext-LICENSE_1_0.txt")
