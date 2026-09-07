# Vendored: boost.context fcontext assembly

Hand-context-switch primitives (`jump_fcontext` / `make_fcontext` /
`ontop_fcontext`) taken from **boost.context** (author Oliver Kowalke),
distributed under the **Boost Software License 1.0** (see `LICENSE_1_0.txt`).

## What and why

HYLE's scheduler needs user-space stackful context switching. Per
`docs/architecture/sched-design.md` §1, fcontext is the chosen primitive: it
saves only callee-saved registers per the calling convention, touches no signal
mask, and makes no syscall (unlike ucontext). We take **only the three assembly
routines** -- NOT boost.fiber (which carries its own scheduler) and NOT
boost.context's C++ wrappers. The HYLE-side declarations live in
`src/sched/fcontext.h` (a ~10-line `extern "C"` shim), so there is zero
dependency on boost headers or build system.

## Source

- Upstream: https://github.com/boostorg/context  (`libs/context/src/asm/`)
- Release: `boost-1.92.0`
- Pinned commit: `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600`
- Files are copied **verbatim**, not modified.

## Contents (6 platform/arch x 3 routines = 18 files)

| platform / arch | assembler | files |
|---|---|---|
| Linux x86_64    | GAS  | `*_x86_64_sysv_elf_gas.S` |
| Linux arm64     | GAS  | `*_arm64_aapcs_elf_gas.S` |
| macOS x86_64    | GAS  | `*_x86_64_sysv_macho_gas.S` |
| macOS arm64     | GAS  | `*_arm64_aapcs_macho_gas.S` |
| Windows x86_64  | MASM | `*_x86_64_ms_pe_masm.asm` |
| Windows arm64   | armasm | `*_arm64_aapcs_pe_armasm.asm` |

## Build note

The MASM files reference `BOOST_CONTEXT_EXPORT` (undefined in-file). It is a PROC
visibility macro; we pass `/DBOOST_CONTEXT_EXPORT=EXPORT` to ml64 so the routines
are PUBLIC (linkable from `fiber.cc`). The product build selects exactly one trio
from the target OS/architecture. Current HYLE closure evidence executes the
Windows x86_64 MASM, Linux x86_64 ELF/GAS and macOS arm64 Mach-O/GAS variants;
the packaged Linux arm64, macOS x86_64 and Windows arm64 variants remain supported
source selections but require a matching validation machine before a round may
claim execution evidence for them.
