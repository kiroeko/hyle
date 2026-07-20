cmake_minimum_required(VERSION 3.25)

project(hyle
  VERSION 0.1.0
  DESCRIPTION "HYbrid LEvels - general-purpose high-performance C++ infra"
  LANGUAGES C CXX ASM)

# ---- C++23 baseline + toolchain gate ----------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C baseline. HYLE's own C-ABI surface (extern "C" vtables) is compiled as C++;
# this only governs real .c/.S translation units (vendored C deps like liburing/
# XXH3, fcontext asm shims) so they don't fall back to compiler-default C.
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Toolchain gate (charter: GCC14+ / Clang18+ / MSVC19.43+ for C++23 std::expected etc.)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "HYLE requires GCC 14+ (got ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "HYLE requires Clang 18+ (got ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.43)
  message(FATAL_ERROR "HYLE requires MSVC 19.43+ / VS 17.13+ (got ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# ---- global options ---------------------------------------------------------
option(HYLE_BUILD_TESTS "build unit tests (GoogleTest)" ON)
option(HYLE_BUILD_BENCH "build microbenchmarks (nanobench)" OFF)
set(HYLE_SIMD_EXACT_TARGET "" CACHE STRING
    "Build hyle_simd for one required ISA; empty enables runtime dispatch")
set_property(CACHE HYLE_SIMD_EXACT_TARGET PROPERTY STRINGS
             "" SCALAR SSE4 AVX2 AVX512 NEON SVE RVV)
option(HYLE_STATIC      "build static lib (default SHARED)" OFF)
option(HYLE_WERROR      "treat warnings as errors (CI on, local off)" OFF)
set(HYLE_SANITIZER "off" CACHE STRING "off|asan|tsan|ubsan")
set_property(CACHE HYLE_SANITIZER PROPERTY STRINGS off asan tsan ubsan)
# VALIDATE + NORMALIZE the value. set_property(STRINGS) only fills the GUI dropdown; it does
# NOT reject a bad -D value, so `-DHYLE_SANITIZER=thread` (a natural mistake -- it's the
# compiler's own -fsanitize=THREAD name) used to fall through every STREQUAL and silently
# build with NO instrumentation -> a green test run that proved nothing (a FALSE pass). Accept
# the compiler-native aliases (address/thread/undefined/none) AND hard-FAIL on anything else,
# so a typo can never again degrade to "off" unnoticed.
string(TOLOWER "${HYLE_SANITIZER}" _san_lc)
if(_san_lc STREQUAL "address")
  set(_san_lc "asan")
elseif(_san_lc STREQUAL "thread")
  set(_san_lc "tsan")
elseif(_san_lc STREQUAL "undefined")
  set(_san_lc "ubsan")
elseif(_san_lc STREQUAL "none" OR _san_lc STREQUAL "")
  set(_san_lc "off")
endif()
if(NOT _san_lc MATCHES "^(off|asan|tsan|ubsan)$")
  message(FATAL_ERROR
    "HYLE_SANITIZER='${HYLE_SANITIZER}' is not valid. Use one of: off | asan | tsan | ubsan "
    "(aliases address/thread/undefined/none also accepted). Refusing to silently build "
    "WITHOUT instrumentation -- that would make a sanitizer run a false pass.")
endif()
set(HYLE_SANITIZER "${_san_lc}" CACHE STRING "off|asan|tsan|ubsan" FORCE)
message(STATUS "HYLE_SANITIZER = ${HYLE_SANITIZER}")

# Build configurations. Multi-config generators (VS, Xcode, Ninja Multi-Config)
# pick per-build via --config; single-config (plain Ninja, Makefiles) via
# CMAKE_BUILD_TYPE. RelWithDebInfo kept: a perf lib needs an optimized build that
# still profiles/symbolicates. MinSizeRel dropped (irrelevant for infra).
get_property(_hyle_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_hyle_multi_config)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)  # perf-first default
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

if(HYLE_STATIC)
  set(BUILD_SHARED_LIBS OFF)
else()
  set(BUILD_SHARED_LIBS ON)
endif()

# Unify runtime artifacts (.exe + .dll on Windows) into one dir so test/bench
# exes find the shared libs they link against without PATH juggling.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# Static archives (.a/.lib) land in the SAME bin/ as shared libs, so a prebuilt-library
# consumer (samples/04) finds the artifact in one predictable place regardless of
# HYLE_STATIC. Without this, static .a defaults to per-target build subdirs.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HyleWarnings)
include(HyleSanitizers)
include(HyleModule)
include(HyleTrace)    # obs-02: HYLE_TRACE gate + optional tracy backend (default OFF)
include(HyleTesting)  # HYLE_TESTING gate: test-only interfaces (derived from HYLE_BUILD_TESTS)

# ---- optional OpenSSL (io-08 TLS) -------------------------------------------
# TLS is an OPTIONAL capability: if OpenSSL is found the io module builds tls.cc
# and links it; if not, TLS simply isn't compiled and a user who doesn't want TLS
# carries ZERO OpenSSL dependency (same "opt-in capability = opt-in dependency"
# rule as HYLE_TRACE/tracy). DYNAMIC link on purpose -- a security lib should
# track the system's CVE updates, not be baked statically into HYLE's artifact.
# Linux: found zero-config via the system libssl-dev. Windows: pass OPENSSL_ROOT_DIR
# or use a vcpkg toolchain. Install/runtime-DLL distribution is a close-phase
# concern (see docs/milestones/close-01), not any single io milestone.
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  message(STATUS "HYLE: OpenSSL ${OPENSSL_VERSION} found -- io-08 TLS enabled")
else()
  message(STATUS "HYLE: OpenSSL not found -- io-08 TLS disabled (set OPENSSL_ROOT_DIR to enable)")
endif()

# ---- HTTP/2 protocol core (io-11) ------------------------------------------
# nghttp2 owns RFC 9113 framing, HPACK, stream state and flow control. HYLE owns
# the coroutine transport over TcpStream/TlsStream and never imports nghttp2's
# event loop, threads, applications or examples. Keep the protocol dependency
# pinned: HTTP/2 parsers are security-sensitive and an uncontrolled system
# version would make identical HYLE builds behave differently.
option(HYLE_IO_HTTP2 "io-11: HTTP/2 via nghttp2 protocol core" ON)
if(HYLE_IO_HTTP2)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps"
      CACHE PATH "Persistent FetchContent cache (survives build/ wipes; per-machine tree)")
  include(FetchContent)
  set(NGHTTP2_PIN_COMMIT "68cb6900fde14c77f0cd7add0e094a862960eb99") # v1.69.0
  set(ENABLE_LIB_ONLY ON CACHE BOOL "" FORCE)
  set(ENABLE_APP OFF CACHE BOOL "" FORCE)
  set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(ENABLE_HPACK_TOOLS OFF CACHE BOOL "" FORCE)
  set(ENABLE_DOC OFF CACHE BOOL "" FORCE)
  set(ENABLE_HTTP3 OFF CACHE BOOL "" FORCE)
  # nghttp2's generic BUILD_STATIC_LIBS knob must not leak into an embedding
  # application's cache. Give only the dependency subdirectory a normal-variable
  # override, then restore the caller's value after FetchContent returns.
  set(_hyle_had_build_static_libs FALSE)
  if(DEFINED BUILD_STATIC_LIBS)
    set(_hyle_had_build_static_libs TRUE)
    set(_hyle_saved_build_static_libs "${BUILD_STATIC_LIBS}")
  endif()
  set(BUILD_STATIC_LIBS ${HYLE_STATIC})
  FetchContent_Declare(nghttp2
    GIT_REPOSITORY https://github.com/nghttp2/nghttp2.git
    GIT_TAG        ${NGHTTP2_PIN_COMMIT}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(nghttp2)
  if(_hyle_had_build_static_libs)
    set(BUILD_STATIC_LIBS "${_hyle_saved_build_static_libs}")
  else()
    unset(BUILD_STATIC_LIBS)
  endif()
  unset(_hyle_had_build_static_libs)
  unset(_hyle_saved_build_static_libs)
  message(STATUS
    "HYLE: nghttp2 v1.69.0 (${NGHTTP2_PIN_COMMIT}) -- io-11 HTTP/2 enabled")
endif()

# ---- optional CUDA (compute-01 CUDA backend) --------------------------------
# The compute pillar's CUDA backend is an OPTIONAL capability, same "opt-in
# capability = opt-in dependency" rule as OpenSSL/tracy: if the CUDA Toolkit is
# found we enable the CUDA language and compile cuda_backend.cu (nvcc), so a real
# GPU can be driven; if not, compute still builds with ONLY the host fallback
# backend and carries zero CUDA dependency. HYLE_COMPUTE_CUDA gates the .cu source +
# the cuda_vtable() reference (backend.h) so a CUDA-free build is fully inert.
#
# CMake's first-class CUDA support: find_package finds the toolkit; enable_language
# turns .cu files into nvcc TUs (host side via -ccbin = our C++ compiler). We do NOT
# make CUDA a hard project language (LANGUAGES line stays C CXX ASM) so a machine
# with no CUDA (mac, CI without a toolkit) configures fine.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(HYLE_COMPUTE_CUDA ON)
  message(STATUS "HYLE: CUDA Toolkit ${CUDAToolkit_VERSION} found -- compute CUDA backend enabled")
else()
  set(HYLE_COMPUTE_CUDA OFF)
  message(STATUS "HYLE: CUDA Toolkit not found -- compute uses host fallback only")
endif()

# ---- optional ggml operator library (compute-05 ggml_bridge) -----------------
# ggml is BORROWED as an operator-kernel supplier (compute-05: matmul/softmax/... on
# HYLE-managed VRAM + completion + scheduling). Same "opt-in capability = opt-in
# dependency" rule as CUDA/OpenSSL: HYLE_COMPUTE_GGML gates ggml_bridge.{h,cc}; without
# it compute still builds (01-04) with zero ggml dependency.
#
# INTEGRATION = FetchContent PINNED BY COMMIT (NOT find_package(ggml)): a system-installed
# ggml has UNCONTROLLED build options (OpenMP on/off, backends, version), but the
# n_threads=1 leaf contract (compute-design 发现 10) REQUIRES OpenMP OFF and zero-copy
# REQUIRES a pinned version. So we own the build via the knobs below, pinned to a commit.
#
# ZERO REPEATED DOWNLOADS = PERSISTENT FETCH CACHE (cil2cpp runtime/CMakeLists.txt form):
# FETCHCONTENT_BASE_DIR pinned to an in-source .deps (.gitignored) so a FIRST `cmake` on a
# fresh clone auto-fetches ggml into .deps/ggml-src ONCE, and every later `rm -rf build` /
# reconfigure / second build dir on this machine REUSES it -- no re-download. This is the
# "clone the repo, run cmake, it just works (online)" contract; there is NO pre-placed
# source mirror (we tried SOURCE_DIR into third_party/ggml but it needs a manual per-
# machine archive step -- against the open-box-auto-download intent, so removed).
# .deps is per-machine (Win C:\..., WSL /root/hyle native ext4 NEVER /mnt/c, mac ~/hyle).
option(HYLE_COMPUTE_GGML "compute-05: borrow ggml operators via ggml_bridge" ON)
if(HYLE_COMPUTE_GGML)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps"
      CACHE PATH "Persistent FetchContent cache (survives build/ wipes; per-machine tree)")
  include(FetchContent)
  set(GGML_PIN_COMMIT "eced84c86f8b012c752c016f7fe789adea168e1e")  # v0.15.3

  # Own ggml's build knobs BEFORE it configures (cache vars it reads at option()).
  set(GGML_OPENMP        OFF CACHE BOOL "" FORCE)  # 发现 10: n_threads=1 leaf; no OMP threads
  set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_CPU           ON  CACHE BOOL "" FORCE)  # CPU backend (n_threads=1 leaf path)
  # ggml follows HYLE's BUILD_SHARED_LIBS (HYLE builds shared by default, line ~76): a
  # FORCE OFF here is shadowed by that normal var anyway, and a lone-static ggml among
  # shared HYLE libs invites ODR/symbol skew. So ggml is shared like the rest (runtime
  # needs libggml.so on the path, handled by install/rpath in close-01).
  # GGML_CUDA_FA OFF: skip ~120 FlashAttention kernel instances (attention, not matmul).
  # The bridge borrows GEMM/elementwise; FA is unused and dominates first-build time.
  set(GGML_CUDA_FA       OFF CACHE BOOL "" FORCE)
  if(HYLE_COMPUTE_CUDA)
    set(GGML_CUDA ON  CACHE BOOL "" FORCE)  # CUDA backend (zero-copy device_ptr path)
  else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  endif()

  message(STATUS "HYLE: ggml via FetchContent pin ${GGML_PIN_COMMIT} (cache: .deps, GGML_CUDA=${HYLE_COMPUTE_CUDA})")
  FetchContent_Declare(ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        ${GGML_PIN_COMMIT}
    GIT_SHALLOW    TRUE)
  # EXCLUDE_FROM_ALL: only the parts hyle_compute links get built (no ggml install/tests).
  FetchContent_MakeAvailable(ggml)
endif()


# ---- optional Vulkan compute (compute-06 self-authored SPIR-V kernels) --------
# compute-06 gives HYLE its OWN cross-vendor GPU compute path (SPIR-V), the Vulkan
# twin of compute-04's CUDA NVRTC "any kernel". Same "opt-in capability = opt-in
# dependency" rule as CUDA/ggml: HYLE_COMPUTE_VULKAN gates the Vulkan backend +
# vulkan_bridge; without it compute still builds (01-05) with zero Vulkan dependency.
#
# WHY volk (NOT find_package(Vulkan)): the Vulkan LOADER is the one piece we can't just
# FetchContent -- but linking it the classic way needs vulkan-1.lib, which only ships
# with an installed SDK (this machine has the runtime vulkan-1.dll from the driver but
# NO .lib -- find_package(Vulkan) fails). volk (zeux/volk, the industry-standard Vulkan
# meta-loader -- NOT a glad-style code generator, a fixed 2-file lib) sidesteps this: it
# dlopen's the runtime loader at startup (volkInitialize -> vkGetInstanceProcAddr) and
# resolves all vk* pointers itself, so NO import lib / NO SDK install is needed -- only
# the driver's runtime loader (always present). Bonus: device-level function resolution
# skips the loader trampoline (one less indirection per call = faster, matches HYLE's
# perf goal). kompute uses the same dynamic-dispatch idea (VULKAN_HPP_DEFAULT_DISPATCHER).
#
# THREE FetchContent DEPS, ALL PINNED TO vulkan-sdk-1.4.341.0 (matches the local runtime
# loader 1.4.341 -- zero version skew), all Khronos/zeux source repos (NOT the LunarG SDK
# packaging layer), landing in .deps (inherits FETCHCONTENT_BASE_DIR -> survives build
# wipes, per-machine, open-box auto-download -- same as ggml/googletest):
#   1. Vulkan-Headers (Khronos)  -- pure headers, version OWNED by HYLE not the machine.
#   2. volk (zeux)               -- meta-loader; runtime dlopen, no .lib/SDK needed.
#   3. glslang (Khronos)         -- GLSL->SPIR-V: ENABLE_OPT=OFF (no SPIRV-Tools dep),
#      tests/install/HLSL off, ENABLE_GLSLANG_BINARIES=ON to also build the `glslang`
#      validator tool (the "pre-compile tool folder" for HYLE devs + clients).
# The slate-bundled Vulkan (headers v270, stale) is NOT used.
option(HYLE_COMPUTE_VULKAN "compute-06: self-authored Vulkan compute SPIR-V kernels" ON)
if(HYLE_COMPUTE_VULKAN)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps"
      CACHE PATH "Persistent FetchContent cache (survives build/ wipes; per-machine tree)")
  include(FetchContent)
  # All pinned to vulkan-sdk-1.4.341.0 (== local loader 1.4.341, no skew).
  set(VULKAN_HEADERS_PIN "b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2")  # Vulkan-Headers 1.4.341.0
  set(VOLK_PIN           "d979819fd03de3a7606d6e23d6ff4968942599da")  # volk 1.4.341.0
  set(GLSLANG_PIN        "f0bd0257c308b9a26562c1a30c4748a0219cc951")  # glslang 1.4.341.0

  # 1. Vulkan-Headers first -- creates the Vulkan::Headers target volk/glslang can see.
  FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        ${VULKAN_HEADERS_PIN}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(vulkan_headers)

  # 2. volk -- point it at the fetched headers (VOLK_PULL_IN_VULKAN tries this var first).
  set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
  set(VOLK_PULL_IN_VULKAN ON  CACHE BOOL "" FORCE)
  set(VOLK_INSTALL        OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        ${VOLK_PIN}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(volk)

  # 3. glslang build knobs OWNED before configure (cache vars it reads at option()).
  set(ENABLE_OPT              OFF CACHE BOOL "" FORCE)  # no spirv-opt -> no SPIRV-Tools dep
  set(ENABLE_HLSL             OFF CACHE BOOL "" FORCE)  # HYLE writes GLSL .comp, not HLSL
  set(GLSLANG_TESTS           OFF CACHE BOOL "" FORCE)
  set(GLSLANG_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)
  set(BUILD_EXTERNAL          OFF CACHE BOOL "" FORCE)  # no /External (we pin deps ourselves)
  set(ENABLE_SPIRV            ON  CACHE BOOL "" FORCE)  # GLSL -> SPIR-V (the whole point)
  set(ENABLE_GLSLANG_BINARIES ON  CACHE BOOL "" FORCE)  # build the `glslang` validator tool
  FetchContent_Declare(glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG        ${GLSLANG_PIN}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(glslang)

  # Python3 drives tools/hyle_shader.py (GLSL->SPIR-V->embedded header) via hyle_embed_shader.
  # Interpreter-only (no dev headers); used at build time to regenerate the test shader headers.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(HYLE_COMPUTE_VULKAN_ENABLED ON)
  message(STATUS "HYLE: Vulkan compute enabled -- volk + Vulkan-Headers + glslang via "
                 "FetchContent pin 1.4.341 (cache: .deps; loader dlopen'd at runtime, no SDK)")
endif()


# ---- global correctness flags (apply to ALL targets incl. fetched deps) ------
# These change language semantics, not just diagnostics, so they must be uniform
# across the whole build (gtest included) to avoid ODR/ABI skew. Scoped to C/CXX
# via a generator expression so they are NOT passed to the assembler (ml64 would
# choke on /Zc:__cplusplus etc. when we compile fcontext .asm).
if(MSVC)
  # /Zc:__cplusplus  : report the true __cplusplus value (else MSVC pins 199711L
  #                    and every C++23 feature-test macro misfires).
  # /utf-8           : treat source AND execution charset as UTF-8 (kills C4819).
  # One quoted genex PER flag: a ';' inside $<...> is list-split by
  # add_compile_options before evaluation, which would mangle the flags.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
  # fcontext fiber switching needs an assembler. On MSVC x64 that is MASM (ml64).
  enable_language(ASM_MASM)
endif()

# ---- x86-64-v2 baseline: SSSE3 everywhere (GCC/Clang) ------------------------
# SwissTable's GroupSse2::match_empty (probe.h, a HEADER-inline member) has a 1-insn
# SSSE3 fast path gated on HYLE_HAVE_SSSE3. MSVC exposes SSSE3 intrinsics on x64
# unconditionally; GCC/Clang gate __SSSE3__ behind -mssse3. This MUST be GLOBAL (not
# per-module): probe.h is inlined into container TUs AND test/bench TUs (via registry.h
# etc.), so a per-module flag would compile match_empty two different ways in one binary
# = ODR violation. Setting it project-wide makes every TU agree. SSSE3 is the x86-64-v2
# baseline (universal since ~2009); a distributable binary requiring it is standard. This
# does NOT touch the wide AVX2/AVX512 SIGILL contract -- those stay confined to the
# per-ISA kernel TUs in simd/ (which add their own -mavx2 on top).
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mssse3" HYLE_CXX_HAS_MSSSE3)
  if(HYLE_CXX_HAS_MSSSE3)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-mssse3>")
  endif()
endif()

# ---- modules (added per dependency layer, roadmap P-1..P6) ------------------
add_subdirectory(src/core)   # 00-scaffolding: empty core + macro skeleton
add_subdirectory(src/error)  # error-01: Status + Result + propagation macros
add_subdirectory(src/obs)    # obs-01: structured log production (sink registry)
add_subdirectory(src/platform)  # platform-01: thread + park (wait_on_address)
add_subdirectory(src/mem)    # mem-01: Allocator vtable + alignment fix
add_subdirectory(src/time)   # time-01: monotonic + realtime clock + sleep
add_subdirectory(src/util)   # util-01: ref_counted (Ref<T>) + scope_guard (DEFER)
add_subdirectory(src/math)   # math: scalar helpers (div_ceil/sgn/approx_eq)
add_subdirectory(src/container)  # container-01: bounded MPMC ring (Vyukov)
add_subdirectory(src/simd)   # simd-01: runtime CPUID dispatch + vec<T,N> D-tag + scalar

# sched fiber switching needs platform fcontext asm. Vendored asm covers x86_64 (MS/PE MASM,
# SysV/ELF GAS, SysV/macho GAS) AND arm64 (AAPCS ELF/macho/PE). Wire each (platform, arch)
# to its flavor; the sched CMakeLists picks the matching .S/.asm. Gate the module on having a
# usable asm flavor for this target.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    set(HYLE_HAS_FCONTEXT TRUE)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
    set(HYLE_HAS_FCONTEXT TRUE)  # GAS .S via the ASM language (elf on Linux, macho on Apple)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(HYLE_HAS_FCONTEXT TRUE)  # AAPCS GAS .S (macho on Apple Silicon, elf on arm64 Linux)
  endif()
endif()
if(HYLE_HAS_FCONTEXT)
  add_subdirectory(src/sched)  # sched-01/02: fcontext fiber + run-queue/worker
endif()
add_subdirectory(src/io)     # io-01: proactor completion seam (IOCP; io_uring next)

# compute-01: device/stream/event abstraction + event-poll completion primitive.
# Host fallback backend always; CUDA backend when HYLE_COMPUTE_CUDA (toolkit found).
# Depends only on error for now (compute-03 will add sched for GPU->fiber wakeup).
add_subdirectory(src/compute)

# graph-01: task-dependency DAG with dirty-flag incremental recompute (north-star ②
# CAD/BIM/CAE parametric/simulation recompute). step-1 serial core depends on
# error/mem/container only (builds anywhere); step-2 parallel evaluate (parallel_eval.cc)
# is added inside src/graph/CMakeLists under HYLE_HAS_FCONTEXT (needs hyle_sched), exactly
# like compute's gpu_poller.cc.
add_subdirectory(src/graph)

if(HYLE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(HYLE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# F5 in a generated VS solution should launch the smoke test, not ALL_BUILD.
if(_hyle_multi_config AND HYLE_BUILD_TESTS AND TARGET hyle_core_test)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT hyle_core_test)
endif()

message(STATUS "HYLE ${PROJECT_VERSION} | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | C++${CMAKE_CXX_STANDARD} | sanitizer=${HYLE_SANITIZER} | shared=${BUILD_SHARED_LIBS}")
