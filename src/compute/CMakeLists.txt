# compute-01: device/stream/event abstraction + completion (event-poll) primitive,
# with a manual-vtable backend split (backend.h). The host fallback backend is
# ALWAYS built (any platform, no GPU needed). The CUDA backend is added only when
# the CUDAToolkit was found (HYLE_COMPUTE_CUDA set by the root CMakeLists) -- then
# cuda_backend.cu is compiled by nvcc as a CUDA-language TU and CUDA::cudart is linked.
#
# compute depends on error (Status/Result) + mem (pinned Allocator) PUBLICLY. As of
# compute-03 it also depends on sched (gpu_await.h's co_await uses sched Waker/suspend)
# -- but ONLY where fcontext (hence sched) exists, exactly as io_context.cc did. The
# host/device/buffer/pool layer (compute-01/02/02b/02c) needs no sched and builds
# anywhere; gpu_poller.cc (compute-03) is added only under HYLE_HAS_FCONTEXT.
set(_compute_sources compute.cc host_backend.cc)
set(_compute_public_deps hyle_error hyle_mem)

if(HYLE_COMPUTE_CUDA)
  # cuda_backend.cu is a CUDA-language TU (compiled by nvcc, host side by the same
  # MSVC/GCC as everything else via -ccbin). CMake's first-class CUDA support
  # (enable_language(CUDA) in the root) drives this from the .cu extension.
  list(APPEND _compute_sources cuda_backend.cu)
endif()

if(HYLE_HAS_FCONTEXT)
  # compute-03 GPU completion -> fiber: gpu_poller.cc uses sched Waker/suspend to wake
  # a fiber awaiting a GPU event. Only buildable where sched exists (same gate as io's
  # io_context.cc). Adds sched to compute's PUBLIC deps (gpu_await.h includes
  # sched/waker.h, so a consumer of that header needs sched).
  list(APPEND _compute_sources gpu_poller.cc)
  list(APPEND _compute_public_deps hyle_sched)
endif()

hyle_add_module(compute SOURCES ${_compute_sources} PUBLIC_DEPS ${_compute_public_deps})

if(HYLE_COMPUTE_CUDA)
  # Link the CUDA runtime (cudaMalloc/Stream/Event/EventQuery live here). CUDA::cudart
  # is the shared runtime target find_package(CUDAToolkit) provides; PRIVATE because
  # the CUDA types never appear in compute's public headers (backend.h is internal).
  # CUDA::cuda_driver: the VMM device pool (compute-02c) uses the driver API
  # (cuMemAddressReserve/cuMemCreate/cuMemMap) -- link libcuda (WSL /usr/lib/wsl/lib,
  # Win cuda.lib). Runtime API alone (cudart) doesn't provide the cu* VMM calls.
  # CUDA::nvrtc: compute-04 kernel compile (nvrtcCompileProgram source -> PTX).
  target_link_libraries(hyle_compute PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  # The .cu TU needs to see HYLE_COMPUTE_CUDA + HYLE_COMPUTE_BUILD (the export macro
  # is set PRIVATE by hyle_add_module for the shared build; a .cu is part of the same
  # target so it inherits it, but the CUDA-language define set needs HYLE_COMPUTE_CUDA
  # too so backend.h's #if compiles cuda_vtable()).
  target_compile_definitions(hyle_compute PRIVATE HYLE_COMPUTE_CUDA)
  # CUDA dialect = C++20, NOT 23. hyle_add_module sets cxx_std_23 as a PUBLIC compile
  # feature (for the C++ TUs); CMake would otherwise map that onto a "CUDA23" dialect
  # requirement, which CMake's flag table doesn't know for nvcc 13.3 -> configure
  # error. cuda_backend.cu is plain runtime-API + manual-vtable fill; it needs no
  # C++23 feature, so pin the CUDA TU to CUDA20 (nvcc supports c++20 cleanly). The
  # host-side C++ TUs still compile at C++23.
  set_target_properties(hyle_compute PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native)
  # Target the build host's GPU arch (sm_89 / RTX 4070 here) via CUDA_ARCHITECTURES
  # native (CMake >= 3.24): the self-compiled / known-hardware path (coding-conventions
  # §10 build-time tier). A distributable multi-arch fat-binary + driver JIT is a later
  # concern (compute packaging), not compute-01.
endif()
