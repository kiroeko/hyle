"""One-shot, read-only inventory of the design baseline; not a validation driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASELINE = "17ff6555c7af0a8bd44d648209c40f53de758420"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries: list[dict] = []
    files: dict[str, bytes] = {}

    def add(group: str, name: str, path: str, anchor: str, audience: str,
            owner: str, consumer: str, effect: str, decision: str) -> None:
        if path not in files:
            files[path] = subprocess.check_output(
                ["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
        lines = files[path].decode("utf-8").splitlines()
        matches = [n for n, line in enumerate(lines, 1) if anchor in line]
        if not matches:
            raise ValueError(f"Missing baseline anchor {path}: {anchor}")
        entries.append(dict(group=group, name=name, audience=audience, owner=owner,
                            consumer=consumer, effect=effect, decision=decision,
                            source=dict(path=path, line=matches[0], anchor=anchor,
                                        sha256=hashlib.sha256(files[path]).hexdigest())))

    # Unique controls, not references, overloads, or internal derived *_ENABLED vars.
    for name, path, anchor, effect in (
        ("HYLE_COMPUTE_CPU", "cmake/HyleComponents.cmake", "option(HYLE_COMPUTE_CPU", "BOOL ON; CPU math closure, independent of GPU selections"),
        ("HYLE_COMPUTE_CUDA", "cmake/HyleCudaDependencies.cmake", "hyle_auto_feature_option(HYLE_COMPUTE_CUDA", "OFF default; AUTO may omit route; ON requires Toolkit"),
        ("HYLE_COMPUTE_APPLE", "cmake/HyleAppleDependencies.cmake", "hyle_auto_feature_option(HYLE_COMPUTE_APPLE", "OFF default; platform/Metal/MLX closure"),
        ("HYLE_COMPUTE_VULKAN", "cmake/HyleVulkanDependencies.cmake", "hyle_auto_feature_option(HYLE_COMPUTE_VULKAN", "OFF default; shader/native closure"),
        ("HYLE_COMPUTE_ONEDNN", "cmake/HyleComputeDependencies.cmake", "hyle_auto_feature_option(HYLE_COMPUTE_ONEDNN", "AUTO x86; OFF removes eligible CPU programs and recipe"),
        ("HYLE_COMPUTE_TORCH_CPU", "src/compute_provider_native/torch_cpu_sdk.cmake", "hyle_auto_feature_option(HYLE_COMPUTE_TORCH_CPU", "AUTO on supported platform; OFF removes original ATen/format programs"),
    ):
        add("capability_controls", name, path, anchor, "ordinary_current", "build_system", "source/package producer", effect, "REPLACE: one HYLE_COMPUTE_ROUTES set; mandatory platform closure")

    for name, anchor, effect, audience in (
        ("HYLE_COMPONENTS", 'set(HYLE_COMPONENTS "${_hyle_selection_input}" CACHE STRING', "public product selection; graph already depends on compute", "ordinary"),
        ("HYLE_DEV_FACTORS", 'set(HYLE_DEV_FACTORS "${_hyle_selection_input}" CACHE STRING', "top-level maintenance only; excludes embedding/install", "maintainer"),
    ):
        add("product_selection", name, "cmake/HyleComponents.cmake", anchor, audience, "build_system", "source producer", effect, "KEEP: central product/factor DAG, no duplicate route graph")

    for name, anchor in (("HYLE_STATIC", "option(HYLE_STATIC"), ("HYLE_INSTALL", "option(HYLE_INSTALL")):
        add("delivery_controls", name, "CMakeLists.txt", anchor, "ordinary", "build_system", "SDK producer", "HYLE linkage/install shape; not provider or operator coverage", "KEEP: three deliveries have the same route capability contract")

    for name, path, anchor in (
        ("HYLE_DEP_SOURCE_CACHE", "cmake/HyleDependencyCache.cmake", 'set(HYLE_DEP_SOURCE_CACHE "${_normalized}" CACHE PATH'),
        ("HYLE_DEP_BUILD_DIR", "CMakeLists.txt", 'set(HYLE_DEP_BUILD_DIR "${_hyle_dep_build_dir_normalized}" CACHE PATH'),
        ("HYLE_COMPUTE_TORCH_CPU_ROOT", "src/compute_provider_native/torch_cpu_fetch.cmake", 'set(HYLE_COMPUTE_TORCH_CPU_ROOT "" CACHE PATH'),
        ("HYLE_COMPUTE_TORCH_CPU_ARCHIVE", "src/compute_provider_native/torch_cpu_fetch.cmake", 'set(HYLE_COMPUTE_TORCH_CPU_ARCHIVE "" CACHE FILEPATH'),
        ("HYLE_COMPUTE_TORCH_CUDA_ARCHIVE", "src/compute_provider_torch_cuda/torch_cuda_fetch.cmake", 'set(HYLE_COMPUTE_TORCH_CUDA_ARCHIVE "" CACHE FILEPATH'),
        ("HYLE_COMPUTE_TORCH_CUDA_ARCHIVES_DIR", "src/compute_provider_torch_cuda/torch_cuda_linux.cmake", 'set(HYLE_COMPUTE_TORCH_CUDA_ARCHIVES_DIR "" CACHE PATH'),
    ):
        add("acquisition_inputs", name, path, anchor, "advanced_acquisition", "build_system", "online/offline producer", "source/binary location only, must satisfy locked identity", "KEEP: optional location override; not an OFF capability switch")

    for name, path, anchor, effect in (
        ("CMAKE_CUDA_ARCHITECTURES / CUDAARCHS", "cmake/HyleCudaDependencies.cmake", "if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)", "one standard architecture axis; default 80,89-real; binary admission is separate"),
        ("CUDAToolkit discovery", "cmake/HyleCudaDependencies.cmake", "find_package(CUDAToolkit 13.3 REQUIRED)", "standard Toolkit path/toolchain; no downloaded driver"),
        ("HYLE_VULKAN_GLSLANG / VULKAN_SDK", "cmake/HyleVulkanDependencies.cmake", "find_program(HYLE_VULKAN_GLSLANG", "host shader compiler; binary SDK must not require compiler"),
        ("CMAKE_OSX_DEPLOYMENT_TARGET", "CMakeLists.txt", "set(CMAKE_OSX_DEPLOYMENT_TARGET", "platform minimum and ABI; compiler/device support are distinct"),
        ("CC / CXX toolchain pair", "scripts/hyle-unix.sh", "Set both CC and CXX", "actual selected host compiler, never runtime performance search"),
        ("FETCHCONTENT_FULLY_DISCONNECTED", "cmake/HyleDependencyCache.cmake", "FETCHCONTENT_FULLY_DISCONNECTED", "no-network acquisition, selected missing dependencies must fail"),
        ("FETCHCONTENT_SOURCE_DIR_<LOCK_NAME>", "cmake/HyleDependencyVersions.cmake", "function(hyle_verify_locked_fetchcontent_override", "exact reviewed source substitution, not arbitrary system versions"),
    ):
        add("toolchain_and_offline_axes", name, path, anchor, "advanced_standard", "build_system", "source/binary producer", effect, "KEEP: standard axis, no second HYLE compiler/device option")

    for name, path, anchor, effect in (
        ("HYLE_BUILD_TESTS", "CMakeLists.txt", "option(HYLE_BUILD_TESTS", "top-level default ON; install requires OFF"),
        ("HYLE_BUILD_BENCH", "CMakeLists.txt", "option(HYLE_BUILD_BENCH", "default OFF; not ordinary product coverage"),
        ("HYLE_WERROR", "CMakeLists.txt", "option(HYLE_WERROR", "compiler diagnostics"),
        ("HYLE_SANITIZER", "CMakeLists.txt", 'set(HYLE_SANITIZER "off" CACHE STRING', "instrumented build identity; not measured winner recipe"),
        ("HYLE_TRACE", "cmake/HyleTrace.cmake", "option(HYLE_TRACE", "instrumentation identity; default OFF"),
        ("HYLE_SIMD_EXACT_TARGET", "CMakeLists.txt", 'set(HYLE_SIMD_EXACT_TARGET "${HYLE_SIMD_EXACT_TARGET}" CACHE STRING', "explicit ISA promise, alters qualified CPU recipe; keep SIMD itself"),
    ):
        add("maintenance_controls", name, path, anchor, "maintainer", "build_system", "developer build", effect, "KEEP: separate from ordinary complete compute preset")

    for name, path, anchor, effect, decision in (
        ("compute-cpu preset", "CMakePresets.json", '"name": "compute-cpu"', "CPU ON; GPU OFF; two provider AUTO knobs; build jobs 2", "REPLACE cache selection, preserve consumer convenience"),
        ("compute-cuda preset", "CMakePresets.json", '"name": "compute-cuda"', "CUDA ON; CPU OFF; no matching Apple/Vulkan ordinary presets", "REPLACE cache selection; add symmetric presets"),
        ("Windows toolchain wrapper", "scripts/hyle-windows.ps1", "param(", "Configure/Build public; smoke actions maintenance, forwards CMake args", "KEEP environment selection, do not invent route semantics here"),
        ("Unix toolchain wrapper", "scripts/hyle-unix.sh", "scripts/hyle-unix.sh [--] COMMAND", "forwards selected compiler, platform-local build dirs", "KEEP; same presets and CMake contract"),
        ("source embedding", "cmake/HyleComponents.cmake", "An embedding parent must set HYLE_COMPONENTS", "parent selection before add_subdirectory", "KEEP public hyle::compute/graph targets"),
        ("installed package discovery", "cmake/HyleConfig.cmake.in", "foreach(_Hyle_component IN LISTS Hyle_FIND_COMPONENTS)", "find_package Hyle COMPONENTS + exported feature metadata", "EXTEND route requirement check; do not rebuild or change SDK recipe"),
        ("generic runtime staging", "cmake/HyleRuntimeStaging.cmake", "function(hyle_stage_runtime_dependencies target)", "Windows DLL copy, Unix loader paths; linked owner closure", "COMPOSE under hyle_stage_runtime(app)"),
        ("Apple metallib staging", "cmake/HyleAppleRuntime.cmake", "function(hyle_stage_apple_runtime target)", "native resource adjacent to app; source and relocated SDK", "COMPOSE under the same helper, only for linked Apple route"),
    ):
        add("integration_workflows", name, path, anchor, "ordinary", "build_system", "C++ application integration", effect, decision)

    for route, direct_path, direct_anchor, graph_path, graph_anchor, effect in (
        ("cpu", "include/hyle/compute/cpu.h", "create_context(DefaultContextOptions", "include/hyle/graph/cpu.h", "create_context(compute::cpu::DefaultContextOptions", "common Context + prepare/compile; caller executor/native thread budget"),
        ("cuda", "include/hyle/compute/cuda.h", "create_context(DefaultContextOptions", "include/hyle/graph/cuda.h", "compile(const Graph &graph", "common Context/native stream and MemoryResource; cold replay/capture"),
        ("apple", "include/hyle/compute/apple.h", "static Result<Context> create()", "include/hyle/graph/apple.h", "compile(const compute::apple::Context", "MLX arrays/compiled region/owned native outputs; checked completion"),
        ("vulkan", "include/hyle/compute/vulkan.h", "create(ContextOptions options", "include/hyle/graph/vulkan.h", "compile(const compute::vulkan::Context", "native Tensor/Program/Invocation with buffers/commands/fences"),
    ):
        add("ordinary_route_roots", route + " Direct", direct_path, direct_anchor, "ordinary", "compute_route_" + route, "Direct operation/region consumer", effect, "KEEP native flow; no common queue required")
        add("ordinary_route_roots", route + " Graph", graph_path, graph_anchor, "ordinary", "graph_route_" + route, "typed Graph consumer", effect, "KEEP default Provider assembly; hybrid is a new explicit opt-in")

    for name, path, anchor, effect in (
        ("Operation schema / parameters", "include/hyle/compute/operation.h", "class Operation final", "actual numerical semantics; schema inference is not executable qualification"),
        ("TensorDesc / physical encoding", "include/hyle/compute/tensor_desc.h", "class HYLE_COMPUTE_OPS_API TensorDesc final", "shared descriptor/physical storage identity, every input/output port checked"),
        ("ModelArtifactProfile::create", "include/hyle/compute/workload.h", "class ModelArtifactProfile final", "owned immutable model/artifact identity; generic empty profile legal"),
        ("WorkloadContext", "include/hyle/compute/workload.h", "struct WorkloadContext final", "phase/versioned region; not model loader or global precision"),
        ("common compute::prepare", "include/hyle/compute/prepared_operation.h", "prepare(const Context &context, Operation operation,", "all-port operation/resource binding; overloads are one workflow"),
        ("common graph::compile", "include/hyle/graph/executable.h", "compile(const Graph &graph, CompileEnvironment environment,", "bounded region/planning/copy/reuse; exact or declared shape domain"),
        ("Invocation::prepare_replay", "include/hyle/graph/executable.h", "Status prepare_replay(std::span<const QueueBinding>", "cold queue/transfer/completion binding; submit uses fixed result"),
        ("native ResultLease / OwnedResult", "include/hyle/compute/result.h", "class [[nodiscard]] ResultLease final", "ready successful native ownership and terminal completion, not implicit host copy"),
    ):
        add("shared_semantic_and_lifecycle_entries", name, path, anchor, "ordinary_or_integration", "compute" if "compute/" in path else "graph", "Direct/Graph prepared consumer", effect, "KEEP existing semantics; reuse, do not duplicate selector/profile/result")

    for name, path, anchor, owner, effect in (
        ("Provider factories", "include/hyle/compute/providers/native.h", "namespace hyle::compute::native", "compute/provider factors", "expert native/ATen/oneDNN/ggml/cuBLAS/FlashInfer/SGLang/DeepSeek/CCCL/Torch construction; ordinary route assembles them"),
        ("Custom kernel extension", "include/hyle/compute/custom_kernel.h", "make_custom_kernel(Kernel", "compute_ops", "user pure/native execution extension, not a runtime timing policy"),
        ("Direct delegated preparation", "include/hyle/compute/delegated_operation.h", "prepare_delegated(const Context &context", "compute", "native private resources/output ownership; no Graph requirement"),
        ("Apple region callable", "include/hyle/compute/apple.h", "using RegionFunction", "compute_route_apple", "complete MLX compiled region through Context::compile; preserve mature native scheduling"),
        ("CUDA resource/stream interop", "include/hyle/compute/backends/cuda.h", "borrow_stream(const Context", "compute_backend_cuda", "exact Context native stream, device/pinned/managed resource, actual completion"),
        ("Graph compile budgets / replay constraints", "include/hyle/graph/executable.h", "struct CompileOptions final", "graph", "bounded proposals/search/resource/concurrency/capture requirements, not timing profiles"),
        ("ContextOptions::providers / ProviderBinding", "include/hyle/compute/device.h", "struct ProviderBinding final", "compute", "expert custom integrations can give explicit families/preferences; ordinary facade owns ranking"),
        ("CompileEnvironment targets/storage/lanes", "include/hyle/graph/executable.h", "struct CompileEnvironment final", "graph", "one exact Device; not CPU+GPU execution"),
        ("node/value placement + external contracts", "include/hyle/graph/executable.h", "struct NodePlacement final", "graph", "explicit preferred/required placement independent of Provider selection"),
        ("allowed/denied/node Provider constraints", "include/hyle/graph/executable.h", "std::span<const compute::ProviderId> allowed_providers", "graph", "hard capability restrictions; not ordinary provider sorting"),
        ("RegionProvider extension", "include/hyle/graph/region_provider.h", "struct RegionProviderVTable final", "graph", "whole region proposal/artifact; protect mature fusion"),
        ("match_route_selector / match_route_policy", "include/hyle/compute/route_policy.h", "RouteMatchResult match_route_selector(", "compute", "public allocation-free cold selector semantics; no timing winner search"),
        ("compute::copy_async / transfer_properties", "include/hyle/compute/transfer.h", "transfer_properties(const Queue", "compute", "actual Queue/resource pair, unsupported errors, no fake async staging"),
        ("Vulkan native borrow/queue", "include/hyle/compute/vulkan.h", "borrow_ready(const TensorDesc", "compute_route_vulkan", "exact VkDevice/queue-family ready data; owner retained; no pending semaphore bridge"),
        ("Apple custom lowering", "include/hyle/graph/apple.h", "using CustomLowering", "graph_route_apple", "cold trace in native runtime/stream; not generic CPU interop"),
    ):
        add("expert_and_extension_entries", name, path, anchor, "expert", owner, "custom Provider/native interop integration", effect, "KEEP hard constraints/native extensions; no new ordinary ranking control")

    for name, path, anchor, owner, effect in (
        ("CPU default policy", "src/compute_route_cpu/default_policy.cc", "Family baseline(Operation operation)", "compute_route_cpu", "mature defaults + separately qualified exact Native overrides"),
        ("CUDA default policy", "src/compute_route_cuda/default_policy.cc", "Family baseline(const ProviderOperationRequest", "compute_route_cuda", "hard-admitted mature choices with per-op exact qualified overrides"),
        ("Apple mature policy match", "src/compute_route_apple/apple_operations.cc", "match_route_selector(kAffineQ4Rule", "compute_route_apple", "ggml/MLX native program boundary"),
        ("Vulkan mature policy match", "src/compute_route_vulkan/vulkan_policy.h", "match_route_selector(selector", "compute_route_vulkan", "ggml/IREE engineering default; not a formal performance WIN"),
        ("shared ggml SDK source recipe", "src/compute_ggml_runtime/CMakeLists.txt", "if(_cpu AND NOT HYLE_COMPUTE_CUDA_ENABLED", "compute_ggml_runtime", "CPU-only newer pin vs mixed SDK GPU-qualified shared pin; changes software identity"),
        ("Graph same-Device guard", "src/graph/executable.cc", "target.context->device() != environment.targets[0].context->device()", "graph", "CPU+GPU rejected; must add native fragment composition, not remove guard blindly"),
    ):
        add("internal_binding_seams", name, path, anchor, "internal", owner, "default assembly/compiler", effect, "DESIGN input; reopen only finite implementation scope with new card")

    key_pairs = [(entry["group"], entry["name"]) for entry in entries]
    if len(set(key_pairs)) != len(entries):
        raise ValueError("Duplicate inventory identity")
    payload = dict(
        schema="hyle.compute-entry-design.inventory.v1",
        baseline=BASELINE,
        method="Unique configuration control or integration/route lifecycle family. Overloads, aliases, references, individual operator descriptors, derived vars/macros and internal dependency targets are not extra entries. Toolchain/offline groups name standard axes rather than every CMake spelling. This is the design perimeter, not a symbol count or implementation certificate.",
        counts=dict(total=len(entries), by_group=dict(sorted(Counter(e["group"] for e in entries).items())),
                    by_audience=dict(sorted(Counter(e["audience"] for e in entries).items()))),
        entries=entries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
