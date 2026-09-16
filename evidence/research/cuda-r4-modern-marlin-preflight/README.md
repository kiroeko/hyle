# R4 modern Marlin native preflight

This is a bounded adapter feasibility result, **not a HYLE CUDA route acceptance or performance claim**.
Source: vLLM `51918e252af4dfff381001f42b7ebe6a0454501b`, Apache-2.0; the bundle retains its license,
source hashes, modified scratch headers, exact driver, compiler logs and output. No reference checkout was changed.

Windows RTX 4070 SM89, CUDA 13.3.73 / MSVC 19.50: symmetric W4A16, group128, K=N=1024,
M=1/7/17/64, three changed-input replays each. All 12 comparisons passed against host double accumulation;
maximum absolute error 0.001953125. Resident weights are 524,288 packed bytes plus 16,384 scale bytes.
GPTQ-to-Marlin packing and scale permutation occur before replay. This probe uses native CUDA allocation/stream;
HYLE admission, Graph, two-slot lifetime, delivery and P1 are still unverified for the new provider.

Native adaptation isolates the upstream Torch exception include for static scalar descriptors, replaces the
Marlin dtype template's `long` with its real `ScalarTypeId` for Windows LLP64, and takes the unchanged GPTQ
repack device kernel without the Torch host wrapper. Production adaptation must preserve fallible HYLE boundaries;
the scratch throwing check is not a proposed product error contract. The first compile missed upstream's
`--expt-relaxed-constexpr`; one correction restored it and specified UTF-8 SDK input. No kernel arithmetic changed.

The executable linked the CUDA runtime and CRT statically; `dumpbin /dependents` reported only KERNEL32.dll.
The parent disabled interactive loader errors before launch. No additional runtime DLL was missing or required.

Files: `preflight.zip`, `result.json`, `sha256.json`. The ZIP also contains the unified LOC report
(Compute + Graph implementation, excluding tests: **31,021**, self-check PASS) and inactive-identity projection
proof. No product implementation growth or R4 completion is claimed from these results.
