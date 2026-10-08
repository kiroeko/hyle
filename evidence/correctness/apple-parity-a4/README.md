# Apple parity: public source/static/shared delivery sample

Consumer candidate `8e95478c822326b0ddcb3ecb43acb511b8bef7f6`,
Windows canonical, complete Git bundle forwarded to MacBook Air Mac16,12.
Air M4/macOS27.0.1/AppleClang21/SDK27.0, deployment14.4, AC and lowpower0.
This is a targeted current delivery witness, **not whole Apple goal closure**.
Official FP4 and two official DSpark numerical witnesses remain open.

The same [public consumer](../../../tests/compute_route_apple/parity_delivery/main.cc)
and [CMake](../../../tests/compute_route_apple/parity_delivery/CMakeLists.txt) link
only `hyle::graph` and stage the Metal resource through the public helper.
There are no private HYLE/vendor include paths, mock providers or CPU model
fallbacks. Both ordinary Direct and Graph execute nonzero original Q4_K
MatMul with two changed-input replays, exact selected rows, checked index
failure and retained results after close, plus original mature NF4 decode.
This SDK sample complements owning full numerical matrices; it does not
rerun559/34/45 or certify untested official FP4.

| Delivery | Authoritative successful evidence |
|---|---|
| shared relocated SDK | [receipt](receipt-shared-confirmed.json), [configure](shared-consumer-configure.log), [build](shared-consumer-build.log), [runtime](shared-runtime.log), [commands](shared-commands.log) |
| embedded source | [receipt](receipt.json), [configure](source-configure.log), [build](source-build.log), [runtime](source-runtime.log), [commands](source-commands.log) |
| static relocated SDK | [receipt](receipt.json), [install](static-install.log), [configure](static-consumer-configure.log), [build](static-consumer-build.log), [runtime](static-runtime.log), [commands](static-commands.log) |

Both installed SDKs are moved to independent prefixes before consumer
configure. Configure uses disconnected FetchContent and unreachable local
proxy endpoints; no SDK source/cache download is needed. Their compiler/link
commands exclude canonical include/src and producer paths. Runtime hides
producer bin and all producer Metal libraries, clears inherited DYLD settings,
uses the staged Metal resource and records loaded dylibs. All hidden assets
are restored in finally. The SDK and staged `mlx.metallib` hashes match:
`2ae3af693d8e941861063c9f3ae5249e17665e8761837a9d87ab6e07002f568c`.
All new NF4/AWQ/CT/Q4_K/IQ1 license/source notices are present in the receipts.
The [task-local batch driver](run.py) is recorded; public governance/performance
tools were not rewritten.

Shared producer was already built from unchanged current product sources.
The consumer-only commits after `a8809428` do not change include/src/cmake/
third_party/root CMake inputs, so the installed SDK is reused for those
corrections; no repeat producer build or install. Source embedding reuses the
existing source producer; its actual incremental build took58.95s, installed
static consumer build0.96s. Heavy work stayed serial on Air, build2.

Preserved failures and one corresponding correction per named issue:

- [tests-on configure](shared-configure-tests-on.log): shipping install requires
  tests/instrumentation OFF. [Shipping configure](shared-configure.log) and
  [producer build](shared-build.log)/[install](shared-install.log) then pass.
- [first consumer compile](shared-consumer-build-first.log): missing public MLX
  transforms declaration and explicit Operation construction. Corrected only
  the consumer; SDK untouched.
- [first runtime](shared-runtime-dimensions-first.log): consumer reversed public
  GGUF K/N parameter order; no operator ran. Corrected only that argument pair.
- [source compiler refresh](source-configure-compiler-refresh.log): CMake changed
  compiler paths and refreshed the old cache, clearing first-call variables.
  One same-argument configure on the refreshed cache passed; no product fix.

These fixture/tooling failures remain evidence; none is mislabeled a product
math failure or counted as a passing whole model. A4 full header/boundary
matrix/final cleanup and A3 default/cost verdicts still need their own evidence.

## Directed CPU + Metal shared-core compatibility

Candidate21bcb41f, same current product inputs. [Receipt](mixed-core-21bcb41f-receipt.json),
[configure](mixed-core-21bcb41f-configure.log), [build](mixed-core-21bcb41f-build.log),
[actual commands](mixed-core-21bcb41f-commands.log), [CPU](mixed-core-21bcb41f-cpu.log)/
[XML](mixed-core-21bcb41f-cpu.xml)9/9 PASS and
[Metal](mixed-core-21bcb41f-metal.log)/[XML](mixed-core-21bcb41f-metal.xml)8/8 PASS.
The task-local [driver](run-mixed-core.py) selects only runtime+ggml Metal
maintainer factors, CPU/Apple ON, other GPUs OFF, shared/instrumentation OFF,
build2 and serial tests. No CPU product code/review or P1 was reopened.

The six actual selected header/core/quant/Metal source hashes exactly match
Apple-only. CPU tests execute original three narrow IQ1 MatMul/GetRows and
metadata/workspace/lifetimes; Metal tests execute original native packed
MatMul/grouped/recurrent programs and checked resource boundaries. The new
Metal IQ1 actual eight contracts retain earlier matching-source numerical
evidence, while mixed compilation and native admission are confirmed here.
This is directed source/backend compatibility, not whole mixed-model support.
The GgmlVulkanMetadata suite name refers only to shared arena tests; no Vulkan
GPU verification is claimed.

The [first configure](mixed-core-21bcb41f-configure-first.log) succeeded;
the script then rejected multiple cached source directories before compiling.
Source identity now comes from actual ggml-base Ninja commands, not every
old cache directory. This task-local correction did not change product input.

## Explicit missing-resource failure and corrected initialization

One missing-resource batch on the existing three consumer binaries safely
returned exit1 but concealed the cause as "MLX Metal GPU is unavailable":
[source](missing-resource-first-source.log), [static](missing-resource-first-static.log),
[shared](missing-resource-first-shared.log), [receipt](missing-resource-first-receipt.json).
Original MLX gpu::device_count catches the missing-metallib exception and
returns0. HYLE's initial is_available call therefore hid the actual error.
All hidden resources were restored with their exact hashes; no system setting
or mathematical algorithm was changed.

Finite F5 correction `4c3a0b53` reads the already required GPU Environment
first on the same cold owner, preserving the original native initialization
error at its existing fallible boundary, then creates the owned stream.
No extra path scan, caller timing or CPU fallback is introduced. The public
API and hot program are unchanged. [Batch receipt](context-error-4c3a0b53-receipt.json)
and [task-local driver](run-context-error-fix.py) record targeted Context9/9
[tests](context-error-4c3a0b53-context-tests.log)/[XML](context-error-4c3a0b53-context.xml),
shipping shared/source producer incremental builds and SDK updates.

The same consumer's corrected [source](context-error-4c3a0b53-source-positive.log),
[static](context-error-4c3a0b53-static-positive.log),
[shared](context-error-4c3a0b53-shared-positive.log) successful Direct/Graph
math/resources remain PASS. Only this initialization change justified these
three targeted journeys; unrelated model matrices and P1 were not repeated.

Corrected fresh-process resource negatives now clearly report failure to load
the metallib, exit1 in all three deliveries:
[source](missing-resource-source.log), [static](missing-resource-static.log),
[shared](missing-resource-shared.log), [receipt](missing-resource-receipt.json).
[Driver](run-missing-resource.py) hides every app/SDK/producer mlx.metallib
inside the owned task directories, clears inherited DYLD, and restores all
files in finally. All original asset hashes match after restoration. The
negative verdict is3/3; it is not a missing-device or wrong-model diagnosis.

## Updated FP4 public deliveries

`326a40ee` [driver](run-fp4-delivery.py)/
[receipt](fp4-delivery-326a40ee-receipt.json): cold Direct1/1, Graph1/1 and
source/static/shared public consumers all PASS. The consumer only links
hyle::graph. It now includes tiny raw BF16/signed zeros, original FP4 experts
with FP8 activation, duplicate IDs, changed-input replay, negative/upper
invalid IDs, failure recovery in fresh slots and retained outputs after close.
Both relocated SDKs configure offline; producer binaries/metallibs are
hidden at runtime and restored in finally. Required original DS4, DeepSeek,
TileLang/CUTLASS licenses are present. The first driver expected incorrect
filenames/nonexistent NOTICE files; only that assertion was corrected, and
completed build/cold/static configuration steps were reused. No product fix
or repeated model matrix resulted. Old missing-resource tests still apply.

Final directed Apple-OFF Windows configure also PASS:
[environment-corrected log](win-off-fp4-configure.log). No MLX/Metal probe or
download occurs. The [first call](win-off-fp4-configure-first.log) lacked the
MSVC standard-library environment; using the existing Windows entry fixed
only that process environment. No Windows full build/model test was run.

The final source review rechecked new FP4 finite-BF16 inference guards,
canonical carrier extents/selected-expert decode, checked completion status,
old-result lifetime, shared original core/route isolation, and delivery
resource ownership. No further product FIX arose. The user coverage page
still described the D5 FP4 rejection as current; it now explicitly separates
that historical snapshot from current559/34/45 and official FP4/MTP evidence.
No CPU/CUDA/Vulkan state or old FAIL was changed. Performance remains a
separate gate; these reviews do not sign an unrun P1.

P1 is now separately qualified PASS in the
[final performance record](../../performance/apple-parity-overlap-p1/README.md).
Task-owned cleanup [Windows script](cleanup-owned-temporaries.ps1)/
[receipt](cleanup-windows.json) removed21,183,219bytes of consumed transfer
bundles and the temporary Windows MLX reference. The exact rejected Air
producer [script](cleanup-failed-producer.py)/[receipt](cleanup-air.json)
removed98,771,171bytes after its complete raw archive was copied to Windows.
Successful producer/SDKs, model fixtures, source cache and qualified MLX peer
remain. Complete canonical raw copies retain every observation; duplicate
compressed transfer packages were removed after extraction.
